"""Exercise real Chronos-2 internals with random tiny weights; no HF download.

Requires the optional chronos-forecasting==2.2.2 environment. The core CT-SSF
tests work without Chronos. These tests verify the actual export hooks/API.
"""

import pickle
import numpy as np
import pytest
import torch

chronos = pytest.importorskip("chronos")
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model

from base_predictor.chronos_features import ChronosFeatureCapture, export_chronos_head, load_chronos_head
from base_predictor.chronos_predictor import ChronosPredictor
from baselines.ct_ssf.data import prepare_sequence
from baselines.ct_ssf.run_ct_ssf import evaluate_sequence


def tiny_pipeline(arcsinh=True, patch_size=16):
    torch.manual_seed(7)
    config = Chronos2CoreConfig(
        d_model=8, d_ff=16, d_kv=4, num_heads=2, num_layers=1, dropout_rate=0,
        chronos_config=dict(context_length=128, input_patch_size=patch_size, input_patch_stride=patch_size,
                            output_patch_size=patch_size, max_output_patches=4, quantiles=[i / 10 for i in range(1, 10)],
                            use_reg_token=True, use_arcsinh=arcsinh),
    )
    return chronos.Chronos2Pipeline(Chronos2Model(config).eval())


@pytest.mark.parametrize("covariates,arcsinh,horizon,patch_size", [(0, True, 10, 16), (2, True, 10, 16), (2, False, 6, 4)])
def test_export_reconstructs_real_chronos_and_preserves_forecasts(tmp_path, covariates, arcsinh, horizon, patch_size):
    torch.set_num_threads(1)
    pipeline = tiny_pipeline(arcsinh, patch_size)
    rng = np.random.default_rng(4)
    n, window = 133, 100
    data = {"series": {"x": rng.normal(size=(n, covariates)).astype("float32"), "y": rng.normal(5, 2, n).astype("float32")}}
    predictor = ChronosPredictor.__new__(ChronosPredictor)
    predictor.chronos2, predictor.data, predictor.data_type = pipeline, data, "toy"
    predictor.predictions, predictor.model_revision = {}, None
    predictor.predict(window, horizon)
    ordinary = np.asarray(predictor.predictions["series"]).copy()
    predictor.predict(window, horizon, save_ctssf=True)
    np.testing.assert_array_equal(predictor.predictions["series"], ordinary)
    predictor.save(tmp_path)
    with (tmp_path / "chronos_toy_data.pkl").open("rb") as stream:
        saved = pickle.load(stream)["series"]
    head, artifact = load_chronos_head(tmp_path / "chronos_toy_ctssf_head.pt")
    prepared = prepare_sequence(saved, head, artifact, {"train_ratio": 0.4, "valid_ratio": 0.1, "calibration_ratio": 0.16, "test_ratio": 0.34})
    np.testing.assert_array_equal(prepared["raw_target_indices"], np.arange(window, n))
    assert prepared["lead"][-1] == ((n - window - 1) % horizon) + 1
    assert saved["ctssf"]["features"].shape[0] < len(ordinary)
    assert len(pipeline.model.output_patch_embedding._forward_pre_hooks) == 0
    assert len(pipeline.model.instance_norm._forward_hooks) == 0
    config = {
        "seed": 7,
        "data": {"train_ratio": 0.4, "valid_ratio": 0.1, "calibration_ratio": 0.16, "test_ratio": 0.34},
        "model": {"target_quantiles": [[0.05, 0.95]], "inverse_steps": 10, "batch_size": 16},
        "attention": {"epochs": 2, "embedding_dim": 4, "memory_size": 4},
    }
    result = evaluate_sequence("series", saved, config, head, artifact, checkpoint_dir=tmp_path)
    assert result["metadata"]["attention_training"]["objective"] == "semantic_error_similarity_kl"
    assert result["metadata"]["attention_training"]["semantic_error_dim"] == 8
    assert (tmp_path / result["metadata"]["attention_checkpoint"]).is_file()
    interval = result["evaluation_results"][(0.05, 0.95)]
    start = prepared["boundaries"]["test_start"]
    np.testing.assert_array_equal(interval["target_y"], prepared["y"][start:])
    assert interval["avg_coverage"] == np.mean(interval["coverage"])
    assert not np.isnan(interval["lower_interval"]).any()
    assert not np.isnan(interval["upper_interval"]).any()


def test_capture_removes_hooks_on_exception_and_rejects_unsupported_head():
    pipeline = tiny_pipeline()
    capture = ChronosFeatureCapture(pipeline, export_chronos_head(pipeline.model))
    with pytest.raises(RuntimeError):
        with capture:
            raise RuntimeError("failed prediction")
    assert not pipeline.model.output_patch_embedding._forward_pre_hooks
    assert not pipeline.model.instance_norm._forward_hooks
    pipeline.model.config.dense_act_fn = "gelu"
    with pytest.raises(ValueError, match="ReLU"):
        export_chronos_head(pipeline.model)
