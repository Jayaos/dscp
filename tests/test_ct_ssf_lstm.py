import copy
import itertools
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from base_predictor import lstm_predictor
from baselines.ct_ssf.bounds import HeadBounder
from baselines.ct_ssf.data import prepare_sequence
from baselines.ct_ssf.lstm import LSTMLinearHead, attach_lstm_features, load_lstm_backbone
from baselines.ct_ssf.run_ct_ssf import evaluate_sequence
from baselines.ct_ssf.scores import latent_scores
from sbatch.sbatch_run_ct_ssf.run_ct_ssf import build_parser, main, resolve_config


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def saved_lstm(tmp_path):
    data = SimpleNamespace(data_type="toy", data={})
    # Unequal lengths, nonzero response location/scale, and a constant
    # covariate exercise the exact preprocessing stored by --save-model.
    for key, length, offset in (("a", 96, 3.0), ("b", 111, 5.0)):
        t = np.arange(length, dtype=np.float64)
        data.data[key] = {"x": np.column_stack((t / 20, np.ones_like(t))),
                          "y": offset + np.sin(t / 4)}
    torch.manual_seed(17)
    with patch.object(lstm_predictor, "tqdm", side_effect=lambda values: values):
        predictor = lstm_predictor.LSTMPredictor(data, 3, 4, 2, 0.4, 5)
        predictor.fit_predict(0.75, 8, 0.001, 2, 1, device="cpu")
        predictor.save(tmp_path, save_model=True)
    model_path, data_path = tmp_path / "lstm_toy_model.pt", tmp_path / "lstm_toy_data.pkl"
    with data_path.open("rb") as stream:
        records = pickle.load(stream)
    model, head, checkpoint, artifact = load_lstm_backbone(model_path)
    config = {
        "base_predictor": "lstm", "seed": 2, "device": "cpu", "num_threads": 1,
        "data": {"data_path": str(data_path), "model_path": str(model_path),
                 "train_ratio": 0.4, "valid_ratio": 0.1, "calibration_ratio": 0.16, "test_ratio": 0.34},
        "model": {"target_quantiles": [[0.05, 0.95]], "inverse_steps": 20,
                  "inverse_lr": 0.01, "inverse_optimizer": "adam", "bound_method": "linear_exact",
                  "calibration_size": 15, "batch_size": 8},
        "attention": {"epochs": 2, "patience": 1, "embedding_dim": 4, "memory_size": 8},
        "saving_dir": str(tmp_path / "results"),
    }
    return model, head, checkpoint, artifact, records, config


def test_features_match_original_windows_predictions_and_batching(saved_lstm):
    model, head, checkpoint, artifact, records, config = saved_lstm
    stats = checkpoint["normalization"]
    assert stats["x_std"][1] == 0
    for item in records.values():
        attached = attach_lstm_features(item, model, checkpoint, artifact, batch_size=7)
        prepared = prepare_sequence(attached, head, artifact, config["data"])
        one_by_one = attach_lstm_features(item, model, checkpoint, artifact, batch_size=1)
        np.testing.assert_allclose(attached["ctssf"]["features"], one_by_one["ctssf"]["features"], atol=2e-7, rtol=1e-5)
        normalized = {
            name: (item[name] - stats[f"{kind}_mean"].numpy()) / (stats[f"{kind}_std"].numpy() + stats["epsilon"])
            for name, kind in (("train_x", "x"), ("heldout_x", "x"), ("train_y", "y"), ("heldout_y", "y"))
        }
        windows, _ = lstm_predictor._make_heldout_sequence_prediction_data(**normalized, k=5)
        with torch.no_grad():
            expected = model.encode(torch.from_numpy(windows).float())[:, -1, :].numpy()
        np.testing.assert_allclose(prepared["features"], expected, atol=2e-7, rtol=1e-5)
        n, start = len(item["heldout_y"]), len(item["train_y"])
        np.testing.assert_array_equal(prepared["raw_target_indices"], np.arange(start, start + n))
        np.testing.assert_array_equal(prepared["lead"], np.ones(n))
        np.testing.assert_array_equal(prepared["feature_indices"], np.arange(n))
        assert "ctssf" not in item
        assert attached["heldout_predictions"] is item["heldout_predictions"]
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert all(p.grad is None for p in model.parameters())


def test_rejects_mismatched_forecasts_and_checkpoint_metadata(saved_lstm, tmp_path):
    model, head, checkpoint, artifact, records, config = saved_lstm
    item = attach_lstm_features(records["a"], model, checkpoint, artifact)
    item["heldout_predictions"] = item["heldout_predictions"].copy() + 1
    with pytest.raises(ValueError, match="reconstruct"):
        prepare_sequence(item, head, artifact, config["data"])
    item = attach_lstm_features(records["a"], model, checkpoint, artifact)
    item["ctssf"]["checkpoint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_sequence(item, head, artifact, config["data"])
    changed = copy.deepcopy(checkpoint)
    changed["metadata"]["input_order"] = ["lagged_target", "covariates"]
    path = tmp_path / "invalid.pt"
    torch.save(changed, path)
    with pytest.raises(ValueError, match="input_order"):
        load_lstm_backbone(path)
    changed = copy.deepcopy(checkpoint)
    changed["normalization"]["y_std"] *= -1
    torch.save(changed, path)
    with pytest.raises(ValueError, match="nonnegative"):
        load_lstm_backbone(path)


@pytest.mark.parametrize("method", ["linear_exact", "ibp", "crown"])
def test_linear_bounds_match_corners_and_preserve_infinite_radius(method):
    layer = torch.nn.Linear(2, 1)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, -3.0]]))
        layer.bias.fill_(0.4)
    head = LSTMLinearHead(layer).eval().requires_grad_(False)
    bounder = HeadBounder(head, method)
    assert bounder.method == "linear_exact"
    z = torch.tensor([[1.0, 2.0], [-2.0, 0.5]])
    lo, hi = bounder(z, [0, 0], [0.2, 0.0], [5.0, -3.0], [2.0, 0.3])
    corners = torch.tensor(list(itertools.product([-0.2, 0.2], repeat=2)), dtype=torch.float64)
    values = copy.deepcopy(head).double()(z[:1].double() + corners)[:, 0] * 2 + 5
    assert lo[0].item() == pytest.approx(values.min().item(), abs=1e-7)
    assert hi[0].item() == pytest.approx(values.max().item(), abs=1e-7)
    assert lo[1].item() == pytest.approx(hi[1].item(), abs=1e-7)
    lo, hi = bounder(z, [0, 0], [float("inf"), 0], 0, 1)
    assert torch.isneginf(lo[0]) and torch.isposinf(hi[0])


def test_future_observations_cannot_change_own_feature_or_interval(saved_lstm):
    model, head, checkpoint, artifact, records, config = saved_lstm
    item = records["a"]
    original = attach_lstm_features(item, model, checkpoint, artifact)
    changed = copy.deepcopy(item)
    t = 45  # Beyond attention training/validation, inside the test suffix.
    changed["heldout_y"][t:] += 3
    changed["heldout_x"][t:, 0] += 3
    changed = attach_lstm_features(changed, model, checkpoint, artifact)
    np.testing.assert_allclose(original["ctssf"]["features"][:t + 1], changed["ctssf"]["features"][:t + 1], atol=0, rtol=0)
    assert not np.allclose(original["ctssf"]["features"][t + 1:], changed["ctssf"]["features"][t + 1:])
    # Recompute forecasts for the altered history so the pairing guard passes.
    # They may change only after t, just like the extracted hidden states.
    with torch.no_grad():
        changed["heldout_predictions"] = head.to_response(head(torch.from_numpy(changed["ctssf"]["features"])),
                                                          changed["ctssf"]["location"][0], changed["ctssf"]["scale"][0]).numpy()
    before = evaluate_sequence("a", original, config, head, artifact)
    after = evaluate_sequence("a", changed, config, head, artifact)
    count = t - before["metadata"]["boundaries"]["test_start"] + 1
    for field in ("lower_interval", "upper_interval", "latent_radius"):
        np.testing.assert_array_equal(before["evaluation_results"][(0.05, 0.95)][field][:count], after["evaluation_results"][(0.05, 0.95)][field][:count])
    assert before["metadata"]["attention_training"] == after["metadata"]["attention_training"]


def test_lstm_inversion_keeps_predictor_and_head_fixed(saved_lstm):
    model, head, checkpoint, artifact, records, config = saved_lstm
    item = attach_lstm_features(records["a"], model, checkpoint, artifact)
    data = prepare_sequence(item, head, artifact, config["data"])
    original = {name: value.clone() for name, value in model.state_dict().items()}
    args = (head, data["features"][:12], data["output_indices"][:12], data["y"][:12], data["location"][:12], data["scale"][:12])
    first, _ = latent_scores(*args, steps=10, optimizer="sgd", batch_size=1)
    second, _ = latent_scores(*args, steps=10, optimizer="sgd", batch_size=7)
    np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-7)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    for name, value in head.state_dict().items():
        torch.testing.assert_close(value, artifact["state_dict"][name], rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in head.parameters())


def test_saved_lstm_runs_end_to_end_through_cli(saved_lstm, tmp_path):
    _, _, _, artifact, records, config = saved_lstm
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), config_path)
    model_path, data_path = (config["data"][field] for field in ("model_path", "data_path"))
    original_bytes = {path: Path(path).read_bytes() for path in (model_path, data_path)}
    log = main([str(config_path)])
    output = tmp_path / "results"
    for filename in ("launch_config.yaml", "resolved_config.yaml", "log.pkl", "summary_results.pkl"):
        assert (output / filename).is_file()
    for key, result in log.items():
        metadata = result["metadata"]
        assert metadata["method"] == "ct_ssf_lstm_adapted"
        assert metadata["base_predictor"] == "lstm"
        assert metadata["bound_method"] == "linear_exact"
        assert metadata["base_checkpoint_sha256"] == artifact["checkpoint_sha256"]
        attention = torch.load(output / metadata["attention_checkpoint"], weights_only=True)
        assert attention["base_checkpoint_sha256"] == artifact["checkpoint_sha256"]
        assert attention["objective"] == "semantic_error_similarity_kl"
        evaluation = result["evaluation_results"][(0.05, 0.95)]
        start, n = metadata["boundaries"]["test_start"], len(records[key]["heldout_y"])
        assert evaluation["target_indices"] == list(range(start, n))
        np.testing.assert_array_equal(evaluation["target_y"], records[key]["heldout_y"][start:, 0])
        assert evaluation["avg_coverage"] == np.mean(evaluation["coverage"])
        assert not np.isnan(evaluation["interval_width"]).any()
    for path, expected in original_bytes.items():
        with open(path, "rb") as stream:
            assert stream.read() == expected


@pytest.mark.parametrize("dataset", ["air", "solar", "sapflux"])
def test_lstm_presets_and_path_overrides(dataset, tmp_path):
    args = build_parser().parse_args(["--dataset", dataset, "--base-predictor", "lstm", "--dry-run"])
    config = resolve_config(args)
    assert config.base_predictor == "lstm"
    assert "lstm_model_saved" in config.data.data_path
    assert config.data.model_path.replace("_model.pt", "_data.pkl") == config.data.data_path
    assert config.model.bound_method == "linear_exact"
    args.model_path = tmp_path / "alternative.pt"
    assert resolve_config(args).data.model_path == str(args.model_path.resolve())
    args.head_path = tmp_path / "head.pt"
    with pytest.raises(ValueError, match="incompatible"):
        resolve_config(args)


def test_yaml_predictor_cannot_be_overridden(saved_lstm, tmp_path):
    config = saved_lstm[-1]
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    with pytest.raises(ValueError, match="must match"):
        resolve_config(build_parser().parse_args([str(path), "--base-predictor", "chronos"]))
