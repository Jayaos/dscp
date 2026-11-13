import copy
import itertools
import pickle

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from base_predictor.chronos_features import ChronosMedianHead, head_fingerprint, load_chronos_head
from baselines.ct_ssf.attention import fit_attention
from baselines.ct_ssf.bounds import HeadBounder
from baselines.ct_ssf.data import prepare_sequence
from baselines.ct_ssf.run_ct_ssf import evaluate_sequence, run_ct_ssf
from baselines.ct_ssf.scores import latent_scores, weighted_quantile
from sbatch.sbatch_run_ct_ssf.run_ct_ssf import build_parser, resolve_config


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(n=100):
    torch.manual_seed(17)
    architecture = dict(input_dim=3, hidden_dim=5, output_patch_size=2, use_arcsinh=True)
    head = ChronosMedianHead(**architecture).eval().requires_grad_(False)
    artifact = {
        "format_version": 1, "architecture": architecture,
        "state_dict": {k: v.clone() for k, v in head.state_dict().items()},
        "provenance": {"window_length": 3, "prediction_length": 2, "uses_future_covariates": False},
    }
    artifact["head_sha256"] = head_fingerprint(artifact)
    features = torch.randn((n + 1) // 2, 3)
    mapping = np.arange(n) // 2
    offsets = np.arange(n) % 2
    location, scale = np.repeat(np.arange((n + 1) // 2) / 10, 2)[:n].astype("float32"), np.full(n, 2.0, dtype="float32")
    pred = head.to_response(head.select(features[mapping], torch.as_tensor(offsets)), torch.from_numpy(location), torch.from_numpy(scale)).numpy()
    y = pred + np.sin(np.arange(n)).astype("float32") * 0.2
    cache = {
        "format_version": 1, "head_sha256": artifact["head_sha256"], "features": features.numpy(),
        "feature_indices": mapping, "output_indices": offsets, "location": location, "scale": scale,
        "forecast_origin": 3 + mapping * 2, "lead": offsets + 1, "raw_target_indices": np.arange(n) + 3,
    }
    item = {"heldout_y": y, "heldout_predictions": pred, "heldout_x": np.zeros((n, 1)), "ctssf": cache}
    config = {
        "seed": 2, "device": "cpu", "num_threads": 1,
        "data": {"train_ratio": 0.4, "valid_ratio": 0.1, "calibration_ratio": 0.16, "test_ratio": 0.34},
        "model": {"target_quantiles": [[0.05, 0.95]], "inverse_steps": 25, "inverse_lr": 0.01, "inverse_optimizer": "adam", "batch_size": 32, "calibration_size": 20, "rolling_calibration": True, "weighting": "learned_attention", "bound_method": "crown"},
        "attention": {"epochs": 2, "patience": 1, "memory_size": 8, "batch_size": 16, "embedding_dim": 4},
    }
    return head, artifact, item, config


@pytest.mark.parametrize("method", ["ibp", "crown"])
@pytest.mark.parametrize("arcsinh", [False, True])
def test_bounds_contain_dense_grid_and_corners(method, arcsinh):
    torch.manual_seed(5)
    head = ChronosMedianHead(2, 5, 2, arcsinh).eval().requires_grad_(False)
    z = torch.tensor([[0.3, -0.2], [-0.2, 0.6]])
    lo, hi = HeadBounder(head, method)(z, [0, 1], [0.8, 0.4], [2, -4], [0.2, 3])
    grid = torch.tensor(list(itertools.product(np.linspace(-1, 1, 41), repeat=2)), dtype=torch.float64)
    for i, eps in enumerate((0.8, 0.4)):
        points = grid * eps + z[i].double()
        values = copy.deepcopy(head).double()(points)[:, i]
        values = head.to_response(values, [2, -4][i], [0.2, 3][i])
        assert float(values.min()) >= lo[i]
        assert float(values.max()) <= hi[i]
    center_lo, center_hi = HeadBounder(head, method)(z, [0, 1], 0, 0, 1)
    exact = copy.deepcopy(head).double().select(z.double(), torch.tensor([0, 1]))
    exact = head.to_response(exact, 0, 1)
    torch.testing.assert_close(center_lo, exact, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(center_hi, exact, atol=1e-7, rtol=1e-7)
    infinite_lo, infinite_hi = HeadBounder(head, method)(z, [0, 1], float("inf"), 0, 1)
    assert torch.isneginf(infinite_lo).all() and torch.isposinf(infinite_hi).all()


def test_linear_head_bounds_are_exact():
    head = ChronosMedianHead(2, 2, 1, False).eval().requires_grad_(False)
    with torch.no_grad():
        head.output_layer.weight.zero_()
        head.output_layer.bias.zero_()
        head.residual_layer.weight.copy_(torch.tensor([[2.0, -3.0]]))
        head.residual_layer.bias.fill_(1.0)
    lo, hi = HeadBounder(head)(torch.tensor([[1.0, 2.0]]), [0], 0.2, 0, 1)
    np.testing.assert_allclose([lo.item(), hi.item()], [-4, -2], atol=1e-7)


def test_weighted_quantile_has_finite_sample_correction_and_test_mass():
    assert weighted_quantile(np.arange(1, 10), np.ones(9), 0.1) == 9
    assert np.isinf(weighted_quantile([1, 2], [1, 1], 0.1))
    assert weighted_quantile([4, 1, 2], [0, 9, 1], 0.5) == 1
    assert weighted_quantile([1, 1, 3], [1, 1, 1], 0.5) == 1
    assert np.isinf(weighted_quantile([1, 2], [0, 0], 0.5))
    with pytest.raises(ValueError):
        weighted_quantile([1], [-1], 0.1)


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_latent_inversion_is_batch_independent_and_freezes_head(optimizer):
    head, artifact, item, config = example(30)
    cache = item["ctssf"]
    x = cache["features"][cache["feature_indices"]]
    args = (head, x, cache["output_indices"], item["heldout_y"], cache["location"], cache["scale"])
    scores1, errors1 = latent_scores(*args, steps=15, optimizer=optimizer, batch_size=1)
    scores2, errors2 = latent_scores(*args, steps=15, optimizer=optimizer, batch_size=11)
    np.testing.assert_allclose(scores1, scores2, atol=2e-6, rtol=1e-4)
    np.testing.assert_allclose(errors1, errors2, atol=2e-6, rtol=1e-4)
    for name, value in head.state_dict().items():
        torch.testing.assert_close(value, artifact["state_dict"][name], rtol=0, atol=0)
    assert all(p.grad is None for p in head.parameters())


def test_pairing_alignment_and_forecast_validation(tmp_path):
    head, artifact, item, config = example(101)
    prepared = prepare_sequence(item, head, artifact, config["data"])
    assert prepared["boundaries"]["test_start"] == 66
    path = tmp_path / "head.pt"
    torch.save(artifact, path)
    restored, _ = load_chronos_head(path)
    torch.testing.assert_close(restored(torch.zeros(1, 3)), head(torch.zeros(1, 3)))
    for field in ("head_sha256", "feature_indices", "raw_target_indices", "output_indices", "scale"):
        wrong = copy.deepcopy(item)
        if field == "head_sha256":
            wrong["ctssf"][field] = "wrong"
        else:
            wrong["ctssf"][field][0] = -1
        with pytest.raises(ValueError):
            prepare_sequence(wrong, head, artifact, config["data"])
    wrong = copy.deepcopy(item)
    wrong["heldout_predictions"][0] += 2
    with pytest.raises(ValueError, match="reconstruct"):
        prepare_sequence(wrong, head, artifact, config["data"])
    artifact["state_dict"]["hidden_layer.bias"][0] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        load_chronos_head(artifact)


def test_future_labels_cannot_affect_earlier_intervals_or_attention():
    head, artifact, item, config = example()
    original = evaluate_sequence("a", item, config, head, artifact)
    changed = copy.deepcopy(item)
    changed["heldout_y"][72:] += 25
    rerun = evaluate_sequence("a", changed, config, head, artifact)
    pair = (0.05, 0.95)
    upto = 72 - original["metadata"]["boundaries"]["test_start"] + 1
    for field in ("lower_interval", "upper_interval", "latent_radius"):
        np.testing.assert_allclose(original["evaluation_results"][pair][field][:upto], rerun["evaluation_results"][pair][field][:upto])
    assert original["metadata"]["attention_training"] == rerun["metadata"]["attention_training"]
    assert original["metadata"]["calibration_size"][0] == 16
    assert original["metadata"]["calibration_size"][-1] == 20


def test_attention_fit_ignores_calibration_test_inputs_and_labels():
    rng = np.random.default_rng(1)
    x, s = rng.normal(size=(40, 3)).astype("float32"), rng.random(40).astype("float32")
    opts = {"epochs": 2, "embedding_dim": 4, "memory_size": 4}
    torch.manual_seed(1)
    a, info = fit_attention(x, s, 20, 25, opts)
    x[25:] += 1000
    s[25:] += 1000
    torch.manual_seed(1)
    b, info2 = fit_attention(x, s, 20, 25, opts)
    assert info == info2
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)
    encoded = a(torch.from_numpy(x[:5]))
    assert a.weights(encoded[0], encoded[:1]).item() == pytest.approx(1)


def test_standard_artifacts_end_to_end_and_infinite_metrics(tmp_path):
    head, artifact, item, config = example()
    config["model"]["weighting"] = "uniform"
    config["model"]["calibration_size"] = 1
    config["model"]["rolling_calibration"] = False
    data_path, head_path = tmp_path / "data.pkl", tmp_path / "head.pt"
    with data_path.open("wb") as stream:
        pickle.dump({"series": item}, stream)
    torch.save(artifact, head_path)
    config["data"].update(data_path=str(data_path), head_path=str(head_path))
    config["saving_dir"] = str(tmp_path / "results")
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    log = run_ct_ssf(path)
    assert (tmp_path / "results" / "log.pkl").is_file()
    assert (tmp_path / "results" / "summary_results.pkl").is_file()
    assert (tmp_path / "results" / "resolved_config.yaml").is_file()
    result = log["series"]["evaluation_results"][(0.05, 0.95)]
    assert result["target_indices"] == list(range(66, 100))
    assert result["infinite_interval_count"] == 34
    assert np.isinf(result["avg_winkler_score"])
    assert not np.isnan(result["winkler_score"]).any()


@pytest.mark.parametrize("dataset", ["air", "solar", "sapflux"])
def test_presets_resolve_paired_paths(dataset):
    config = resolve_config(build_parser().parse_args(["--dataset", dataset, "--dry-run"]))
    assert "chronos_ctssf" in config.data.data_path
    assert config.data.head_path.replace("_ctssf_head.pt", "_data.pkl") == config.data.data_path
    assert config.data.test_ratio == 0.34
