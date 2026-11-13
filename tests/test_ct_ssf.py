import copy
import itertools
import pickle

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from base_predictor.chronos_features import ChronosMedianHead, head_fingerprint, load_chronos_head
from baselines.ct_ssf.attention import OBJECTIVE, SemanticErrorAttention, fit_attention
from baselines.ct_ssf.bounds import HeadBounder
from baselines.ct_ssf.data import prepare_sequence
from baselines.ct_ssf.run_ct_ssf import _inversion_diagnostics, _metrics, evaluate_sequence, run_ct_ssf
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


def test_weighted_quantile_preserves_alignment_and_extreme_common_scale():
    scores = np.array([4, 1, 3, 2])
    weights = np.array([1.0, 4.0, 3.0, 2.0])
    expected = weighted_quantile(scores, weights, 0.3, test_weight=2)
    for scale in (1e-300, 1e307):
        assert weighted_quantile(scores, weights * scale, 0.3, test_weight=2 * scale) == expected
    order = [2, 0, 3, 1]
    assert weighted_quantile(scores[order], weights[order], 0.3, test_weight=2) == expected


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_latent_inversion_is_batch_independent_and_freezes_head(optimizer):
    head, artifact, item, config = example(30)
    cache = item["ctssf"]
    x = cache["features"][cache["feature_indices"]]
    args = (head, x, cache["output_indices"], item["heldout_y"], cache["location"], cache["scale"])
    scores1, errors1 = latent_scores(*args, steps=15, optimizer=optimizer, batch_size=1)
    scores2, errors2, details = latent_scores(*args, steps=15, optimizer=optimizer, batch_size=11, return_details=True, semantic_end=17)
    np.testing.assert_allclose(scores1, scores2, atol=2e-6, rtol=1e-4)
    np.testing.assert_allclose(errors1, errors2, atol=2e-6, rtol=1e-4)
    assert details["semantic_errors"].shape == (17, x.shape[1])
    np.testing.assert_allclose(scores2[:17], np.abs(details["semantic_errors"]).max(1))
    u = torch.from_numpy(x[:17] + details["semantic_errors"])
    fitted = head.select(u, torch.as_tensor(cache["output_indices"][:17])).double()
    reconstructed = head.to_response(fitted, torch.from_numpy(cache["location"][:17]).double(), torch.from_numpy(cache["scale"][:17]).double())
    # u = z0 + (u-z0) and GEMM batch shape can round differently in float32.
    np.testing.assert_allclose(details["reconstructed_targets"][:17], reconstructed.numpy(), atol=2e-6, rtol=1e-6)
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
    x = rng.normal(size=(40, 3)).astype("float32")
    s = rng.normal(size=(40, 2)).astype("float32")
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
    masses, self_mass = a.weights(encoded[0], encoded[:1])
    assert masses.item() == pytest.approx(0.5)
    assert self_mass.item() == pytest.approx(0.5)
    np.testing.assert_allclose(info["error_mean"], s[:20].mean(0), atol=1e-7)
    assert info["objective"] == OBJECTIVE


def test_attention_learns_vector_similarity_even_with_equal_scalar_scores():
    torch.manual_seed(3)
    signs = (torch.arange(80) % 2).float() * 2 - 1
    x = signs[:, None] + torch.randn(80, 3) * 0.05
    errors = signs[:, None].expand(-1, 2).clone()
    # All scalar L-inf scores equal 1. Only the vector directions distinguish
    # these two groups; the former log-score regression had no such signal.
    model, info = fit_attention(x, errors, 48, 64, {
        "epochs": 35, "patience": 35, "learning_rate": 0.01,
        "embedding_dim": 8, "memory_size": 16,
    })
    assert info["validation_loss"] < info["validation_history"][0]
    encoded = model(x)
    masses, self_mass = model.weights(encoded[64], encoded[:48])
    assert masses[::2].mean() > 3 * masses[1::2].mean()
    assert float(masses.sum() + self_mass) == pytest.approx(1)
    order = torch.randperm(48)
    permuted, permuted_self = model.weights(encoded[64], encoded[:48][order])
    torch.testing.assert_close(permuted, masses[order])
    torch.testing.assert_close(permuted_self, self_mass)
    assert all(not p.requires_grad for p in model.parameters())


def test_attention_checkpoint_reconstructs_first_interval(tmp_path):
    head, artifact, item, config = example()
    result = evaluate_sequence("a", item, config, head, artifact, checkpoint_dir=tmp_path)
    checkpoint = torch.load(tmp_path / result["metadata"]["attention_checkpoint"], weights_only=True)
    assert checkpoint["format_version"] == 2
    assert checkpoint["objective"] == OBJECTIVE
    assert checkpoint["head_sha256"] == artifact["head_sha256"]
    options = checkpoint["options"]
    restored = SemanticErrorAttention(checkpoint["input_dim"], options["embedding_dim"], options.get("temperature", 0.5))
    restored.load_state_dict(checkpoint["state_dict"])
    data = prepare_sequence(item, head, artifact, config["data"])
    end, begin = data["boundaries"]["test_start"], data["boundaries"]["valid_end"]
    scores, _ = latent_scores(head, data["features"][begin:end], data["output_indices"][begin:end], data["y"][begin:end], data["location"][begin:end], data["scale"][begin:end], steps=25)
    with torch.no_grad():
        encoded = restored(torch.from_numpy(data["attention_inputs"][begin:end + 1]))
        weights, test_mass = restored.weights(encoded[-1], encoded[:-1])
    radius = weighted_quantile(scores, weights.numpy(), 1 - (0.95 - 0.05), float(test_mass))
    saved = result["evaluation_results"][(0.05, 0.95)]
    assert radius == pytest.approx(saved["latent_radius"][0], rel=1e-5)
    lo, hi = HeadBounder(head)(torch.from_numpy(data["features"][end:end + 1]), data["output_indices"][end:end + 1], radius, data["location"][end:end + 1], data["scale"][end:end + 1])
    assert lo.item() == pytest.approx(saved["lower_interval"][0], rel=1e-5)
    assert hi.item() == pytest.approx(saved["upper_interval"][0], rel=1e-5)


def test_inversion_diagnostics_and_output_coverage_are_not_feature_coverage():
    head = ChronosMedianHead(2, 2, 1, False).eval().requires_grad_(False)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.residual_layer.weight[0, 0] = 1
    # One SGD step: target in head units is 4, z moves from 0 to 0.8.
    scores, errors, detail = latent_scores(head, np.zeros((1, 2), dtype=np.float32), [0], [13.0], [5.0], [2.0], steps=1, learning_rate=0.1, optimizer="sgd", return_details=True)
    np.testing.assert_allclose(detail["semantic_errors"], [[0.8, 0]], atol=1e-6)
    diagnostics = _inversion_diagnostics(errors, detail, np.array([13.0]), 0, 1)
    assert diagnostics["relative_squared_model_scale_error"] == pytest.approx(0.64)
    assert diagnostics["relative_squared_response_error"] == pytest.approx((6.4 / 13) ** 2)
    assert diagnostics["fraction_response_rows_below_1pct"] == 0
    radius = np.array([1.0])
    lo, hi = HeadBounder(head)(torch.zeros(1, 2), [0], radius, [5.0], [2.0])
    metrics = _metrics(lo.numpy(), hi.numpy(), np.array([13.0]), np.array([5.0]), np.array([0]), (0.05, 0.95), radius, scores, detail["reconstructed_targets"])
    assert metrics["avg_feature_coverage"] == 1
    assert metrics["avg_surrogate_output_coverage"] == 1
    assert metrics["avg_coverage"] == 0


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
