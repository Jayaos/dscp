"""Both QR-CP runners report independent heads on noncrossing timestamps only."""

import csv
import json
import pickle

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dscp import run_qr_cp


PAIRS = [(0.9, 0.1), (0.2, 0.8)]
LEVELS = [0.1, 0.2, 0.8, 0.9]
POINT_FIELDS = (
    "target_indices", "target_y", "target_predictions", "lower_interval",
    "upper_interval", "lower_residual_quantile", "upper_residual_quantile",
    "coverage", "interval_width", "winkler_score",
)
METRICS = ("coverage", "delta_coverage", "interval_width", "winkler_score")


def artifact():
    values = np.linspace(-1, 1, 24, dtype=np.float32)
    return {
        "heldout_x": np.column_stack([values, values**2]),
        "heldout_y": np.sin(values),
        "heldout_predictions": values * 0.2,
    }


def prediction_rows():
    return np.array([
        [-1, -0.5, 0.5, 1],
        # Both requested intervals are ordered, but q(.1) > q(.2).
        [-0.5, -1, 0.5, 1],
        [-1, -0.5, 0.5, 1],
        [-1, 0.7, 0.5, 1],
        # Equal adjacent quantiles remain valid.
        [-1, -1, 0.5, 1],
        [-1, -0.5, 2, 1],
    ], dtype=np.float32)


def run_predictions(
    monkeypatch, tmp_path, architecture, rows_by_sequence, *, normalize=False,
    head_type="independent", use_current_feature=False, plotting=False,
):
    cfg = OmegaConf.create({
        "device": "cpu",
        "saving_dir": str(tmp_path),
        "data": {
            "data_path": "tiny_air_data.pkl", "train_ratio": 0.5,
            "valid_ratio": 0.25, "normalize": normalize,
            "strided_features": "xr",
        },
        "model": {
            "dim_model": 4, "num_heads": 2, "num_layers": 1, "dropout": 0.0,
            "use_current_feature": use_current_feature,
            "target_quantiles": [list(pair) for pair in PAIRS],
            "prediction_step": 1, "window_size": 3, "rnn_type": "gru",
        },
        "training": {
            "batch_size": 2, "learning_rate": 0.001,
            "epochs": 1, "early_stop": 1,
        },
        "plotting": {"plotting": plotting, "plotting_seq_len": 3},
    })
    if head_type is not None:
        cfg.model.head_type = head_type

    model_class = (
        run_qr_cp.QuantileRegressionRNN if architecture == "rnn"
        else run_qr_cp.QuantileRegressionTransformer
    )
    runner = getattr(run_qr_cp, f"run_{architecture}_quantile_regression_cp")
    sequence_rows = iter(rows_by_sequence.values())
    loss_samples = {"train": 0, "validation": 0}
    prediction_calls = []
    plot_calls = []

    def loss(model, features, targets, pairs, current=None):
        # Keep fitting/checkpoint selection active without depending on learned outputs.
        phase = "train" if model.training else "validation"
        loss_samples[phase] += len(targets)
        return next(model.parameters()).square().mean()

    def predict(model, features, current=None):
        if not hasattr(model, "test_rows"):
            model.test_rows = torch.as_tensor(next(sequence_rows))
            model.test_offset = 0
        start, end = model.test_offset, model.test_offset + len(features)
        model.test_offset = end
        prediction_calls.append((start, end))
        assert (current is not None) == use_current_feature
        return model.test_rows[start:end].clone()

    monkeypatch.setattr(run_qr_cp, "load_experiment_config", lambda _: cfg)
    monkeypatch.setattr(
        run_qr_cp, "load_data", lambda _: {key: artifact() for key in rows_by_sequence},
    )
    monkeypatch.setattr(
        run_qr_cp, f"compute_loss_quantile_regression_{architecture}", loss,
    )
    monkeypatch.setattr(model_class, "get_predicted_quantile_values", staticmethod(predict))
    monkeypatch.setattr(run_qr_cp, "tqdm", lambda items, **kwargs: items)
    monkeypatch.setattr(run_qr_cp.torch, "save", lambda *args: None)
    monkeypatch.setattr(
        run_qr_cp, "plot_cp_prediction_intervals", lambda *args: plot_calls.append(args),
    )
    runner("unused_config.yaml")
    with (tmp_path / "log.pkl").open("rb") as stream:
        log = pickle.load(stream)
    with (tmp_path / "summary_results.pkl").open("rb") as stream:
        summary = pickle.load(stream)
    return log, summary, plot_calls, prediction_calls, loss_samples


@pytest.mark.parametrize("architecture", ["rnn", "transformer"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("use_current_feature", [False, True])
def test_crossing_rows_are_excluded_from_every_pair_and_reported_in_raw_units(
    monkeypatch, tmp_path, architecture, normalize, use_current_feature,
):
    rows = prediction_rows()
    log, summary, _, calls, loss_samples = run_predictions(
        monkeypatch, tmp_path, architecture, {"series": rows},
        normalize=normalize, use_current_feature=use_current_feature,
    )
    entry = log["series"]
    metadata = entry["metadata"]
    valid = np.array([True, False, True, False, True, False])
    assert metadata["method"] == "qr_cp"
    assert metadata["head_type"] == "independent"
    assert metadata["valid_prediction_mask"] == valid.tolist()
    assert metadata["target_indices"] == list(range(18, 24))
    assert metadata["total_points"] == 6
    assert metadata["evaluated_points"] == metadata["excluded_points_count"] == 3
    assert metadata["exclusion_rate"] == 0.5
    assert metadata["evaluation_status"] == "completed_with_exclusions"
    assert calls == [(0, 2), (2, 4), (4, 6)]
    # Filtering only affects reporting; every fitting and validation row was used.
    assert loss_samples == {"train": 9, "validation": 6}

    raw = artifact()
    targets = raw["heldout_y"][18:][valid]
    predictions = raw["heldout_predictions"][18:][valid]
    scale = raw["heldout_y"][:12].std() + 1e-8 if normalize else 1
    for pair, result in entry["evaluation_results"].items():
        low_level, high_level = sorted(pair)
        lower_q = rows[valid, LEVELS.index(low_level)]
        upper_q = rows[valid, LEVELS.index(high_level)]
        lower = predictions + lower_q * scale
        upper = predictions + upper_q * scale
        expected_coverage = (lower <= targets) & (targets <= upper)
        expected_width = upper - lower
        expected_score = (
            expected_width + np.maximum(lower - targets, 0) / low_level
            + np.maximum(targets - upper, 0) / (1 - high_level)
        )
        for field in POINT_FIELDS:
            assert len(result[field]) == 3, field
            assert np.isfinite(result[field]).all(), field
        assert result["target_indices"] == [18, 20, 22]
        np.testing.assert_allclose(result["target_y"], targets)
        np.testing.assert_allclose(result["target_predictions"], predictions)
        np.testing.assert_allclose(result["lower_residual_quantile"], lower_q)
        np.testing.assert_allclose(result["upper_residual_quantile"], upper_q)
        np.testing.assert_allclose(result["lower_interval"], lower, atol=1e-7)
        np.testing.assert_allclose(result["upper_interval"], upper, atol=1e-7)
        np.testing.assert_array_equal(result["coverage"], expected_coverage)
        np.testing.assert_allclose(result["interval_width"], expected_width, atol=1e-7)
        np.testing.assert_allclose(result["winkler_score"], expected_score, rtol=1e-6)
        assert result["avg_coverage"] == pytest.approx(expected_coverage.mean())
        assert result["avg_delta_coverage"] == pytest.approx(
            expected_coverage.mean() - (high_level - low_level),
        )
        assert result["avg_interval_width"] == pytest.approx(expected_width.mean())
        assert result["avg_winkler_score"] == pytest.approx(expected_score.mean())
        for field in ("total_points", "evaluated_points", "excluded_points_count",
                      "exclusion_rate", "evaluation_status"):
            assert result[field] == metadata[field]
            assert summary[pair][field] == metadata[field]

    excluded = metadata["excluded_points"]
    assert [point["test_offset"] for point in excluded] == [1, 3, 5]
    assert [point["target_index"] for point in excluded] == [19, 21, 23]
    for point, offset, crossed in zip(excluded, (1, 3, 5), ((0.1, 0.2), (0.2, 0.8), (0.8, 0.9))):
        assert point["crossed_quantile_pairs"] == [list(crossed)]
        assert point["excluded_quantile_pairs"] == [list(pair) for pair in PAIRS]
        assert point["quantile_levels"] == LEVELS
        np.testing.assert_array_equal(point["predicted_quantiles"], rows[offset])
        assert point["bound_scale"] == ("normalized_residual" if normalize else "original_residual")
        assert point["target_y"] == pytest.approx(float(raw["heldout_y"][18 + offset]))
        assert point["reason"] == "quantile_crossing"
        assert point["action"] == "excluded_from_metrics"
    with (tmp_path / "excluded_points.csv").open(newline="", encoding="utf-8") as stream:
        saved_exclusions = list(csv.DictReader(stream))
    assert [int(point["target_index"]) for point in saved_exclusions] == [19, 21, 23]
    assert json.loads(saved_exclusions[0]["crossed_quantile_pairs"]) == [[0.1, 0.2]]


@pytest.mark.parametrize("architecture", ["rnn", "transformer"])
def test_all_crossing_predictions_have_explicit_empty_results_and_no_plot(
    monkeypatch, tmp_path, architecture,
):
    rows = np.tile([1, 0, -1, -2], (6, 1)).astype(np.float32)
    log, summary, plot_calls, calls, loss_samples = run_predictions(
        monkeypatch, tmp_path, architecture, {"empty": rows}, plotting=True,
    )
    metadata = log["empty"]["metadata"]
    assert metadata["valid_prediction_mask"] == [False] * 6
    assert metadata["evaluation_status"] == "no_valid_predictions"
    assert metadata["total_points"] == metadata["excluded_points_count"] == 6
    assert metadata["evaluated_points"] == 0
    assert metadata["exclusion_rate"] == 1
    assert len(metadata["excluded_points"]) == 6
    assert calls == [(0, 2), (2, 4), (4, 6)]
    assert loss_samples == {"train": 9, "validation": 6}
    for pair, result in log["empty"]["evaluation_results"].items():
        for field in POINT_FIELDS:
            assert result[field] == []
        for metric in METRICS:
            assert result[f"avg_{metric}"] is None
            assert summary[pair][f"avg_{metric}_mean"] is None
            assert summary[pair][f"avg_{metric}_std"] is None
        assert summary[pair]["num_sequences"] == 0
        assert summary[pair]["num_total_sequences"] == 1
        assert summary[pair]["num_no_valid_sequences"] == 1
        assert summary[pair]["evaluation_status"] == "no_valid_predictions"
    assert not plot_calls or not plot_calls[0][0]
    with (tmp_path / "excluded_points.csv").open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 6


@pytest.mark.parametrize("architecture", ["rnn", "transformer"])
def test_summary_weights_evaluable_sequences_equally_and_plots_retained_points(
    monkeypatch, tmp_path, architecture,
):
    rows = {
        "partial": prediction_rows(),
        "complete": np.tile([-4, -3, 3, 4], (6, 1)).astype(np.float32),
        "empty": np.tile([1, 0, -1, -2], (6, 1)).astype(np.float32),
    }
    log, summary, plot_calls, _, _ = run_predictions(
        monkeypatch, tmp_path, architecture, rows, plotting=True,
    )
    for pair, result in summary.items():
        assert result["num_sequences"] == 2
        assert result["num_total_sequences"] == 3
        assert result["num_no_valid_sequences"] == 1
        assert result["total_points"] == 18
        assert result["evaluated_points"] == result["excluded_points_count"] == 9
        assert result["exclusion_rate"] == 0.5
        assert result["evaluation_status"] == "completed_with_exclusions"
        for metric in METRICS:
            means = [
                log[key]["evaluation_results"][pair][f"avg_{metric}"]
                for key in ("partial", "complete")
            ]
            assert result[f"avg_{metric}_mean"] == pytest.approx(np.mean(means))
            assert result[f"avg_{metric}_std"] == pytest.approx(np.std(means))
    assert len(plot_calls) == 1
    assert set(plot_calls[0][0]) == {"partial", "complete"}
    assert plot_calls[0][0]["partial"]["evaluation_results"][PAIRS[0]]["target_indices"] == [18, 20, 22]


@pytest.mark.parametrize("architecture", ["rnn", "transformer"])
@pytest.mark.parametrize("head_type", [None, "nondecreasing", "independent"])
def test_noncrossing_predictions_keep_every_point(monkeypatch, tmp_path, architecture, head_type):
    rows = np.tile([-1, -0.5, 0.5, 1], (6, 1)).astype(np.float32)
    log, summary, _, _, _ = run_predictions(
        monkeypatch, tmp_path, architecture, {"complete": rows}, head_type=head_type,
    )
    entry = log["complete"]
    for pair, result in entry["evaluation_results"].items():
        assert len(result["target_y"]) == 6
        assert len(result["coverage"]) == 6
        assert result["avg_interval_width"] == pytest.approx(2 if pair == PAIRS[0] else 1)
        assert summary[pair]["avg_interval_width_mean"] == result["avg_interval_width"]
    if head_type == "independent":
        metadata = entry["metadata"]
        assert metadata["valid_prediction_mask"] == [True] * 6
        assert metadata["evaluation_status"] == "complete"
        assert metadata["excluded_points"] == []
        assert metadata["excluded_points_count"] == metadata["exclusion_rate"] == 0
        assert metadata["evaluated_points"] == metadata["total_points"] == 6
        for result in entry["evaluation_results"].values():
            assert result["target_indices"] == list(range(18, 24))


@pytest.mark.parametrize("architecture", ["rnn", "transformer"])
@pytest.mark.parametrize("head_type", [None, "nondecreasing"])
def test_default_reporting_does_not_enable_independent_head_exclusions(
    monkeypatch, tmp_path, architecture, head_type,
):
    log, _, _, _, _ = run_predictions(
        monkeypatch, tmp_path, architecture, {"series": prediction_rows()}, head_type=head_type,
    )
    for result in log["series"]["evaluation_results"].values():
        assert len(result["target_y"]) == len(result["coverage"]) == 6
