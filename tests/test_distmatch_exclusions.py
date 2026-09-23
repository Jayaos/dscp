"""DistMatch skips only crossed-bound test predictions, preserving online state."""

import importlib
import csv
import json
import pickle

import numpy as np
import pytest
from omegaconf import OmegaConf

from baselines.distmatch.data import prepare_sequence
from baselines.distmatch.model import (
    DistMatchCrossedBoundsError,
    DistMatchResidualIntervalEstimator,
)
from baselines.distmatch.run_distmatch import evaluate_sequence


RUNNER = importlib.import_module("baselines.distmatch.run_distmatch")
PAIRS = ((0.05, 0.95), (0.1, 0.9))
DIAGNOSTICS = {
    "tree_index": 1,
    "triggering_quantile_pair": (0.1, 0.9),
    "beta": 0.1,
    "lower_quantile": 0.1,
    "upper_quantile": 0.9,
    "lower_bound": 1.0000001,
    "upper_bound": 1.0,
}


def config(normalize_residual=False):
    return {
        "seed": 37,
        "num_cores": 1,
        "threads_per_worker": 1,
        "show_progress": False,
        "data": {
            "train_ratio": 0.5, "valid_ratio": 0.25, "test_ratio": 0.25,
            "normalize_residual": normalize_residual,
        },
        "model": {
            "past_window_len": 3, "n_trees": 2, "qrf_n_estimators": 2,
            "beta_bins": 3, "target_quantiles": [list(pair) for pair in PAIRS],
        },
    }


def artifact():
    index = np.arange(24, dtype=float)
    predictions = 40 + index * 0.2
    return {
        "heldout_y": predictions + 2 * np.sin(index * 0.8),
        "heldout_predictions": predictions[:, None],
    }


def inject_predictions(monkeypatch, failing_offsets, exception_factory=None):
    """Keep the real fit and observe path; substitute predictable QRF outputs."""
    calls, updates = [], []
    original_observe = DistMatchResidualIntervalEstimator.observe

    def predict(self, pairs):
        offset = len(calls)
        calls.append(self._history.copy())
        if offset in failing_offsets:
            raise (exception_factory() if exception_factory else
                   DistMatchCrossedBoundsError(**DIAGNOSTICS))
        center = float(self._history[-1])
        return {pair: (center - 0.5, center + 0.5, [offset / 100, offset / 100 + 0.01])
                for pair in pairs}

    def observe(self, residual):
        updates.append(residual)
        return original_observe(self, residual)

    monkeypatch.setattr(DistMatchResidualIntervalEstimator, "predict_intervals", predict)
    monkeypatch.setattr(DistMatchResidualIntervalEstimator, "observe", observe)
    monkeypatch.setattr(DistMatchResidualIntervalEstimator, "_load_qrf", staticmethod(lambda: None))
    return calls, updates


@pytest.mark.parametrize("normalize_residual", [False, True])
def test_middle_failure_excludes_all_pairs_and_preserves_history(monkeypatch, normalize_residual):
    item, cfg = artifact(), config(normalize_residual)
    prepared = prepare_sequence(item, cfg)
    calls, updates = inject_predictions(monkeypatch, {2})
    entry = evaluate_sequence("station", item, cfg)

    valid = np.array([True, True, False, True, True, True])
    metadata = entry["metadata"]
    assert metadata["evaluation_status"] == "completed_with_exclusions"
    assert metadata["valid_prediction_mask"] == valid.tolist()
    assert metadata["target_indices"] == list(range(18, 24))
    assert metadata["total_points"] == 6
    assert metadata["evaluated_points"] == 5
    assert metadata["excluded_points_count"] == 1
    assert metadata["exclusion_rate"] == pytest.approx(1 / 6)
    assert metadata["final_memory_size"] - metadata["initial_memory_size"] == 6
    assert metadata["diagnostics"]["observed_updates"] == 12
    np.testing.assert_array_equal(updates, np.concatenate([
        prepared["warmup_residuals"], prepared["residuals"],
    ]))
    all_residuals = np.concatenate([
        prepared["train_residuals"], prepared["warmup_residuals"], prepared["residuals"],
    ])
    for offset, history in enumerate(calls):
        np.testing.assert_array_equal(history, all_residuals[15 + offset:18 + offset])

    scale = prepared["normalization"]["target_std"]
    centers = all_residuals[17:23][valid]
    targets = prepared["y"][valid]
    predictions = prepared["predictions"][valid]
    expected_lower = predictions + (centers - 0.5) * scale
    expected_upper = predictions + (centers + 0.5) * scale
    for pair, result in entry["evaluation_results"].items():
        # The same retained points feed saved metrics, plots, and rolling windows.
        for field in (
            "target_indices", "target_y", "target_predictions", "lower_interval", "upper_interval",
            "lower_residual_quantile", "upper_residual_quantile", "selected_beta_per_tree",
            "coverage", "interval_width", "winkler_score",
        ):
            assert len(result[field]) == 5, field
            assert np.isfinite(result[field]).all(), field
        assert result["target_indices"] == [18, 19, 21, 22, 23]
        np.testing.assert_array_equal(result["target_y"], targets)
        np.testing.assert_array_equal(result["target_predictions"], predictions)
        np.testing.assert_allclose(result["lower_interval"], expected_lower)
        np.testing.assert_allclose(result["upper_interval"], expected_upper)
        np.testing.assert_allclose(result["selected_beta_per_tree"], [
            [offset / 100, offset / 100 + 0.01] for offset in (0, 1, 3, 4, 5)
        ])
        expected_coverage = (expected_lower <= targets) & (targets <= expected_upper)
        expected_width = expected_upper - expected_lower
        alpha = 1 - (pair[1] - pair[0])
        expected_winkler = expected_width + 2 / alpha * (
            np.maximum(expected_lower - targets, 0) + np.maximum(targets - expected_upper, 0)
        )
        np.testing.assert_array_equal(result["coverage"], expected_coverage)
        np.testing.assert_allclose(result["interval_width"], expected_width)
        np.testing.assert_allclose(result["winkler_score"], expected_winkler)
        assert result["avg_coverage"] == pytest.approx(expected_coverage.mean())
        assert result["avg_interval_width"] == pytest.approx(expected_width.mean())
        assert result["avg_winkler_score"] == pytest.approx(expected_winkler.mean())

    assert len(metadata["excluded_points"]) == 1
    excluded = metadata["excluded_points"][0]
    assert excluded["test_offset"] == 2
    assert excluded["target_index"] == 20
    assert excluded["tree_index"] == DIAGNOSTICS["tree_index"]
    assert excluded["lower_bound"] == DIAGNOSTICS["lower_bound"]
    assert excluded["upper_bound"] == DIAGNOSTICS["upper_bound"]


def test_all_failed_predictions_have_empty_scores_and_no_numerical_average(monkeypatch):
    calls, updates = inject_predictions(monkeypatch, set(range(6)))
    entry = evaluate_sequence("station", artifact(), config())
    metadata = entry["metadata"]
    assert metadata["evaluation_status"] == "no_valid_predictions"
    assert metadata["valid_prediction_mask"] == [False] * 6
    assert metadata["total_points"] == metadata["excluded_points_count"] == 6
    assert metadata["evaluated_points"] == 0
    assert metadata["exclusion_rate"] == 1
    assert len(metadata["excluded_points"]) == 6
    assert len(calls) == 6
    assert len(updates) == 12
    assert metadata["final_memory_size"] - metadata["initial_memory_size"] == 6
    for result in entry["evaluation_results"].values():
        for field in ("target_y", "target_predictions", "target_indices", "coverage", "interval_width",
                      "winkler_score", "lower_interval", "upper_interval", "selected_beta_per_tree"):
            assert result[field] == []
        for field in ("avg_coverage", "avg_delta_coverage", "avg_interval_width", "avg_winkler_score"):
            assert result[field] is None


def test_successful_sequence_keeps_all_points(monkeypatch):
    inject_predictions(monkeypatch, set())
    entry = evaluate_sequence("station", artifact(), config())
    metadata = entry["metadata"]
    assert metadata["evaluation_status"] == "complete"
    assert metadata["valid_prediction_mask"] == [True] * 6
    assert metadata["excluded_points"] == []
    assert metadata["excluded_points_count"] == 0
    assert metadata["exclusion_rate"] == 0
    for result in entry["evaluation_results"].values():
        assert result["target_indices"] == list(range(18, 24))


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError])
def test_unrelated_prediction_errors_still_abort(monkeypatch, exception_type):
    # Even matching text is insufficient: only the dedicated exception is recoverable.
    calls, updates = inject_predictions(
        monkeypatch, {2}, lambda: exception_type("DistMatch QRF returned crossed interval bounds."),
    )
    with pytest.raises(exception_type, match="crossed interval bounds"):
        evaluate_sequence("station", artifact(), config())
    assert len(calls) == 3
    assert len(updates) == 8  # Warmup and the two completed test observations.


def test_validation_retains_fail_fast_behavior(monkeypatch):
    calls, updates = inject_predictions(monkeypatch, {2})
    with pytest.raises(DistMatchCrossedBoundsError):
        evaluate_sequence("station", artifact(), config(), split="validation")
    assert len(calls) == 3
    assert len(updates) == 2


def test_model_reports_crossed_candidate_with_picklable_diagnostics(monkeypatch):
    class CrossedQRF:
        def __init__(self, **options):
            self.quantiles = options["q"]

        def fit(self, xs, ys):
            return self

        def predict(self, query):
            np.testing.assert_array_equal(self.quantiles, [0.1, 0.9])
            return np.array([[1.0000001], [1.0]])

    monkeypatch.setattr(DistMatchResidualIntervalEstimator, "_load_qrf", staticmethod(lambda: CrossedQRF))
    estimator = DistMatchResidualIntervalEstimator(
        past_window_len=3, n_trees=1, qrf_n_estimators=2, use_beta_search=False,
    ).fit(np.sin(np.arange(12)))
    before = estimator.diagnostics()["observed_updates"]
    with pytest.raises(DistMatchCrossedBoundsError) as raised:
        estimator.predict_intervals([(0.1, 0.9)])
    error = raised.value
    assert isinstance(error, ValueError)
    assert error.diagnostics["tree_index"] == 0
    assert tuple(error.diagnostics["triggering_quantile_pair"]) == (0.1, 0.9)
    assert error.diagnostics["beta"] == pytest.approx(0.1)
    assert error.diagnostics["lower_bound"] == 1.0000001
    assert error.diagnostics["upper_bound"] == 1.0
    assert estimator.diagnostics()["observed_updates"] == before
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is DistMatchCrossedBoundsError
    assert str(restored) == str(error)
    assert restored.diagnostics == error.diagnostics


def test_completed_outputs_and_plots_use_only_successful_predictions(monkeypatch, tmp_path):
    cfg = config()
    log = {}
    for key, failed in (("partial", {2}), ("complete", set()), ("empty", set(range(6)))):
        with monkeypatch.context() as patched:
            inject_predictions(patched, failed)
            log[key] = evaluate_sequence(key, artifact(), cfg)
    cfg.update(saving_dir=str(tmp_path), plotting={"plotting": True, "plotting_seq_len": 4})
    plotted = []
    monkeypatch.setattr("utils.plotting.plot_cp_prediction_intervals", lambda *args: plotted.append(args))
    assert RUNNER._write_run_results(cfg, log, {"elapsed_seconds": 0}) is log

    with (tmp_path / "summary_results.pkl").open("rb") as stream:
        summary = pickle.load(stream)
    for pair, result in summary.items():
        assert result["num_sequences"] == 2
        assert result["num_total_sequences"] == 3
        assert result["num_no_valid_sequences"] == 1
        assert result["total_points"] == 18
        assert result["evaluated_points"] == 11
        assert result["excluded_points_count"] == 7
        assert result["exclusion_rate"] == pytest.approx(7 / 18)
        assert result["evaluation_status"] == "completed_with_exclusions"
        # Preserve the established equal weight per evaluable sequence.
        for metric in ("coverage", "interval_width", "winkler_score", "delta_coverage"):
            means = [log[key]["evaluation_results"][pair][f"avg_{metric}"] for key in ("partial", "complete")]
            assert result[f"avg_{metric}_mean"] == pytest.approx(np.mean(means))
            assert result[f"avg_{metric}_std"] == pytest.approx(np.std(means))

    assert len(plotted) == 1
    plotted_log, plotted_pairs, length, directory = plotted[0]
    assert set(plotted_log) == {"partial", "complete"}
    assert plotted_log["partial"]["evaluation_results"][PAIRS[0]]["target_indices"] == [18, 19, 21, 22, 23]
    assert length == 4
    for entry in plotted_log.values():
        for result in entry["evaluation_results"].values():
            assert len(result["target_y"]) == len(result["coverage"])
    with (tmp_path / "log.pkl").open("rb") as stream:
        saved_log = pickle.load(stream)
    assert saved_log["empty"]["metadata"]["evaluation_status"] == "no_valid_predictions"
    with (tmp_path / "excluded_points.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7
    run_metadata = OmegaConf.load(tmp_path / "run_metadata.yaml")
    assert run_metadata.excluded_points_count == 7
    assert run_metadata.evaluated_points == 11


def test_entirely_excluded_run_saves_explicit_empty_summary(monkeypatch, tmp_path):
    cfg = config()
    inject_predictions(monkeypatch, set(range(6)))
    log = {"empty": evaluate_sequence("empty", artifact(), cfg)}
    cfg.update(saving_dir=str(tmp_path), plotting={"plotting": True})
    plotted = []
    monkeypatch.setattr("utils.plotting.plot_cp_prediction_intervals", lambda *args: plotted.append(args))
    RUNNER._write_run_results(cfg, log, {})
    with (tmp_path / "summary_results.pkl").open("rb") as stream:
        summary = pickle.load(stream)
    for result in summary.values():
        assert result["evaluation_status"] == "no_valid_predictions"
        assert result["num_sequences"] == 0
        assert result["evaluated_points"] == 0
        assert result["excluded_points_count"] == 6
        assert result["avg_coverage_mean"] is None
        assert result["avg_interval_width_mean"] is None
        assert result["avg_winkler_score_mean"] is None
    assert not plotted or plotted[0][0] == {}
    assert (tmp_path / "excluded_points.csv").is_file()


def test_evaluator_journals_exclusion_before_later_unrelated_error(monkeypatch, tmp_path):
    errors = iter([DistMatchCrossedBoundsError(**DIAGNOSTICS), RuntimeError("unrelated later error")])
    calls, updates = inject_predictions(monkeypatch, {1, 3}, lambda: next(errors))
    cfg = config()
    cfg["saving_dir"] = str(tmp_path)
    with pytest.raises(RuntimeError, match="unrelated later error"):
        evaluate_sequence("station", artifact(), cfg)
    journals = list((tmp_path / "exclusions").glob("*.jsonl"))
    assert len(journals) == 1
    events = [json.loads(line) for line in journals[0].read_text(encoding="utf-8").splitlines()]
    assert len(events) == 1
    assert events[0]["test_offset"] == 1
    assert events[0]["target_index"] == 19
    assert len(calls) == 4
    assert len(updates) == 9
