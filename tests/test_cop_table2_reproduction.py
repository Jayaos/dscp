"""Data alignment and reporting regressions for the AMZN sanity run."""

import contextlib
from datetime import date, timedelta
import io
from pathlib import Path

import numpy as np
import pytest

from baselines.cop_table2.run_reproduction import (
    author_intervals, evaluate_author_metrics, load_amzn_data,
    matches_table2, run_reproduction,
)


def write_inputs(directory, forecasts, targets=None, timestamps=None):
    directory = Path(directory)
    forecasts = np.asarray(forecasts)
    n = len(forecasts)
    targets = list(forecasts) if targets is None else targets
    timestamps = [str(date(2020, 1, 1) + timedelta(days=i)) for i in range(n)] if timestamps is None else timestamps
    forecast_path = directory / "forecasts.npz"
    target_path = directory / "targets.csv"
    np.savez(forecast_path, forecasts=forecasts)
    target_path.write_text("timestamp,Open\n" + "".join(
        f"{timestamp},{target}\n" for timestamp, target in zip(timestamps, targets)
    ), encoding="utf-8")
    return forecast_path, target_path


def test_loader_interpolates_union_row_positions_and_keeps_startup_predictions(tmp_path):
    paths = write_inputs(tmp_path, [1., 1., 1.], [10., "", 20.],
                         ["2020-01-01", "2020-01-02", "2020-01-10"])
    loaded = load_amzn_data(*paths)
    np.testing.assert_array_equal(loaded["y"], [10., 15., 20.])
    np.testing.assert_array_equal(loaded["forecasts"], [1., 1., 1.])
    assert loaded["metadata"]["interpolated_target_dates"] == ["2020-01-02"]
    assert loaded["metadata"]["leading_forecast_ones_retained"] == 3
    assert not loaded["metadata"]["matches_authors_prophet_forecasts"]


@pytest.mark.parametrize("forecasts", [
    np.array([[1.], [2.]]), np.array([1., np.nan]), np.array([1., np.inf]),
    np.array(["1", "2"], dtype=object),
])
def test_loader_rejects_invalid_or_pickled_forecasts(tmp_path, forecasts):
    paths = write_inputs(tmp_path, forecasts, [1., 2.])
    with pytest.raises(ValueError):
        load_amzn_data(*paths)


@pytest.mark.parametrize("targets,timestamps,error", [
    ([1.], ["2020-01-01"], "lengths differ"),
    ([1., 2.], ["2020-01-01", "2020-01-01"], "strictly increasing"),
    (["", 2.], ["2020-01-01", "2020-01-02"], "interpolatable"),
    ([1., np.inf], ["2020-01-01", "2020-01-02"], "finite"),
])
def test_loader_rejects_misaligned_or_unusable_targets(tmp_path, targets, timestamps, error):
    paths = write_inputs(tmp_path, [1., 2.], targets, timestamps)
    with pytest.raises(ValueError, match=error):
        load_amzn_data(*paths)


def test_metrics_keep_infinity_and_use_author_coverage_and_width_timelines():
    y = np.zeros(105)
    y[101:103] = [5., 1.]
    lower, upper = np.full(105, -1.), np.full(105, 1.)
    lower[:101], upper[:101] = -10., 10.
    lower[0], upper[0] = -np.inf, np.inf
    result = evaluate_author_metrics(y, lower, upper)
    assert result["coverage_percent"] == 75.  # Boundary equality is covered.
    assert np.isposinf(result["mean_width"])
    assert result["median_width"] == 20.
    assert result["post_burnin_mean_width"] == 2.
    assert result["post_burnin_median_width"] == 2.
    assert result["num_coverage_points"] == 4
    assert result["num_width_points"] == 105
    assert result["num_infinite_widths"] == 1
    assert result["num_infinite_widths_post_burnin"] == 0


def test_author_wrapper_reorders_crossings_and_retains_full_sets():
    lower, upper, crossed = author_intervals(
        np.array([10., 20., 30.]), np.array([-2., 3., np.inf]),
        np.array([-1., -4., np.inf]),
    )
    np.testing.assert_array_equal(lower, [9., 16., -np.inf])
    np.testing.assert_array_equal(upper, [12., 17., np.inf])
    np.testing.assert_array_equal(crossed, [True, True, False])


def test_display_precision_comparison_handles_infinite_reference():
    reference = {"coverage_percent": 90.2, "mean_width": np.inf, "median_width": 46.97}
    measured = {"coverage_percent": 90.2364, "mean_width": np.inf, "median_width": 46.97198}
    assert matches_table2(measured, reference)
    assert not matches_table2({**measured, "mean_width": 1e100}, reference)
    assert not matches_table2({**measured, "coverage_percent": 90.3}, reference)


def test_full_runner_reports_every_candidate_without_writes_or_shared_method_state(tmp_path, monkeypatch, capsys):
    index = np.arange(135)
    y = 100. + .1 * index + np.sin(index / 3.)
    forecasts = y + 2. * np.cos(index / 5.)
    paths = write_inputs(tmp_path, forecasts, y)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.chdir(tmp_path)
    rows = run_reproduction(*paths)
    log = capsys.readouterr().out
    assert len(rows) == 15
    assert [row["learning_rate"] for row in rows if row["method"] == "aci"] == [.1, .05, .01, .005]
    assert "no calibration/test split" in log
    assert "Primary mean/median widths: all 135 points" in log
    assert "Coverage: indices 101:135 (34 points)" in log
    assert "diagnostic comparison, not hyperparameter selection" in log
    with contextlib.redirect_stdout(io.StringIO()):
        cop_only = run_reproduction(*paths, methods=("cop",))
    assert cop_only == [row for row in rows if row["method"] == "cop"]
    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after
