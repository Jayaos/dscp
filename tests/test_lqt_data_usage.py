"""Protocol, tuning, reporting, and replay integration for LQT."""

from copy import deepcopy
from pathlib import Path
import pickle
import tempfile

import numpy as np
import pytest
from omegaconf import OmegaConf

from baselines.lqt.config import candidates, validate_config
from baselines.lqt.data import prepare_sequence
from baselines.lqt.run_lqt import (
    evaluate_sequence, evaluate_sequences, run_lqt, select_trial, tune_hyperparameters,
)


def config():
    return {"data": {"calibration_ratio": .66, "test_ratio": .34},
            "model": {"method": "lqt", "target_quantiles": [[.05, .95], [.1, .9]]},
            "tuning": {"enabled": True, "learning_rate_grid": [.01, .1],
                       "p_order_grid": [0, 1], "bias_grid": [1.]},
            "plotting": {"plotting": False}}


def artifact(length=31):
    time = np.arange(length, dtype=float)
    predictions = 10 + .1 * time
    return {"heldout_y": predictions + 2 * np.sin(time),
            "heldout_predictions": predictions[:, None]}


def test_calibration_selection_never_reads_test_values():
    settings = config()
    item = artifact()
    reference = tune_hyperparameters({"a": item}, settings)
    poisoned = deepcopy(item)
    start = int(np.floor(len(item["heldout_y"]) * .66))
    poisoned["heldout_y"][start:] = np.nan
    poisoned["heldout_predictions"][start:] = np.nan
    assert reference == tune_hyperparameters({"a": poisoned}, settings)
    for index in reference["sequence_results"]["a"]["calibration_target_indices"]:
        assert index < start
    with pytest.raises(ValueError):
        evaluate_sequence("a", poisoned, settings)


def test_test_starts_cold_and_current_outcome_cannot_affect_issued_interval():
    settings = config()
    settings["model"]["learning_rate_schedule"] = "decaying"
    item = artifact()
    original = evaluate_sequence("a", item, settings)
    changed_cal = deepcopy(item)
    changed_cal["heldout_y"][:20] += 10000
    cold = evaluate_sequence("a", changed_cal, settings)
    assert original["evaluation_results"] == cold["evaluation_results"]
    future = deepcopy(item)
    future["heldout_y"][25:] += 500
    altered = evaluate_sequence("a", future, settings)
    for pair, metrics in original["evaluation_results"].items():
        np.testing.assert_array_equal(metrics["score_threshold"][:6],
                                      altered["evaluation_results"][pair]["score_threshold"][:6])
        assert metrics["online_diagnostics"]["learning_rates"][0] == .1
        assert metrics["target_indices"] == list(range(20, 31))


def test_nexcp_floor_and_explicit_per_series_boundaries():
    settings = validate_config(config())
    for length, ratio in ((101, .66), (100, .29)):
        settings["data"].update(calibration_ratio=ratio, test_ratio=1-ratio)
        prepared = prepare_sequence(artifact(length), settings, "a", "calibration")
        assert len(prepared["scores"]) == int(np.floor(length * ratio))
    settings["data"]["test_start"] = {"a": 28}
    assert prepare_sequence(artifact(), settings, "a")["target_indices"][0] == 28
    with pytest.raises(ValueError):
        prepare_sequence(artifact(), settings, "missing")


def test_selection_coverage_constraint_fallback_and_tie_order():
    trials = [{"status": "ok", "trial_id": 0, "eligible": False, "quantile_loss": .1},
              {"status": "ok", "trial_id": 1, "eligible": True, "quantile_loss": 2.},
              {"status": "ok", "trial_id": 2, "eligible": True, "quantile_loss": 2.}]
    assert select_trial(trials)["trial_id"] == 1
    for trial in trials:
        trial["eligible"] = False
    assert select_trial(trials)["trial_id"] == 0
    with pytest.raises(RuntimeError, match="All hyperparameter"):
        select_trial([{"status": "failed", "failure": "numerical failure"}])


def test_published_grids_and_scope():
    validated = validate_config({"model": {"method": "lqt"}})
    grid = list(candidates(validated))
    assert len(grid) == 231
    assert grid[0]["learning_rate"] == 1e-5
    assert grid[-1]["learning_rate"] == 1e5
    assert validated["tuning"]["enabled"]
    data = {"a": artifact(31), "b": artifact(25)}
    data["b"]["heldout_y"] += 20
    settings = config()
    together = tune_hyperparameters(data, settings)
    for key, item in data.items():
        alone = tune_hyperparameters({key: item}, settings)
        assert together["selected_hyperparameters"][key] == alone["selected_hyperparameters"][key]


def test_negative_thresholds_empty_zero_thresholds_singleton():
    settings = config()
    settings["model"].update(p_order=0, initial_quantile=-1, learning_rate=0)
    result = evaluate_sequence("a", artifact(), settings)["evaluation_results"][(.05, .95)]
    assert result["num_empty_sets"] == result["num_test_points"]
    assert result["avg_interval_width"] is None
    assert result["avg_coverage"] == 0
    assert all(q == -1 for q in result["score_threshold"])
    settings["model"]["initial_quantile"] = 0
    perfect = {"heldout_y": np.ones(6), "heldout_predictions": np.ones(6)}
    result = evaluate_sequence("a", perfect, settings)["evaluation_results"][(.05, .95)]
    assert result["avg_coverage"] == 1
    assert result["num_empty_sets"] == 0
    assert result["avg_interval_width"] == 0


def test_parallel_evaluation_and_independent_coverage_states():
    data = {"a": artifact(), "b": artifact(24)}
    settings = config()
    selected = tune_hyperparameters(data, settings)["selected_hyperparameters"]
    serial = evaluate_sequences(data, settings, num_cores=1, selected_hyperparameters=selected)
    parallel = evaluate_sequences(data, settings, num_cores=2, selected_hyperparameters=selected)
    for key in data:
        assert serial[key]["evaluation_results"] == parallel[key]["evaluation_results"]
        for pair in serial[key]["evaluation_results"]:
            reduced = deepcopy(settings)
            reduced["model"]["target_quantiles"] = [list(pair)]
            separate = evaluate_sequence(key, data[key], reduced, {pair: selected[key][pair]})
            assert separate["evaluation_results"][pair] == serial[key]["evaluation_results"][pair]


def test_run_saved_configs_replay_selected_hyperparameters():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with (root / "forecasts.pkl").open("wb") as stream:
            pickle.dump({"a": artifact(15), "b": artifact(12)}, stream)
        settings = config()
        settings["data"]["data_path"] = str(root / "forecasts.pkl")
        settings["saving_dir"] = str(root / "run")
        settings["plotting"]["plotting"] = True
        original = run_lqt(settings)
        resolved = OmegaConf.load(root / "run" / "resolved_config.yaml")
        assert not resolved.tuning.enabled
        assert Path(resolved.selected_hyperparameters_path).is_file()
        assert (root / "run" / "tuning_results.pkl").is_file()
        assert len(list((root / "run" / "plots").glob("*.png"))) == 4
        resolved.saving_dir = str(root / "replay")
        resolved.plotting.plotting = False
        replayed = run_lqt(resolved)
        for key in original:
            assert original[key]["evaluation_results"] == replayed[key]["evaluation_results"]
        resolved.model.decay_exponent = .7
        with pytest.raises(ValueError, match="decay_exponent"):
            run_lqt(resolved)


@pytest.mark.parametrize("section,key,value", [
    ("model", "prediction_step", 10), ("model", "score_type", "relative"),
    ("model", "method", "sqt_ar"),
    ("model", "target_quantiles", [[.1, .8]]), ("model", "clipping", True),
    ("data", "normalize", True), ("data", "calibration_ratio", 1),
    ("tuning", "validation_ratio", .2), ("tuning", "learning_rate_grid", []),
])
def test_reject_invalid_settings(section, key, value):
    settings = config()
    settings[section][key] = value
    with pytest.raises(ValueError):
        validate_config(settings)
