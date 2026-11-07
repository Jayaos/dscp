"""COP integration: signed tails, causal calibration, continuation, and replay."""

from copy import deepcopy
from pathlib import Path
import pickle
import tempfile

import numpy as np
import pytest
from omegaconf import OmegaConf

from baselines.cop.config import validate_config, stream_alphas
from baselines.cop.data import prepare_sequence, interval_endpoints
from baselines.cop.model import predict_thresholds
from baselines.cop.run_cop import (
    evaluate_sequence, evaluate_sequences, run_cop, select_trial, tune_hyperparameters,
)


def config(score_type="signed"):
    return {"data": {"calibration_ratio": .66, "test_ratio": .34},
            "model": {"score_type": score_type, "window_size": 4,
                      "target_quantiles": [[.05, .95], [.1, .9]]},
            "tuning": {"enabled": True, "learning_rate_grid": [1., .1],
                       "window_size_grid": [3, 4], "scale_grid": [.5]},
            "plotting": {"plotting": False}}


def artifact(length=41):
    time = np.arange(length, dtype=float)
    point = 10 + .1 * time
    return {"heldout_y": point + np.sin(time) + .4 * (time % 3),
            "heldout_predictions": point[:, None]}


@pytest.mark.parametrize("score_type", ["signed", "absolute"])
def test_tuning_never_reads_test_targets_or_predictions(score_type):
    settings = config(score_type)
    item = artifact()
    expected = tune_hyperparameters({"a": item}, settings)
    poisoned = deepcopy(item)
    start = int(np.floor(41 * .66))
    poisoned["heldout_y"][start:] = np.nan
    poisoned["heldout_predictions"][start:] = np.nan
    assert expected == tune_hyperparameters({"a": poisoned}, settings)
    result = expected["sequence_results"]["a"]
    assert result["warmup_steps"] == 5  # common max(window grid)+1
    assert result["validation_target_indices"] == list(range(5, start))
    with pytest.raises(ValueError):
        evaluate_sequence("a", poisoned, settings)


@pytest.mark.parametrize("score_type", ["signed", "absolute"])
@pytest.mark.parametrize("initialization", ["calibration_replay", "cold_start"])
def test_state_protocol_matches_direct_uninterrupted_stream(score_type, initialization):
    settings = config(score_type)
    settings["model"]["initialization"] = initialization
    settings = validate_config(settings)
    item = artifact()
    result = evaluate_sequence("a", item, settings)
    prepared = prepare_sequence(item, settings, "a")
    offset = prepared["prefix_length"]
    assert offset == (27 if initialization == "calibration_replay" else 0)
    for pair, metrics in result["evaluation_results"].items():
        for stream, alpha in stream_alphas(pair, score_type).items():
            reference = predict_thresholds(prepared["scores"][stream], alpha, settings["model"])
            np.testing.assert_array_equal(metrics["score_thresholds"][stream],
                                          reference["thresholds"][offset:])
            diagnostics = metrics["online_diagnostics"][stream]
            assert diagnostics["final_state"]["seen"] == (41 if offset else 14)
            if offset:
                assert diagnostics["cdf_active"][0]
            else:
                assert not diagnostics["cdf_active"][0]
        assert metrics["target_indices"] == list(range(27, 41))


@pytest.mark.parametrize("score_type", ["signed", "absolute"])
def test_current_future_outcome_does_not_change_issued_threshold(score_type):
    settings = config(score_type)
    first = artifact()
    expected = evaluate_sequence("a", first, settings)
    changed = deepcopy(first)
    changed["heldout_y"][32:] += 10000
    altered = evaluate_sequence("a", changed, settings)
    for pair, metrics in expected["evaluation_results"].items():
        for stream, thresholds in metrics["score_thresholds"].items():
            np.testing.assert_array_equal(thresholds[:6],
                altered["evaluation_results"][pair]["score_thresholds"][stream][:6])


def test_signed_tail_levels_and_score_conventions():
    settings = validate_config(config())
    prepared = prepare_sequence(artifact(), settings, "a", "calibration")
    residual = prepared["y"] - prepared["predictions"]
    np.testing.assert_array_equal(prepared["scores"]["lower"], -residual)
    np.testing.assert_array_equal(prepared["scores"]["upper"], residual)
    assert stream_alphas((.05, .95), "signed") == {"lower": .05, "upper": .05}
    assert stream_alphas((.1, .9), "absolute") == {"absolute": .2}


def test_negative_signed_threshold_does_not_imply_empty():
    point = np.array([10., 10., 10.])
    thresholds = {"lower": np.array([-2., -3., -2.]),
                  "upper": np.array([4., 1., 2.])}
    lower, upper, raw_lower, raw_upper, empty = interval_endpoints(point, thresholds)
    np.testing.assert_array_equal(empty, [False, True, False])
    assert lower[0] == 12 and upper[0] == 14
    assert lower[2] == upper[2] == 12
    assert lower[1] == np.inf and upper[1] == -np.inf
    assert raw_lower[1] == 13 and raw_upper[1] == 11


def test_empty_sets_are_reported_without_sorting_or_skipping_time():
    settings = config()
    settings["model"].update(initial_quantile=-1., learning_rate=0.)
    metrics = evaluate_sequence("a", artifact(), settings)["evaluation_results"][(.05, .95)]
    assert metrics["num_empty_sets"] == 14
    assert metrics["num_test_points"] == 14
    assert metrics["avg_coverage"] == 0
    assert metrics["avg_interval_width"] is None
    assert metrics["avg_winkler_score"] is None
    assert all(lo > hi for lo, hi in zip(metrics["raw_lower_interval"], metrics["raw_upper_interval"]))
    for stream in metrics["online_diagnostics"].values():
        assert stream["final_state"]["seen"] == 41


def test_nexcp_floor_and_explicit_boundaries():
    settings = validate_config(config())
    for length, ratio in ((101, .66), (100, .29)):
        settings["data"].update(calibration_ratio=ratio, test_ratio=1-ratio)
        prepared = prepare_sequence(artifact(length), settings, "a", "calibration")
        assert len(prepared["y"]) == int(np.floor(length * ratio))
    settings["data"]["test_start"] = {"a": 30}
    assert prepare_sequence(artifact(), settings, "a")["target_indices"][0] == 30
    with pytest.raises(ValueError):
        prepare_sequence(artifact(), settings, "b")


def test_selection_constraint_fallback_and_grid_ties():
    trials = [{"status": "ok", "trial_id": 0, "eligible": False, "quantile_loss": .1},
              {"status": "ok", "trial_id": 1, "eligible": True, "quantile_loss": 2.},
              {"status": "ok", "trial_id": 2, "eligible": True, "quantile_loss": 2.}]
    assert select_trial(trials)["trial_id"] == 1
    for trial in trials:
        trial["eligible"] = False
    assert select_trial(trials)["trial_id"] == 0
    with pytest.raises(RuntimeError, match="All COP"):
        select_trial([{"status": "failed", "failure": "overflow"}])


def test_short_calibration_requires_explicit_warmup_override():
    settings = config()
    with pytest.raises(ValueError, match="tuning warmup"):
        tune_hyperparameters({"a": artifact(7)}, settings)
    settings["tuning"]["warmup_steps"] = 0
    result = tune_hyperparameters({"a": artifact(7)}, settings)
    assert result["sequence_results"]["a"]["validation_target_indices"] == [0, 1, 2, 3]


def test_independent_series_coverage_tails_and_parallel_execution():
    settings = config()
    data = {"a": artifact(), "b": artifact(32)}
    data["b"]["heldout_y"] += 30
    combined = tune_hyperparameters(data, settings)
    for key, item in data.items():
        single = tune_hyperparameters({key: item}, settings)
        assert combined["selected_hyperparameters"][key] == single["selected_hyperparameters"][key]
    selected = combined["selected_hyperparameters"]
    serial = evaluate_sequences(data, settings, selected_hyperparameters=selected)
    parallel = evaluate_sequences(data, settings, num_cores=2, selected_hyperparameters=selected)
    for key in data:
        assert serial[key]["evaluation_results"] == parallel[key]["evaluation_results"]
        for pair in serial[key]["evaluation_results"]:
            restricted = deepcopy(settings)
            restricted["model"]["target_quantiles"] = [list(pair)]
            alone = evaluate_sequence(key, data[key], restricted, {pair: selected[key][pair]})
            assert serial[key]["evaluation_results"][pair] == alone["evaluation_results"][pair]


@pytest.mark.parametrize("score_type", ["signed", "absolute"])
def test_saved_selection_replays_exactly_and_plots(score_type):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data_path = root / "forecasts.pkl"
        with data_path.open("wb") as stream:
            pickle.dump({"a": artifact()}, stream)
        settings = config(score_type)
        settings["data"]["data_path"] = str(data_path)
        settings["saving_dir"] = str(root / "run")
        settings["plotting"]["plotting"] = True
        original = run_cop(settings)
        resolved = OmegaConf.load(root / "run" / "resolved_config.yaml")
        assert not resolved.tuning.enabled
        assert Path(resolved.selected_hyperparameters_path).is_file()
        assert (root / "run" / "tuning_results.pkl").is_file()
        assert len(list((root / "run" / "plots").glob("*.png"))) == 2
        resolved.saving_dir = str(root / "replay")
        resolved.plotting.plotting = False
        replay = run_cop(resolved)
        assert original["a"]["evaluation_results"] == replay["a"]["evaluation_results"]
        resolved.model.score_type = "absolute" if score_type == "signed" else "signed"
        with pytest.raises(ValueError, match="score_type"):
            run_cop(resolved)


@pytest.mark.parametrize("section,key,value", [
    ("model", "prediction_step", 10), ("model", "score_type", "relative"),
    ("model", "initialization", "random"), ("model", "target_quantiles", [[.1, .8]]),
    ("data", "normalize", True), ("data", "calibration_ratio", 1),
    ("tuning", "warmup_steps", -1), ("tuning", "warmup_steps", True),
    ("tuning", "learning_rate_grid", []), ("tuning", "window_size_grid", [0]),
    ("tuning", "scale_grid", [-1]), ("tuning", "coverage_tolerance", 1),
])
def test_invalid_settings_are_rejected(section, key, value):
    settings = config()
    settings[section][key] = value
    with pytest.raises(ValueError):
        validate_config(settings)


def test_replay_cannot_override_untuned_model_settings():
    with pytest.raises(ValueError, match="must contain only"):
        evaluate_sequence("a", artifact(), config(), {
            (.05, .95): {"learning_rate": .1, "window_size": 4, "scale": .5, "initialization": "cold_start"},
            (.1, .9): {"learning_rate": .1, "window_size": 4, "scale": .5},
        })


def test_summed_tail_loss_overflow_is_explicit():
    from baselines.cop.run_cop import _loss
    path = {"scores": {"lower": np.array([1e308]), "upper": np.array([1e308])},
            "thresholds": {"lower": np.array([0.]), "upper": np.array([0.])}}
    with pytest.raises(RuntimeError, match="Summed COP pinball"):
        _loss(path, {"score_type": "signed"}, (.05, .95))
