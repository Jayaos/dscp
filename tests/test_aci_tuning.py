"""QR selection parity, test isolation, candidate states, and runnable exports."""

import ast
from copy import deepcopy
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Dict, Tuple

import numpy as np
from omegaconf import OmegaConf
import pytest

from baselines.aci.data import prepare_sequence
from baselines.aci.reporting import evaluate_intervals
from baselines.aci.run_aci import run_aci
from baselines.aci.tuning import (
    choose_sequence_keys, summarize_candidate, tune_hyperparameters, validate_tuning_config,
)
from report.inspect_tuning_results import inspect_results
from sbatch.sbatch_run_tuning.run_aci_tuning import resolve_inputs, run_tuning

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [[.05, .95], [.1, .9]]


def settings():
    return {"data": {"calibration_ratio": .66, "test_ratio": .34},
            "model": {"target_quantiles": PAIRS}, "plotting": {"plotting": False},
            "tuning": {"initialization_ratio": .5, "gamma_grid": [0., .005],
                       "window_size_grid": [5, 10], "delta_threshold": -.01}}


def series(n=100, poison_test=False):
    y, pred = np.ones(n), np.zeros(n)
    if poison_test:
        y[int(n * .66):] = pred[int(n * .66):] = np.nan
    return {"heldout_y": y, "heldout_predictions": pred}


def metrics(coverage=1., score=2., full=0, empty=0):
    return {"avg_coverage": coverage, "avg_winkler_score": score,
            "avg_interval_width": score, "num_full_sets": full, "num_empty_sets": empty,
            "num_finite_intervals": 10 - full - empty, "full_set_rate": full / 10,
            "empty_set_rate": empty / 10}


def test_matches_actual_qr_selection_helpers_on_finite_intervals():
    # Load just the pure QR helpers: the ACI environment need not import Torch.
    tree = ast.parse((ROOT / "sbatch/sbatch_run_tuning/common.py").read_text())
    names = {"summarize_evaluation_results", "aggregate_sequence_results"}
    selected = ast.Module(body=[node for node in tree.body
                              if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = {"np": np, "Dict": Dict, "Tuple": Tuple}
    exec(compile(selected, "qr_selection_helpers", "exec"), namespace)
    raw, qr = {}, {}
    for key, n, width in (("short", 10, 2.), ("long", 100, 3.)):
        results = {tuple(pair): evaluate_intervals(np.ones(n), np.zeros(n),
                   np.full(n, -width), np.full(n, width), 1 - (pair[1] - pair[0]), np.arange(n))
                   for pair in PAIRS}
        pair_metrics, score, passed = namespace["summarize_evaluation_results"](results, PAIRS, -.01)
        qr[key] = {"pair_metrics": pair_metrics, "selection_score": score,
                   "positive_delta_coverage": passed}
        raw[key] = results
    expected = namespace["aggregate_sequence_results"](qr, PAIRS)
    actual = summarize_candidate(raw, PAIRS, -.01)
    assert actual["selection_score"] == pytest.approx(expected["selection_score"])
    assert actual["positive_delta_coverage"] == expected["positive_delta_coverage"]
    for pair, values in expected["pair_metrics"].items():
        for name, value in values.items():
            assert actual["pair_metrics"][pair][name] == pytest.approx(value)


def test_coverage_filter_is_strict_and_applies_to_every_sequence_level():
    raw = {"a": {tuple(pair): metrics() for pair in PAIRS},
           "b": {tuple(pair): metrics() for pair in PAIRS}}
    pair = tuple(PAIRS[0])
    raw["b"][pair] = metrics(coverage=.88)
    boundary = .88 - (max(pair) - min(pair))
    assert not summarize_candidate(raw, PAIRS, boundary)["coverage_eligible"]
    assert summarize_candidate(raw, PAIRS, boundary - .001)["coverage_eligible"]


def test_exception_counts_are_reported_without_becoming_ranking_penalties():
    pair = tuple(PAIRS[0])
    cheap = summarize_candidate({"a": {pair: metrics(score=1., full=5)}}, [pair], -.01)
    costly = summarize_candidate({"a": {pair: metrics(score=3.)}}, [pair], -.01)
    assert cheap["coverage_eligible"] and costly["coverage_eligible"]
    assert cheap["selection_score"] < costly["selection_score"]
    assert cheap["sequence_results"]["a"]["pair_metrics"][str(pair)]["num_full_sets"] == 5
    undefined = summarize_candidate({"a": {pair: metrics(score=None, full=10)}}, [pair], -.01)
    assert not undefined["coverage_eligible"]
    assert np.isinf(undefined["selection_score"])


@pytest.mark.parametrize("options,expected", [
    ({"num_sequences": 2, "sequence_index": 1}, ["b", "c"]),
    ({"num_sequences": "all", "sequence_index": 99}, ["a", "b", "c"]),
    ({"num_sequences": 3, "sequence_key": "b"}, ["b"]),
])
def test_qr_sequence_subset_selection(options, expected):
    config = settings()
    config["tuning"].update(options)
    tuning = validate_tuning_config(config)["tuning"]
    assert choose_sequence_keys({"c": None, "a": None, "b": None}, tuning) == expected


def test_selection_split_and_unused_series_and_test_values():
    config = settings()
    config["tuning"].update(num_sequences=1)
    reference = tune_hyperparameters({"a": series()}, config)
    actual = tune_hyperparameters({"a": series(poison_test=True), "z": {"invalid": None}}, config)
    assert actual == reference
    assert actual["validation_target_indices"]["a"] == list(range(50, 66))
    assert not actual["final_test_evaluated"]
    assert actual["scope"] == "shared_across_selected_sequences_and_coverage_levels"
    # Candidate evaluation is independent of grid order or previous candidates.
    reversed_config = deepcopy(config)
    for field in ("gamma_grid", "window_size_grid"):
        reversed_config["tuning"][field].reverse()
    reordered = tune_hyperparameters({"a": series(poison_test=True)}, reversed_config)
    for trial in actual["all_trials"]:
        same = next(row for row in reordered["all_trials"] if row["grid_values"] == trial["grid_values"])
        assert same["result"] == trial["result"]
        assert "tuning" not in trial["resolved_config"]
    assert actual["selected_trial_id"] == 1  # All tie on constant residuals; grid order wins.


def test_no_coverage_fallback_and_no_final_test(tmp_path):
    config = settings()
    config.pop("tuning")
    artifact = tmp_path / "data.pkl"
    artifact.write_bytes(pickle.dumps({"a": series(poison_test=True)}))
    config["data"]["data_path"] = str(artifact)
    base, grid = tmp_path / "base.yaml", tmp_path / "grid.yaml"
    OmegaConf.save(OmegaConf.create(config), base)
    OmegaConf.save(OmegaConf.create({"grid": {"model.gamma": [0.]},
                    "tuning": {"initialization_ratio": .5, "delta_threshold": .2}}), grid)
    output = tmp_path / "output"
    result = run_tuning(base, grid, output)
    assert result["selected_hyperparameters"] is None
    assert result["top_trials"] == []
    assert (output / "tuning_results.pkl").is_file()
    assert not list(output.rglob("log.pkl"))
    assert not list(output.rglob("final_run"))


def test_invalid_initialization_cannot_reach_the_test_set():
    config = settings()
    config["tuning"]["initialization_ratio"] = .7
    with pytest.raises(ValueError, match="initialization and tuning"):
        prepare_sequence(series(poison_test=True), config, "a", split="validation")


def test_legacy_coverage_tolerance_migrates_to_signed_threshold():
    config = settings()
    del config["tuning"]["delta_threshold"]
    config["tuning"]["coverage_tolerance"] = .02
    normalized = validate_tuning_config(config)["tuning"]
    assert normalized["delta_threshold"] == -.02
    assert "coverage_tolerance" not in normalized


def test_tuning_cli_exports_inspectable_replayable_configs_without_testing(tmp_path, capsys):
    config = settings()
    config.pop("tuning")
    artifact = tmp_path / "data.pkl"
    artifact.write_bytes(pickle.dumps({"a": series(poison_test=True)}))
    config["data"]["data_path"] = str(artifact)
    base, grid = tmp_path / "base.yaml", tmp_path / "grid.yaml"
    OmegaConf.save(OmegaConf.create(config), base)
    OmegaConf.save(OmegaConf.create({"grid": {"model.gamma": [0., .005],
                    "model.window_size": [5, 10]}, "tuning": {"initialization_ratio": .5}}), grid)
    output = tmp_path / "output"
    command = [sys.executable, "-m", "sbatch.sbatch_run_tuning.run_aci_tuning",
               "--base-config", str(base), "--grid-config", str(grid), "--save-dir", str(output)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    payload = pickle.loads((output / "tuning_results.pkl").read_bytes())
    assert payload["num_trials"] == 4
    assert len(payload["top_trials"]) == 3
    assert not payload["final_test_evaluated"]
    chosen = payload["top_trials"][0]
    trial_dir = output / ("trial_%04d" % chosen["trial_index"])
    trial_config = OmegaConf.load(trial_dir / "resolved_config.yaml")
    assert "tuning" not in trial_config
    assert "enabled" not in payload["tuning_settings"]
    assert not (trial_dir / "final_run").exists()
    inspect_results(output / "tuning_results.pkl", top_k=1)
    assert "Rank 1" in capsys.readouterr().out
    # Only the separate final run consumes test outcomes; it recreates state at index66.
    artifact.write_bytes(pickle.dumps({"a": series()}))
    replay = run_aci(trial_config)
    assert replay["a"]["metadata"]["target_indices"] == list(range(66, 100))
    assert replay["a"]["evaluation_results"][(.05, .95)]["alpha_used"][0] == .1


@pytest.mark.parametrize("dataset,count", [("air", 3), ("solar", 10), ("sapflux", 5)])
def test_dedicated_presets_for_all_predictors(dataset, count, tmp_path):
    for predictor in ("lr", "lstm", "chronos"):
        config, _, _, _ = resolve_inputs(
            ROOT / f"configs/aci_configs/aci_{predictor}_{dataset}_config.yaml",
            ROOT / f"configs/aci_configs/aci_{dataset}_tuning_config.yaml", tmp_path)
        assert config["tuning"]["num_sequences"] == count
        assert config["tuning"]["initialization_ratio"] == .5
        assert config["tuning"]["delta_threshold"] == -.01
        assert len(config["tuning"]["gamma_grid"]) * len(config["tuning"]["window_size_grid"]) == 16
        assert config["model"]["gamma"] == .005 and config["model"]["window_size"] == 1250


def test_dry_run_writes_nothing(tmp_path):
    output = tmp_path / "absent"
    run_tuning(ROOT / "configs/aci_configs/aci_lr_air_config.yaml",
               ROOT / "configs/aci_configs/aci_air_tuning_config.yaml", output, dry_run=True)
    assert not output.exists()


def test_grid_cannot_change_data_or_coverage_between_candidates(tmp_path):
    grid = tmp_path / "invalid.yaml"
    OmegaConf.save(OmegaConf.create({"grid": {"data.calibration_ratio": [.5, .66]}}), grid)
    with pytest.raises(ValueError, match="keep data splits and coverage fixed"):
        resolve_inputs(ROOT / "configs/aci_configs/aci_lr_air_config.yaml", grid, tmp_path)


@pytest.mark.parametrize("field,value", [
    ("enabled", True), ("enabled", False), ("gamma_grid", []),
    ("gamma_grid", [-.01]), ("window_size_grid", [0]), ("validation_ratio", 0),
])
def test_dedicated_tuning_validates_search_settings(field, value):
    config = settings()
    config["tuning"][field] = value
    with pytest.raises(ValueError):
        validate_tuning_config(config)
