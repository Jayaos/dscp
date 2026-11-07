"""Configurable ACI baseline with chronological, calibration-only tuning."""

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from decimal import Decimal
from itertools import product
import math
from pathlib import Path
import pickle
import time

import numpy as np
from omegaconf import OmegaConf

from baselines.aci.data import prepare_sequence, validate_split_settings
from baselines.aci.model import ACI, PAPER, UPSTREAM_SOURCE, UPSTREAM_COMMIT, finite_number, positive_integer
from baselines.aci.reporting import evaluate_intervals, summarize_results
from utils.experiment_config import load_experiment_config
from utils.forecast_data import canonicalize_forecast_data


MODEL_DEFAULTS = {
    "gamma": 0.005, "window_size": 1250, "initial_alpha": None,
    "quantile_method": "linear", "score_type": "absolute", "prediction_step": 1,
    "target_quantiles": [[0.05, 0.95]],
}


def _plain_config(config):
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping):
        raise ValueError("ACI configuration must be a mapping.")
    return OmegaConf.to_container(OmegaConf.create(config), resolve=True)


def target_quantiles(config):
    pairs = _plain_config(config).get("model", {}).get("target_quantiles", [[0.05, 0.95]])
    if not isinstance(pairs, (list, tuple)) or not pairs:
        raise ValueError("model.target_quantiles must be a nonempty list of pairs.")
    result, seen = [], set()
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each target_quantiles entry must contain two levels.")
        values = tuple(finite_number(value, "quantile level") for value in pair)
        lo, hi = sorted(values)
        if not 0 < lo < hi < 1:
            raise ValueError("Quantile levels must satisfy 0 < lower < upper < 1.")
        if not math.isclose(lo + hi, 1.0, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Absolute/relative-error ACI requires symmetric target_quantiles.")
        if (lo, hi) in seen:
            raise ValueError("Duplicate target coverage levels are not allowed.")
        seen.add((lo, hi))
        result.append(values)
    return result


def _alpha(pair):
    lo, hi = sorted(pair)
    return float(Decimal(1) - Decimal(str(hi)) + Decimal(str(lo)))


def validate_config(config):
    config = _plain_config(config)
    data = config.setdefault("data", {})
    model = config.setdefault("model", {})
    tuning = config.setdefault("tuning", {})
    plotting = config.setdefault("plotting", {})
    if any(not isinstance(node, dict) for node in (data, model, tuning, plotting)):
        raise ValueError("data, model, tuning and plotting must be mappings.")
    forbidden = {"train_ratio", "valid_ratio", "validation_ratio", "normalize"} & data.keys()
    if forbidden:
        raise ValueError("ACI uses calibration/test only and explicit score_type; unsupported data settings: "
                         + ", ".join(sorted(forbidden)))
    ratio = finite_number(data.setdefault("calibration_ratio", .66), "data.calibration_ratio")
    data.setdefault("test_ratio", float(Decimal(1) - Decimal(str(ratio))))
    validate_split_settings(ratio, data["test_ratio"], data.get("test_start"))
    unknown = set(model) - set(MODEL_DEFAULTS)
    if unknown:
        raise ValueError("Unknown ACI model settings: " + ", ".join(sorted(unknown)))
    for key, value in MODEL_DEFAULTS.items():
        model.setdefault(key, deepcopy(value))
    if positive_integer(model["prediction_step"], "model.prediction_step") != 1:
        raise ValueError("ACI currently supports sequential prediction_step=1 only.")
    if model["score_type"] not in {"absolute", "relative"}:
        raise ValueError("model.score_type must be 'absolute' or 'relative'.")
    pairs = target_quantiles(config)
    model["target_quantiles"] = [list(pair) for pair in pairs]
    # Constructor validates all algorithm settings without requiring any data.
    for pair in pairs:
        _estimator(model, pair)
    config["num_cores"] = positive_integer(config.get("num_cores", 1), "num_cores")
    if not isinstance(plotting.setdefault("plotting", False), bool):
        raise ValueError("plotting.plotting must be a boolean.")
    plotting["plotting_seq_len"] = positive_integer(plotting.get("plotting_seq_len", 200), "plotting.plotting_seq_len")
    if not isinstance(tuning.setdefault("enabled", False), bool):
        raise ValueError("tuning.enabled must be a boolean.")
    allowed_tuning = {"enabled", "validation_ratio", "gamma_grid", "window_size_grid", "coverage_tolerance"}
    if set(tuning) - allowed_tuning:
        raise ValueError("Unknown ACI tuning settings: " + ", ".join(sorted(set(tuning) - allowed_tuning)))
    tuning.setdefault("validation_ratio", .2)
    if not 0 < finite_number(tuning["validation_ratio"], "tuning.validation_ratio") < 1:
        raise ValueError("tuning.validation_ratio must be strictly between zero and one.")
    tuning.setdefault("coverage_tolerance", .01)
    if not 0 <= finite_number(tuning["coverage_tolerance"], "tuning.coverage_tolerance") < 1:
        raise ValueError("tuning.coverage_tolerance must be in [0, 1).")
    tuning.setdefault("gamma_grid", [model["gamma"]])
    tuning.setdefault("window_size_grid", [model["window_size"]])
    for key in ("gamma_grid", "window_size_grid"):
        if not isinstance(tuning[key], list) or not tuning[key]:
            raise ValueError(f"tuning.{key} must be a nonempty list.")
    for gamma, window in product(tuning["gamma_grid"], tuning["window_size_grid"]):
        _estimator({**model, "gamma": gamma, "window_size": window}, pairs[0])
    return config


def _estimator(model, pair):
    return ACI(alpha=_alpha(pair), gamma=model["gamma"], window_size=model["window_size"],
               initial_alpha=model["initial_alpha"], quantile_method=model["quantile_method"])


def _evaluate_prepared(prepared, model, pairs):
    results = {}
    for pair in pairs:
        estimator = _estimator(model, pair).fit(prepared["calibration_scores"])
        lower, upper, alpha_used, alpha_next, thresholds, errors = [], [], [], [], [], []
        for i in range(len(prepared["y"])):
            threshold = estimator.predict()
            # Both scale and point forecast are available before the outcome.
            prediction, scale = float(prepared["predictions"][i]), float(prepared["scales"][i])
            if math.isinf(threshold):
                lo, hi = (-math.inf, math.inf) if threshold > 0 else (math.inf, -math.inf)
            else:
                radius = threshold * scale
                lo, hi = prediction - radius, prediction + radius
                if not all(math.isfinite(value) for value in (radius, lo, hi)):
                    raise ValueError("Finite ACI threshold overflowed the response-scale endpoints.")
            lower.append(lo)
            upper.append(hi)
            diagnostic = estimator.observe(float(prepared["scores"][i]))
            alpha_used.append(diagnostic["alpha_used"])
            alpha_next.append(diagnostic["alpha_next"])
            thresholds.append(threshold)
            errors.append(diagnostic["error"])
        result = evaluate_intervals(prepared["y"], prepared["predictions"], lower, upper,
                                    _alpha(pair), prepared["target_indices"])
        result.update({"alpha_used": alpha_used, "alpha_next": alpha_next,
                       "score_threshold": thresholds, "scores": prepared["scores"].tolist(),
                       "adaptive_errors": errors, "final_alpha": estimator.alpha_t,
                       "final_history_size": estimator.history_size})
        results[pair] = result
    return results


def evaluate_sequence(key, item, config, split="test"):
    config = validate_config(config)
    prepared = prepare_sequence(item, config, key=key, split=split)
    started = time.perf_counter()
    results = _evaluate_prepared(prepared, config["model"], target_quantiles(config))
    return {"evaluation_results": results, "metadata": {
        "method": "ACI", "split": split, "boundaries": prepared["boundaries"],
        "target_indices": prepared["target_indices"].tolist(),
        "calibration_size": len(prepared["calibration_scores"]),
        "model": deepcopy(config["model"]),
        "interval_scale": "original_response", "residual_quantile_scale": "original_residual",
        "score_type": config["model"]["score_type"],
        "calibration_update": "rolling_after_observation",
        "initialization": "historical_scores_only; adaptive_level_reset_before_evaluation",
        "feedback_protocol": "one_step_sequential", "evaluation_seconds": time.perf_counter() - started,
        "paper": PAPER, "upstream_source": UPSTREAM_SOURCE, "upstream_commit": UPSTREAM_COMMIT,
    }}


def _sequence_task(args):
    key, item, config, split = args
    return key, evaluate_sequence(key, item, config, split)


def evaluate_sequences(data, config, num_cores=None, split="test"):
    config = validate_config(config)
    data = canonicalize_forecast_data(data)
    if not data:
        raise ValueError("The forecast artifact must contain at least one sequence.")
    workers = config["num_cores"] if num_cores is None else positive_integer(num_cores, "num_cores")
    tasks = [(key, item, config, split) for key, item in data.items()]
    if workers == 1:
        return dict(_sequence_task(task) for task in tasks)
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        return dict(executor.map(_sequence_task, tasks))


def tune_hyperparameters(data, config):
    """Select shared settings across series using only the calibration tail.

    Every candidate has fresh independent states. Prioritize coverage, then
    fewer full/empty sets, then finite-interval Winkler score. All means used
    for selection give each series/coverage pair equal weight.
    """
    config = validate_config(config)
    data = canonicalize_forecast_data(data)
    if not data:
        raise ValueError("The forecast artifact must contain at least one sequence.")
    pairs = target_quantiles(config)
    prepared = {key: prepare_sequence(item, config, key, split="validation")
                for key, item in data.items()}
    trials = []
    tuning = config["tuning"]
    for trial_id, (gamma, window) in enumerate(product(tuning["gamma_grid"], tuning["window_size_grid"])):
        model = {**config["model"], "gamma": gamma, "window_size": window}
        deficits, exceptions, scores, details = [], [], [], {}
        for key, sequence in prepared.items():
            sequence_results = _evaluate_prepared(sequence, model, pairs)
            details[key] = {}
            for pair, metrics in sequence_results.items():
                deficit = max(0.0, 1 - _alpha(pair) - tuning["coverage_tolerance"] - metrics["avg_coverage"])
                deficits.append(deficit)
                exceptions.append(metrics["full_set_rate"] + metrics["empty_set_rate"])
                scores.append(metrics["avg_winkler_score"] if metrics["avg_winkler_score"] is not None else math.inf)
                details[key][pair] = {name: metrics[name] for name in (
                    "avg_coverage", "avg_winkler_score", "num_full_sets", "num_empty_sets", "num_finite_intervals",
                )}
        eligible = max(deficits) == 0 and all(math.isfinite(value) for value in scores)
        rank = (0 if eligible else 1, max(deficits), float(np.mean(exceptions)), float(np.mean(scores)))
        trials.append({"trial_id": trial_id, "gamma": float(gamma), "window_size": window,
                       "eligible": eligible, "rank": rank, "validation_results": details})
    selected = min(trials, key=lambda trial: trial["rank"])
    return {"selected_hyperparameters": {"gamma": selected["gamma"], "window_size": selected["window_size"]},
            "selected_trial_id": selected["trial_id"], "all_trials": trials,
            "scope": "shared_across_sequences_and_coverage_levels",
            "selection_rule": "coverage eligibility, maximum shortfall, exceptional-set rate, finite-interval Winkler",
            "validation_boundaries": {key: value["boundaries"] for key, value in prepared.items()},
            "validation_target_indices": {key: value["target_indices"].tolist() for key, value in prepared.items()}}


def _save_pickle(path, value):
    with Path(path).open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def run_aci(config, num_cores=None):
    """Run optional calibration-only selection, then evaluate the test suffix."""
    if isinstance(config, (str, Path)):
        config = load_experiment_config(config)
    requested = validate_config(config)
    if num_cores is not None:
        requested["num_cores"] = positive_integer(num_cores, "num_cores")
    if not requested["data"].get("data_path") or not requested.get("saving_dir"):
        raise ValueError("data.data_path and saving_dir are required to run an experiment.")
    with Path(requested["data"]["data_path"]).open("rb") as stream:
        data = pickle.load(stream)
    effective = deepcopy(requested)
    selection = None
    if requested["tuning"]["enabled"]:
        print("ACI: selecting hyperparameters on calibration data only.", flush=True)
        selection = tune_hyperparameters(data, requested)
        effective["model"].update(selection["selected_hyperparameters"])
        print(f"Selected: {selection['selected_hyperparameters']}", flush=True)
    # Persist a replayable fixed configuration; never re-tune final-test output.
    effective["tuning"]["enabled"] = False
    print(f"ACI: evaluating {len(data)} independent sequences.", flush=True)
    log = evaluate_sequences(data, effective)
    pairs = target_quantiles(effective)
    summary = summarize_results(log, pairs)
    output = Path(effective["saving_dir"])
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(requested), output / "requested_config.yaml", resolve=True)
    OmegaConf.save(OmegaConf.create(effective), output / "resolved_config.yaml", resolve=True)
    _save_pickle(output / "log.pkl", log)
    _save_pickle(output / "summary_results.pkl", summary)
    if selection is not None:
        _save_pickle(output / "tuning_results.pkl", selection)
    for pair, result in summary.items():
        print(f"Coverage pair {pair}: {result}", flush=True)
    if effective["plotting"]["plotting"]:
        from baselines.aci.reporting import plot_aci_results
        plot_aci_results(log, pairs, effective["plotting"]["plotting_seq_len"], output / "plots")
    return log
