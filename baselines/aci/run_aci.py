"""Fixed-configuration ACI evaluation with chronological calibration/test splits."""

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from decimal import Decimal
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
    if "tuning" in config:
        raise ValueError("Normal ACI runs use fixed model settings and do not accept a tuning section. "
                         "Use sbatch.sbatch_run_tuning.run_aci_tuning with a separate grid config; "
                         "then pass its resolved_config.yaml to the normal runner.")
    plotting = config.setdefault("plotting", {})
    if any(not isinstance(node, dict) for node in (data, model, plotting)):
        raise ValueError("data, model and plotting must be mappings.")
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


def _save_pickle(path, value):
    with Path(path).open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def run_aci(config, num_cores=None):
    """Evaluate the test suffix using the fixed model settings in the config."""
    if isinstance(config, (str, Path)):
        config = load_experiment_config(config)
    requested = validate_config(config)
    if num_cores is not None:
        requested["num_cores"] = positive_integer(num_cores, "num_cores")
    if not requested["data"].get("data_path") or not requested.get("saving_dir"):
        raise ValueError("data.data_path and saving_dir are required to run an experiment.")
    with Path(requested["data"]["data_path"]).open("rb") as stream:
        data = pickle.load(stream)
    print(f"ACI: evaluating {len(data)} independent sequences.", flush=True)
    log = evaluate_sequences(data, requested)
    pairs = target_quantiles(requested)
    summary = summarize_results(log, pairs)
    output = Path(requested["saving_dir"])
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(requested), output / "requested_config.yaml", resolve=True)
    OmegaConf.save(OmegaConf.create(requested), output / "resolved_config.yaml", resolve=True)
    _save_pickle(output / "log.pkl", log)
    _save_pickle(output / "summary_results.pkl", summary)
    print("ACI summary (mean +/- std across sequences; width and Winkler use "
          "finite, nonempty intervals):", flush=True)
    for pair, result in summary.items():
        metrics = []
        for label, metric in (("coverage", "avg_coverage"),
                              ("width", "avg_interval_width"),
                              ("Winkler", "avg_winkler_score")):
            values = (result[f"{metric}_mean"], result[f"{metric}_std"])
            formatted = " +/- ".join("N/A" if value is None else f"{value:.4f}" for value in values)
            metrics.append(f"{label}={formatted}")
        print(f"Coverage pair {pair}: " + ", ".join(metrics), flush=True)
        print(f"  sequences={result['num_sequences']}, "
              f"efficiency_sequences={result['num_efficiency_sequences']}, "
              f"full_sets={result['num_full_sets']} ({result['full_set_rate']:.2%}), "
              f"empty_sets={result['num_empty_sets']} ({result['empty_set_rate']:.2%})", flush=True)
    if requested["plotting"]["plotting"]:
        from baselines.aci.reporting import plot_aci_results
        plot_aci_results(log, pairs, requested["plotting"]["plotting_seq_len"], output / "plots")
    return log
