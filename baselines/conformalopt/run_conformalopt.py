"""Original Areces et al. baselines with calibration-only parameter selection."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import pickle
import time

import numpy as np
from omegaconf import OmegaConf

from baselines.aci.model import positive_integer
from baselines.aci.reporting import evaluate_intervals, summarize_results, plot_aci_results
from baselines.conformalopt.config import validate_config, candidates, target_quantiles, _alpha
from baselines.conformalopt.data import prepare_sequence
from baselines.conformalopt.model import (
    predict_thresholds, compute_ar_scorecasts, validate_model,
    PAPER, UPSTREAM_SOURCE, UPSTREAM_COMMIT,
)
from utils.experiment_config import load_experiment_config
from utils.forecast_data import canonicalize_forecast_data

METHOD_NAMES = {"lqt": "LQT", "sqt_ar": "SQT+AR"}
SELECTION_RULE = "minimum pinball loss within two-sided coverage tolerance; minimum loss fallback"


def quantile_loss(scores, thresholds, alpha):
    with np.errstate(over="ignore", invalid="ignore"):
        residual = np.asarray(scores) - np.asarray(thresholds)
        loss = np.maximum((1 - alpha) * residual, -alpha * residual)
    if not np.isfinite(loss).all():
        raise RuntimeError("Pinball losses overflowed.")
    # Dividing first avoids overflow of a finite mean's intermediate sum.
    return float(np.sum(loss / len(loss)))


def select_trial(trials):
    valid = [trial for trial in trials if trial["status"] == "ok"]
    if not valid:
        raise RuntimeError("All hyperparameter candidates failed: " +
                           "; ".join(dict.fromkeys(trial["failure"] for trial in trials)))
    eligible = [trial for trial in valid if trial["eligible"]]
    # min retains grid iteration order for ties, matching strict < upstream.
    return min(eligible or valid, key=lambda trial: trial["quantile_loss"])


def _tune_sequence(task):
    key, item, config = task
    prepared = prepare_sequence(item, config, key, split="calibration")
    scores = prepared["scores"]
    selections, records = {}, {}
    for pair in target_quantiles(config):
        alpha = _alpha(pair)
        trials, ar_cache = [], {}
        for trial_id, params in enumerate(candidates(config)):
            model = {**config["model"], **params}
            trial = {"trial_id": trial_id, "hyperparameters": params}
            try:
                cached = None
                if model["method"] == "sqt_ar":
                    order = model["p_order"]
                    if order not in ar_cache:
                        try:
                            ar_cache[order] = compute_ar_scorecasts(scores, alpha, model)
                        except RuntimeError as exc:
                            ar_cache[order] = str(exc)
                    cached = ar_cache[order]
                    if isinstance(cached, str):
                        raise RuntimeError(cached)
                result = predict_thresholds(scores, alpha, model, scorecasts=cached)
                thresholds = result["thresholds"]
                coverage = float(np.mean(scores <= thresholds))
                trial.update({"status": "ok", "avg_coverage": coverage,
                              "quantile_loss": quantile_loss(scores, thresholds, alpha),
                              "eligible": abs(coverage - (1 - alpha)) <= config["tuning"]["coverage_tolerance"],
                              "num_negative_thresholds": int(np.count_nonzero(thresholds < 0))})
            except RuntimeError as exc:
                trial.update({"status": "failed", "failure": str(exc), "eligible": False})
            trials.append(trial)
        selected = select_trial(trials)
        selections[pair] = deepcopy(selected["hyperparameters"])
        records[pair] = {"selected_trial_id": selected["trial_id"], "all_trials": trials,
                         "coverage_constraint_satisfied": selected["eligible"]}
    return key, {"selected_hyperparameters": selections, "coverage_results": records,
                 "calibration_boundaries": prepared["boundaries"],
                 "calibration_target_indices": prepared["target_indices"].tolist()}


def _map_tasks(function, tasks, workers):
    if workers == 1:
        return dict(function(task) for task in tasks)
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        return dict(executor.map(function, tasks))


def tune_hyperparameters(data, config):
    """Tune independently for each sequence and coverage on all calibration scores."""
    config = validate_config(config)
    data = canonicalize_forecast_data(data)
    if not data:
        raise ValueError("The forecast artifact must contain at least one sequence.")
    results = _map_tasks(_tune_sequence, [(key, item, config) for key, item in data.items()],
                         config["num_cores"])
    return {"selected_hyperparameters": {key: value["selected_hyperparameters"] for key, value in results.items()},
            "sequence_results": results, "scope": "per_sequence_and_coverage_level",
            "selection_rule": SELECTION_RULE, "method": config["model"]["method"],
            "model": deepcopy(config["model"]), "upstream_commit": UPSTREAM_COMMIT}


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_sequence(key, item, config, selected_hyperparameters=None):
    """Evaluate the test suffix from fresh state; hyperparameters stay fixed."""
    config = validate_config(config)
    prepared = prepare_sequence(item, config, key, split="test")
    started = time.perf_counter()
    results, models = {}, {}
    for pair in target_quantiles(config):
        selected = {} if selected_hyperparameters is None else selected_hyperparameters[pair]
        model = validate_model({**config["model"], **selected})
        output = predict_thresholds(prepared["scores"], _alpha(pair), model)
        thresholds = output["thresholds"]
        predictions = prepared["predictions"]
        empty = thresholds < 0
        with np.errstate(over="ignore", invalid="ignore"):
            lower, upper = predictions - thresholds, predictions + thresholds
        if not np.isfinite(lower[~empty]).all() or not np.isfinite(upper[~empty]).all():
            raise RuntimeError("Finite thresholds overflowed response-scale endpoints.")
        # The set {|y - prediction| <= q} is empty for every negative q.
        lower[empty], upper[empty] = np.inf, -np.inf
        metrics = evaluate_intervals(prepared["y"], predictions, lower, upper,
                                     _alpha(pair), prepared["target_indices"])
        metrics.update({"score_threshold": thresholds.tolist(),
                        "scores": prepared["scores"].tolist(),
                        "avg_quantile_loss": quantile_loss(prepared["scores"], thresholds, _alpha(pair)),
                        "score_coverage": (prepared["scores"] <= thresholds).tolist(),
                        "online_diagnostics": {name: _serializable(value)
                                               for name, value in output.items() if name != "thresholds"}})
        results[pair], models[pair] = metrics, model
    return {"evaluation_results": results, "metadata": {
        "method": METHOD_NAMES[config["model"]["method"]], "boundaries": prepared["boundaries"],
        "target_indices": prepared["target_indices"].tolist(), "models_by_coverage": models,
        "initialization": "fresh_test_state_and_learning_rate_clock; no_calibration_replay",
        "hyperparameter_scope": "per_sequence_and_coverage_level",
        "feedback_protocol": "one_step_sequential", "score_type": "absolute",
        "interval_scale": "original_response", "residual_quantile_scale": "original_residual",
        "evaluation_seconds": time.perf_counter() - started,
        "paper": PAPER, "upstream_source": UPSTREAM_SOURCE, "upstream_commit": UPSTREAM_COMMIT,
    }}


def _evaluate_task(task):
    key, item, config, selection = task
    return key, evaluate_sequence(key, item, config, selection)


def evaluate_sequences(data, config, num_cores=None, selected_hyperparameters=None):
    config = validate_config(config)
    data = canonicalize_forecast_data(data)
    if not data:
        raise ValueError("The forecast artifact must contain at least one sequence.")
    workers = config["num_cores"] if num_cores is None else positive_integer(num_cores, "num_cores")
    if selected_hyperparameters is not None and set(selected_hyperparameters) != set(data):
        raise ValueError("Selected hyperparameters must match the artifact's sequence keys.")
    pairs = set(target_quantiles(config))
    tasks = []
    for key, item in data.items():
        selected = None if selected_hyperparameters is None else selected_hyperparameters[key]
        if selected is not None and set(selected) != pairs:
            raise ValueError(f"Selected coverage levels do not match sequence {key!r}.")
        tasks.append((key, item, config, selected))
    return _map_tasks(_evaluate_task, tasks, workers)


def _save_pickle(path, value):
    with Path(path).open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _load_selection(path, model):
    with Path(path).open("rb") as stream:
        selection = pickle.load(stream)
    if not isinstance(selection, dict) or selection.get("method") != model["method"]:
        raise ValueError("Saved selection belongs to a different method.")
    tuned = {"learning_rate", "p_order", "bias"}
    for name, value in model.items():
        if name not in tuned and selection.get("model", {}).get(name) != value:
            raise ValueError(f"Saved selection used a different model.{name}.")
    return selection


def run_conformalopt(config, num_cores=None):
    """Select on calibration, freeze per-series parameters, and evaluate test."""
    if isinstance(config, (str, Path)):
        config = load_experiment_config(config)
    requested = validate_config(config)
    if num_cores is not None:
        requested["num_cores"] = positive_integer(num_cores, "num_cores")
    if not requested["data"].get("data_path") or not requested.get("saving_dir"):
        raise ValueError("data.data_path and saving_dir are required.")
    with Path(requested["data"]["data_path"]).open("rb") as stream:
        data = pickle.load(stream)
    method_name = METHOD_NAMES[requested["model"]["method"]]
    selection = None
    if requested["tuning"]["enabled"]:
        print(f"{method_name}: selecting hyperparameters on calibration data for each series and coverage.", flush=True)
        selection = tune_hyperparameters(data, requested)
        failures = sum(trial["status"] == "failed"
                       for sequence in selection["sequence_results"].values()
                       for result in sequence["coverage_results"].values()
                       for trial in result["all_trials"])
        if failures:
            print(f"{method_name}: {failures} failed candidates recorded in tuning_results.pkl.", flush=True)
    elif requested.get("selected_hyperparameters_path"):
        selection = _load_selection(requested["selected_hyperparameters_path"], requested["model"])
    effective = deepcopy(requested)
    effective["tuning"]["enabled"] = False
    print(f"{method_name}: evaluating {len(data)} independent test sequences from fresh state.", flush=True)
    log = evaluate_sequences(data, effective, selected_hyperparameters=(
        None if selection is None else selection["selected_hyperparameters"]))
    pairs = target_quantiles(effective)
    summary = summarize_results(log, pairs)
    for pair in pairs:
        losses = [item["evaluation_results"][pair]["avg_quantile_loss"] for item in log.values()]
        summary[pair]["avg_quantile_loss_mean"] = float(np.mean(losses))
        summary[pair]["avg_quantile_loss_std"] = float(np.std(losses))
    destination = Path(effective["saving_dir"]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if selection is not None:
        selection_path = destination / "selected_hyperparameters.pkl"
        compact = {key: selection[key] for key in (
            "selected_hyperparameters", "scope", "selection_rule", "method", "model", "upstream_commit")}
        _save_pickle(selection_path, compact)
        effective["selected_hyperparameters_path"] = str(selection_path)
        if requested["tuning"]["enabled"]:
            _save_pickle(destination / "tuning_results.pkl", selection)
    OmegaConf.save(OmegaConf.create(requested), destination / "requested_config.yaml", resolve=True)
    OmegaConf.save(OmegaConf.create(effective), destination / "resolved_config.yaml", resolve=True)
    _save_pickle(destination / "log.pkl", log)
    _save_pickle(destination / "summary_results.pkl", summary)
    for pair, metrics in summary.items():
        print(f"{method_name} {pair}: coverage={metrics['avg_coverage_mean']:.4f}, "
              f"width={metrics['avg_interval_width_mean']}, empty_rate={metrics['empty_set_rate']:.4f}", flush=True)
    if effective["plotting"]["plotting"]:
        plot_aci_results(log, pairs, effective["plotting"]["plotting_seq_len"],
                         destination / "plots", method_name=method_name)
    return log
