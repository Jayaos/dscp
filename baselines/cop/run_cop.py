"""COP experiment runner with calibration-only selection and sequential feedback."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import pickle
import time

import numpy as np
from omegaconf import OmegaConf

from baselines.aci.model import positive_integer
from baselines.aci.reporting import evaluate_intervals, summarize_results, plot_aci_results
from baselines.cop.config import validate_config, candidates, target_quantiles, stream_alphas, _alpha
from baselines.cop.data import prepare_sequence, interval_endpoints
from baselines.cop.model import predict_thresholds, validate_model, PAPER, UPSTREAM_SOURCE, UPSTREAM_COMMIT
from utils.experiment_config import load_experiment_config
from utils.forecast_data import canonicalize_forecast_data

SELECTION_RULE = "minimum summed tail pinball loss subject to coverage floor; minimum loss fallback"


def _map_tasks(function, tasks, workers):
    if workers == 1:
        return dict(function(task) for task in tasks)
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        return dict(executor.map(function, tasks))


def _serializable(value):
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _pinball(scores, thresholds, alpha):
    with np.errstate(over="ignore", invalid="ignore"):
        residual = scores - thresholds
        values = np.maximum((1-alpha) * residual, -alpha * residual)
    if not np.isfinite(values).all():
        raise RuntimeError("COP pinball loss overflowed.")
    return float(np.sum(values / len(values)))


def _predict_streams(prepared, model, pair):
    """One independent COP state per tail; each path sees only its past scores."""
    offset = prepared["prefix_length"]
    outputs, thresholds, scores = {}, {}, {}
    for stream, alpha in stream_alphas(pair, model["score_type"]).items():
        full_scores = prepared["scores"][stream]
        result = predict_thresholds(full_scores, alpha, model)
        outputs[stream] = {
            name: value[offset:] if isinstance(value, np.ndarray) else value
            for name, value in result.items()
        }
        thresholds[stream] = outputs[stream]["thresholds"]
        scores[stream] = full_scores[offset:]
    lower, upper, raw_lower, raw_upper, empty = interval_endpoints(prepared["predictions"], thresholds)
    return {"outputs": outputs, "thresholds": thresholds, "scores": scores,
            "lower": lower, "upper": upper, "raw_lower": raw_lower, "raw_upper": raw_upper, "empty": empty}


def _loss(path, model, pair, start=0):
    loss = sum(_pinball(path["scores"][stream][start:], path["thresholds"][stream][start:], alpha)
               for stream, alpha in stream_alphas(pair, model["score_type"]).items())
    if not np.isfinite(loss):
        raise RuntimeError("Summed COP pinball loss overflowed.")
    return loss


def select_trial(trials):
    valid = [trial for trial in trials if trial["status"] == "ok"]
    if not valid:
        reasons = "; ".join(dict.fromkeys(trial["failure"] for trial in trials))
        raise RuntimeError("All COP hyperparameter candidates failed: " + reasons)
    eligible = [trial for trial in valid if trial["eligible"]]
    return min(eligible or valid, key=lambda trial: trial["quantile_loss"])


def _tune_sequence(task):
    key, item, config = task
    prepared = prepare_sequence(item, config, key, "calibration")
    warmup = config["tuning"]["warmup_steps"]
    if warmup is None:
        warmup = max(config["tuning"]["window_size_grid"]) + 1
    if warmup >= len(prepared["y"]):
        raise ValueError(f"Sequence {key!r} has {len(prepared['y'])} calibration points, "
                         f"but tuning warmup is {warmup}. Reduce the window grid, set "
                         "tuning.warmup_steps explicitly, or disable tuning.")
    selections, records = {}, {}
    for pair in target_quantiles(config):
        trials = []
        for trial_id, params in enumerate(candidates(config)):
            model = {**config["model"], **params}
            trial = {"trial_id": trial_id, "hyperparameters": params}
            try:
                path = _predict_streams(prepared, model, pair)
                covered = (path["lower"][warmup:] <= prepared["y"][warmup:]) & (
                    prepared["y"][warmup:] <= path["upper"][warmup:])
                coverage = float(np.mean(covered))
                loss = _loss(path, model, pair, warmup)
                trial.update({"status": "ok", "avg_coverage": coverage, "quantile_loss": loss,
                              "eligible": coverage >= 1-_alpha(pair)-config["tuning"]["coverage_tolerance"],
                              "num_empty_sets": int(path["empty"][warmup:].sum())})
            except RuntimeError as exc:
                trial.update({"status": "failed", "eligible": False, "failure": str(exc)})
            trials.append(trial)
        selected = select_trial(trials)
        selections[pair] = deepcopy(selected["hyperparameters"])
        records[pair] = {"selected_trial_id": selected["trial_id"], "all_trials": trials,
                         "coverage_constraint_satisfied": selected["eligible"]}
    return key, {"selected_hyperparameters": selections, "coverage_results": records,
                 "calibration_boundaries": prepared["boundaries"], "warmup_steps": warmup,
                 "validation_target_indices": prepared["target_indices"][warmup:].tolist()}


def tune_hyperparameters(data, config):
    """Repository selection protocol, restricted to the calibration prefix."""
    config = validate_config(config)
    data = canonicalize_forecast_data(data)
    if not data:
        raise ValueError("The forecast artifact must contain at least one sequence.")
    records = _map_tasks(_tune_sequence, [(key, item, config) for key, item in data.items()],
                         config["num_cores"])
    return {"selected_hyperparameters": {key: record["selected_hyperparameters"] for key, record in records.items()},
            "sequence_results": records, "scope": "per_sequence_and_coverage_level; shared_settings_between_tails",
            "selection_rule": SELECTION_RULE, "method": "COP", "model": deepcopy(config["model"]),
            "upstream_commit": UPSTREAM_COMMIT}


def _checked_selection(selection, pairs):
    if selection is None:
        return
    if not isinstance(selection, dict) or set(selection) != set(pairs):
        raise ValueError("Selected coverage levels do not match the requested configuration.")
    allowed = {"learning_rate", "window_size", "scale"}
    for params in selection.values():
        if not isinstance(params, dict) or set(params) != allowed:
            raise ValueError("Selected COP parameters must contain only learning_rate, window_size and scale.")


def evaluate_sequence(key, item, config, selected_hyperparameters=None):
    config = validate_config(config)
    prepared = prepare_sequence(item, config, key, "test")
    pairs = target_quantiles(config)
    _checked_selection(selected_hyperparameters, pairs)
    started = time.perf_counter()
    results, models = {}, {}
    for pair in pairs:
        selected = {} if selected_hyperparameters is None else selected_hyperparameters[pair]
        model = validate_model({**config["model"], **selected})
        path = _predict_streams(prepared, model, pair)
        result = evaluate_intervals(prepared["y"], prepared["predictions"], path["lower"], path["upper"],
                                    _alpha(pair), prepared["target_indices"])
        result.update({"score_thresholds": _serializable(path["thresholds"]),
                       "scores": _serializable(path["scores"]),
                       "raw_lower_interval": path["raw_lower"].tolist(),
                       "raw_upper_interval": path["raw_upper"].tolist(),
                       "avg_quantile_loss": _loss(path, model, pair),
                       "online_diagnostics": _serializable(path["outputs"])})
        results[pair], models[pair] = result, model
    return {"evaluation_results": results, "metadata": {
        "method": "COP", "boundaries": prepared["boundaries"], "models_by_coverage": models,
        "target_indices": prepared["target_indices"].tolist(),
        "initialization": config["model"]["initialization"], "replayed_calibration_size": prepared["prefix_length"],
        "hyperparameter_scope": "per_sequence_and_coverage_level; shared_settings_between_tails",
        "state_scope": "independent_sequence_coverage_and_tail",
        "feedback_protocol": "one_step_sequential", "score_type": config["model"]["score_type"],
        "interval_scale": "original_response", "residual_quantile_scale": "original_residual",
        "empty_set_handling": "retain_crossings_as_empty; never_reorder_endpoints",
        "evaluation_seconds": time.perf_counter()-started,
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
    tasks = [(key, item, config, None if selected_hyperparameters is None else selected_hyperparameters[key])
             for key, item in data.items()]
    return _map_tasks(_evaluate_task, tasks, workers)


def _save_pickle(path, value):
    with Path(path).open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _load_selection(path, model):
    with Path(path).open("rb") as stream:
        selection = pickle.load(stream)
    if not isinstance(selection, dict) or selection.get("method") != "COP":
        raise ValueError("Saved selection belongs to a different method.")
    for name, value in model.items():
        if name not in {"learning_rate", "window_size", "scale"} and selection.get("model", {}).get(name) != value:
            raise ValueError(f"Saved selection used a different model.{name}.")
    return selection


def run_cop(config, num_cores=None):
    if isinstance(config, (str, Path)):
        config = load_experiment_config(config)
    requested = validate_config(config)
    if num_cores is not None:
        requested["num_cores"] = positive_integer(num_cores, "num_cores")
    if not requested["data"].get("data_path") or not requested.get("saving_dir"):
        raise ValueError("data.data_path and saving_dir are required.")
    with Path(requested["data"]["data_path"]).open("rb") as stream:
        data = pickle.load(stream)
    selection = None
    if requested["tuning"]["enabled"]:
        print("COP: selecting parameters using calibration data only.", flush=True)
        selection = tune_hyperparameters(data, requested)
        failures = sum(trial["status"] == "failed"
                       for sequence in selection["sequence_results"].values()
                       for result in sequence["coverage_results"].values()
                       for trial in result["all_trials"])
        if failures:
            print(f"COP: recorded {failures} failed candidates in tuning_results.pkl.", flush=True)
    elif requested.get("selected_hyperparameters_path"):
        selection = _load_selection(requested["selected_hyperparameters_path"], requested["model"])
    effective = deepcopy(requested)
    effective["tuning"]["enabled"] = False
    print(f"COP: evaluating {len(data)} sequences with {effective['model']['initialization']}.", flush=True)
    log = evaluate_sequences(data, effective, selected_hyperparameters=(
        None if selection is None else selection["selected_hyperparameters"]))
    pairs = target_quantiles(effective)
    summary = summarize_results(log, pairs)
    for pair in pairs:
        losses = [record["evaluation_results"][pair]["avg_quantile_loss"] for record in log.values()]
        summary[pair]["avg_quantile_loss_mean"] = float(np.mean(losses))
        summary[pair]["avg_quantile_loss_std"] = float(np.std(losses))
    destination = Path(effective["saving_dir"]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if selection is not None:
        path = destination / "selected_hyperparameters.pkl"
        compact = {name: selection[name] for name in (
            "selected_hyperparameters", "scope", "selection_rule", "method", "model", "upstream_commit")}
        _save_pickle(path, compact)
        effective["selected_hyperparameters_path"] = str(path)
        if requested["tuning"]["enabled"]:
            _save_pickle(destination / "tuning_results.pkl", selection)
    OmegaConf.save(OmegaConf.create(requested), destination / "requested_config.yaml", resolve=True)
    OmegaConf.save(OmegaConf.create(effective), destination / "resolved_config.yaml", resolve=True)
    _save_pickle(destination / "log.pkl", log)
    _save_pickle(destination / "summary_results.pkl", summary)
    for pair, metrics in summary.items():
        print(f"COP {pair}: coverage={metrics['avg_coverage_mean']:.4f}, "
              f"width={metrics['avg_interval_width_mean']}, empty_rate={metrics['empty_set_rate']:.4f}", flush=True)
    if effective["plotting"]["plotting"]:
        plot_aci_results(log, pairs, effective["plotting"]["plotting_seq_len"],
                         destination / "plots", method_name="COP")
    return log
