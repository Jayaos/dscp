"""DSCP-QR-style shared selection, adapted to ACI's calibration history.

No neural training or checkpoint split is needed. Final test values are never
used for selection. Full/empty sets retain the agreed ACI reporting convention.
"""

from copy import deepcopy
from itertools import product
import math
from numbers import Integral

from baselines.aci.model import finite_number, positive_integer

SELECTION_RULE = "coverage gap strictly above threshold for every sequence/level; minimum mean finite-interval Winkler"


def validate_tuning(tuning, model):
    allowed = {"validation_ratio", "initialization_ratio", "gamma_grid",
               "window_size_grid", "coverage_tolerance", "delta_threshold",
               "num_sequences", "sequence_key", "sequence_index", "top_k"}
    if set(tuning) - allowed:
        raise ValueError("Unknown ACI tuning settings: " + ", ".join(sorted(set(tuning) - allowed)))
    ratio = finite_number(tuning.setdefault("validation_ratio", .2), "tuning.validation_ratio")
    if not 0 < ratio < 1:
        raise ValueError("tuning.validation_ratio must be strictly between zero and one.")
    initial = tuning.setdefault("initialization_ratio", None)
    if initial is not None and not 0 < finite_number(initial, "tuning.initialization_ratio") < 1:
        raise ValueError("tuning.initialization_ratio must be strictly between zero and one.")
    # Accept existing configs, but migrate their field to QR's signed gap threshold.
    if "coverage_tolerance" in tuning:
        tolerance = finite_number(tuning.pop("coverage_tolerance"), "tuning.coverage_tolerance")
        if not 0 <= tolerance < 1:
            raise ValueError("tuning.coverage_tolerance must be in [0, 1).")
        if "delta_threshold" in tuning and tuning["delta_threshold"] != -tolerance:
            raise ValueError("Use tuning.delta_threshold instead of conflicting coverage_tolerance.")
        tuning.setdefault("delta_threshold", -tolerance)
    finite_number(tuning.setdefault("delta_threshold", -.01), "tuning.delta_threshold")
    count = tuning.setdefault("num_sequences", "all")
    if isinstance(count, str) and count.strip().lower() == "all":
        tuning["num_sequences"] = "all"
    else:
        tuning["num_sequences"] = positive_integer(count, "tuning.num_sequences")
    index = tuning.setdefault("sequence_index", 0)
    if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
        raise ValueError("tuning.sequence_index must be a nonnegative integer.")
    key = tuning.setdefault("sequence_key", None)
    if key is not None and not isinstance(key, str):
        raise ValueError("tuning.sequence_key must be a string or null.")
    tuning["top_k"] = positive_integer(tuning.setdefault("top_k", 3), "tuning.top_k")
    tuning.setdefault("gamma_grid", [model["gamma"]])
    tuning.setdefault("window_size_grid", [model["window_size"]])
    for name in ("gamma_grid", "window_size_grid"):
        if not isinstance(tuning[name], list) or not tuning[name]:
            raise ValueError(f"tuning.{name} must be a nonempty list.")
    return tuning


def validate_tuning_config(config):
    """Validate dedicated search settings separately from normal run configs."""
    from baselines.aci.run_aci import _plain_config, validate_config, _estimator, target_quantiles

    config = _plain_config(config)
    tuning = config.pop("tuning", {})
    if not isinstance(tuning, dict):
        raise ValueError("tuning must be a mapping.")
    config = validate_config(config)
    config["tuning"] = validate_tuning(tuning, config["model"])
    pair = target_quantiles(config)[0]
    for gamma, window in product(tuning["gamma_grid"], tuning["window_size_grid"]):
        _estimator({**config["model"], "gamma": gamma, "window_size": window}, pair)
    return config


def choose_sequence_keys(data, tuning):
    """Match QR's sorted subset / all / explicit-key precedence."""
    keys = sorted(data)
    if not keys:
        raise ValueError("The forecast artifact must contain at least one sequence.")
    key = tuning["sequence_key"]
    if key is not None:
        if key not in data:
            raise ValueError(f"Sequence key {key!r} was not found.")
        return [key]
    if tuning["num_sequences"] == "all":
        return keys
    start, count = tuning["sequence_index"], tuning["num_sequences"]
    if start >= len(keys) or start + count > len(keys):
        raise ValueError(f"Requested {count} sequences at index {start}; only {len(keys)} sequences exist.")
    return keys[start:start + count]


def summarize_candidate(sequence_results, pairs, delta_threshold):
    """Equal weight per sequence/level, strict QR coverage filter, no fallback.

    ACI efficiency excludes full/empty sets while coverage includes them.
    Undefined efficiency makes a candidate unrankable; counts remain in output.
    """
    sequences = {}
    for key, results in sequence_results.items():
        pair_metrics, scores, passed = {}, [], True
        for pair in pairs:
            metrics = results[tuple(pair)]
            target = float(max(pair) - min(pair))  # QR's exact target convention.
            gap = metrics["avg_coverage"] - target
            winkler = metrics["avg_winkler_score"]
            scores.append(math.inf if winkler is None else float(winkler))
            passed = passed and gap > delta_threshold
            pair_metrics[str(tuple(pair))] = {
                "avg_coverage": metrics["avg_coverage"], "target_coverage": target,
                "avg_delta_coverage": gap, "avg_interval_width": metrics["avg_interval_width"],
                "avg_winkler_score": winkler,
                **{name: metrics[name] for name in (
                    "num_full_sets", "num_empty_sets", "num_finite_intervals",
                    "full_set_rate", "empty_set_rate")},
            }
        score = math.fsum(value / len(scores) for value in scores)
        sequences[key] = {"pair_metrics": pair_metrics, "selection_score": score,
                          "positive_delta_coverage": bool(passed)}
    pair_metrics = {}
    for pair in pairs:
        name = str(tuple(pair))
        rows = [result["pair_metrics"][name] for result in sequences.values()]
        pair_metrics[name] = {}
        for metric in rows[0]:
            values = [row[metric] for row in rows]
            pair_metrics[name][metric] = (None if any(value is None for value in values)
                                         else math.fsum(value / len(values) for value in values))
    score = math.fsum(result["selection_score"] / len(sequences) for result in sequences.values())
    passed = all(result["positive_delta_coverage"] for result in sequences.values())
    return {"sequence_results": sequences, "num_sequences_evaluated": len(sequences),
            "pair_metrics": pair_metrics, "selection_score": score,
            "positive_delta_coverage": passed, "coverage_eligible": passed and math.isfinite(score),
            "evaluation_split": "calibration_tail", "final_test_evaluated": False,
            "metrics_scope": {"coverage": "all_validation_predictions",
                              "selection_score": "mean finite_nonempty Winkler; equal sequence/level weights"}}


def tune_hyperparameters(data, config):
    from baselines.aci.data import prepare_sequence
    from baselines.aci.run_aci import target_quantiles, _evaluate_prepared
    from utils.forecast_data import canonicalize_forecast_data

    config = validate_tuning_config(config)
    tuning, pairs = config["tuning"], target_quantiles(config)
    # Select before inspecting records: unused series never affect a tuning run.
    keys = choose_sequence_keys(data, tuning)
    selected_data = canonicalize_forecast_data({key: data[key] for key in keys})
    prepared = {key: prepare_sequence(item, config, key, split="validation")
                for key, item in selected_data.items()}
    trials = []
    for index, (gamma, window) in enumerate(product(tuning["gamma_grid"], tuning["window_size_grid"]), 1):
        model = {**config["model"], "gamma": gamma, "window_size": window}
        sequence_results = {key: _evaluate_prepared(item, model, pairs) for key, item in prepared.items()}
        result = summarize_candidate(sequence_results, pairs, tuning["delta_threshold"])
        resolved = deepcopy(config)
        resolved["model"] = model
        del resolved["tuning"]
        trials.append({"trial_index": index, "trial_id": index,
                       "grid_values": {"model.gamma": gamma, "model.window_size": window},
                       "gamma": gamma, "window_size": window,
                       "eligible": result["coverage_eligible"], "sequence_keys": keys,
                       "result": result, "resolved_config": resolved,
                       "evaluation_split": "calibration_tail", "final_test_evaluated": False})
    eligible = [trial for trial in trials if trial["result"]["coverage_eligible"]]
    ranked = sorted(eligible, key=lambda trial: (trial["result"]["selection_score"], trial["trial_index"]))
    selected = ranked[0] if ranked else None
    return {
        "method": "aci", "sequence_keys": keys, "num_sequences": len(keys),
        "num_trials": len(trials), "num_eligible_trials": len(eligible),
        "num_positive_delta_coverage_trials": sum(trial["result"]["positive_delta_coverage"] for trial in trials),
        "top_k": tuning["top_k"], "top_trials": ranked[:tuning["top_k"]], "all_trials": trials,
        "selected_hyperparameters": (None if selected is None else
                                     {"gamma": selected["gamma"], "window_size": selected["window_size"]}),
        "selected_trial_id": None if selected is None else selected["trial_index"],
        "delta_threshold": tuning["delta_threshold"],
        "scope": "shared_across_selected_sequences_and_coverage_levels", "selection_rule": SELECTION_RULE,
        "selection_message": ("Eligible candidates ranked by mean finite-interval Winkler score." if ranked else
                              "No eligible candidate; no fallback selected and final test was not evaluated."),
        "evaluation_split": "calibration_tail", "final_test_evaluated": False,
        "tuning_settings": deepcopy(tuning),
        "tuning_protocol": {"initialization_ratio": tuning["initialization_ratio"],
                            "validation_ratio": tuning["validation_ratio"],
                            "history_initialization": "past calibration scores only; fresh alpha per trial",
                            "checkpoint_selection": None, "final_test_evaluated": False},
        "validation_boundaries": {key: item["boundaries"] for key, item in prepared.items()},
        "validation_target_indices": {key: item["target_indices"].tolist() for key, item in prepared.items()},
    }
