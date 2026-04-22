import numpy as np
import torch
from omegaconf import OmegaConf

from baselines.nexcp.model import context_slice, estimate_residual_interval
from sbatch_run_tuning.common import (
    choose_sequence_keys,
    finalize_and_save_results,
    iter_grid_configs,
    load_grid,
    parse_args,
    plain_config,
    resolve_delta_threshold,
    resolve_num_sequences,
    set_global_seed,
    summarize_evaluation_results,
    write_trial_artifacts,
)
from utils.reporting import compute_coverage, compute_interval_width, compute_winkler_score
from utils.utils import load_data


def _resolve_calibration_ratio(config):
    calibration_ratio = OmegaConf.select(config, "data.calibration_ratio", default=None)
    if calibration_ratio is not None:
        return float(calibration_ratio)

    train_ratio = float(OmegaConf.select(config, "data.train_ratio", default=0.0))
    valid_ratio = float(OmegaConf.select(config, "data.valid_ratio", default=0.0))
    return train_ratio + valid_ratio


def _resolve_max_past(config, calibration_size):
    max_past = OmegaConf.select(config, "model.max_past", default=None)
    if max_past is None:
        return calibration_size
    return int(max_past)


def _run_single_trial(config, sequence_data):
    delta_threshold = float(config.tuning.get("delta_threshold", 0.0))
    y = np.asarray(sequence_data["heldout_y"], dtype=float).reshape(-1)
    predictions = np.asarray(sequence_data["heldout_predictions"], dtype=float).reshape(-1)
    if len(y) != len(predictions):
        raise ValueError("heldout_y and heldout_predictions must have the same length.")

    calibration_ratio = _resolve_calibration_ratio(config)
    calibration_size = int(np.floor(len(y) * calibration_ratio))
    if calibration_size <= 0:
        raise ValueError("NexCP tuning requires a positive calibration history.")
    if calibration_size >= len(y):
        raise ValueError("NexCP tuning requires at least one test point.")

    max_past = _resolve_max_past(config, calibration_size)
    residuals = y - predictions
    evaluation_results = {
        tuple(confidence_pair): {
            "coverage": [],
            "interval_width": [],
            "winkler_score": [],
        }
        for confidence_pair in config.model.target_quantiles
    }

    for target_idx in range(calibration_size, len(y)):
        residual_history = residuals[context_slice(target_idx, max_past)]
        target_y = torch.tensor([y[target_idx]], dtype=torch.float32)
        target_prediction = torch.tensor([predictions[target_idx]], dtype=torch.float32)
        target_residual = target_y - target_prediction

        for confidence_pair in config.model.target_quantiles:
            pair_key = tuple(confidence_pair)
            target_coverage = max(pair_key) - min(pair_key)
            alpha = 1.0 - target_coverage
            lo_value, hi_value = estimate_residual_interval(
                residual_history,
                alpha,
                config.model.rho,
            )
            lo = torch.tensor([lo_value], dtype=torch.float32)
            hi = torch.tensor([hi_value], dtype=torch.float32)

            evaluation_results[pair_key]["coverage"].extend(
                compute_coverage(hi, lo, target_residual)
            )
            evaluation_results[pair_key]["interval_width"].extend(
                compute_interval_width(hi, lo, normalized_std=None)
            )
            evaluation_results[pair_key]["winkler_score"].extend(
                compute_winkler_score(
                    hi,
                    lo,
                    target_y,
                    target_prediction,
                    pair_key,
                    normalized_params=None,
                )
            )

    pair_metrics, selection_score, positive_delta_coverage = summarize_evaluation_results(
        evaluation_results,
        config.model.target_quantiles,
        delta_threshold=delta_threshold,
    )

    return {
        "calibration_ratio": calibration_ratio,
        "calibration_size": calibration_size,
        "max_past": max_past,
        "num_test_points": len(y) - calibration_size,
        "pair_metrics": pair_metrics,
        "selection_score": selection_score,
        "positive_delta_coverage": positive_delta_coverage,
    }


def _aggregate_sequence_results(sequence_results: dict, target_quantiles: list) -> dict:
    if not sequence_results:
        raise ValueError("No sequence results were provided for aggregation.")

    ordered_results = list(sequence_results.values())
    pair_metrics = {}
    for confidence_pair in target_quantiles:
        pair_key = str(tuple(confidence_pair))
        pair_metrics[pair_key] = {
            "avg_coverage": float(np.mean([
                item["pair_metrics"][pair_key]["avg_coverage"] for item in ordered_results
            ])),
            "target_coverage": float(np.mean([
                item["pair_metrics"][pair_key]["target_coverage"] for item in ordered_results
            ])),
            "avg_delta_coverage": float(np.mean([
                item["pair_metrics"][pair_key]["avg_delta_coverage"] for item in ordered_results
            ])),
            "avg_interval_width": float(np.mean([
                item["pair_metrics"][pair_key]["avg_interval_width"] for item in ordered_results
            ])),
            "avg_winkler_score": float(np.mean([
                item["pair_metrics"][pair_key]["avg_winkler_score"] for item in ordered_results
            ])),
        }

    return {
        "num_sequences_evaluated": len(sequence_results),
        "sequence_results": sequence_results,
        "mean_calibration_size": float(np.mean([
            item["calibration_size"] for item in ordered_results
        ])),
        "mean_max_past": float(np.mean([
            item["max_past"] for item in ordered_results
        ])),
        "mean_num_test_points": float(np.mean([
            item["num_test_points"] for item in ordered_results
        ])),
        "pair_metrics": pair_metrics,
        "selection_score": float(np.mean([
            item["selection_score"] for item in ordered_results
        ])),
        "positive_delta_coverage": all(
            item["positive_delta_coverage"] for item in ordered_results
        ),
    }


def main():
    args = parse_args("nexcp")
    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    base_config = OmegaConf.load(args.base_config)
    grid, tuning_cfg = load_grid(args.grid_config)

    data = load_data(base_config.data.data_path)
    num_sequences = resolve_num_sequences(tuning_cfg)
    delta_threshold = resolve_delta_threshold(tuning_cfg)
    base_config.tuning = dict(tuning_cfg)
    sequence_keys = choose_sequence_keys(
        data,
        args.sequence_key,
        args.sequence_index,
        num_sequences,
    )

    trials = []
    for trial_index, (trial_config, grid_values) in enumerate(iter_grid_configs(base_config, grid), start=1):
        print(f"[nexcp] starting trial {trial_index} with grid_values={grid_values}", flush=True)
        set_global_seed(args.seed + trial_index)
        sequence_results = {
            sequence_key: _run_single_trial(trial_config, data[sequence_key])
            for sequence_key in sequence_keys
        }
        result = _aggregate_sequence_results(sequence_results, trial_config.model.target_quantiles)
        record = {
            "trial_index": trial_index,
            "sequence_keys": sequence_keys,
            "grid_values": grid_values,
            "result": result,
            "resolved_config": plain_config(trial_config),
        }
        trials.append(record)
        write_trial_artifacts(save_dir, trial_index, trial_config, record)

    positive_trials = [trial for trial in trials if trial["result"]["positive_delta_coverage"]]
    ranked_trials = sorted(positive_trials, key=lambda item: item["result"]["selection_score"])
    top_trials = ranked_trials[: args.top_k]

    payload = {
        "method": "nexcp",
        "base_config_path": str(args.base_config.resolve()),
        "grid_config_path": str(args.grid_config.resolve()),
        "sequence_keys": sequence_keys,
        "num_sequences": len(sequence_keys),
        "delta_threshold": delta_threshold,
        "num_trials": len(trials),
        "num_positive_delta_coverage_trials": len(positive_trials),
        "top_k": args.top_k,
        "top_trials": top_trials,
        "all_trials": trials,
    }
    finalize_and_save_results(save_dir, payload)


if __name__ == "__main__":
    main()
