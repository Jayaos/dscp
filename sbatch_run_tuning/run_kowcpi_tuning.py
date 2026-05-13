import numpy as np
import torch
from omegaconf import OmegaConf

from baselines.kowcpi.model import KOWCPIResidualIntervalEstimator
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


ALLOWED_GRID_KEYS = {"model.kernel", "model.past_window"}


def _flatten(values):
    return np.asarray(values, dtype=float).reshape(-1)


def _normalization_params(y, train_size):
    train_y = np.asarray(y[:train_size], dtype=float)
    mean = train_y.mean(axis=0)
    std = train_y.std(axis=0) + 1e-8
    return mean, std


def _normalize(values, mean, std):
    return (np.asarray(values, dtype=float) - mean) / std


def _scalar_std(std):
    std_arr = np.asarray(std, dtype=float).reshape(-1)
    if std_arr.size != 1:
        raise ValueError("KOWCPI tuning expects a univariate target.")
    return float(std_arr[0])


def _split_sizes(n, train_ratio, valid_ratio):
    train_size = int(np.floor(n * train_ratio))
    valid_size = int(np.ceil(n * valid_ratio))
    test_size = n - train_size - valid_size
    if train_size <= 0 or valid_size < 0 or test_size <= 0:
        raise ValueError(
            "Invalid split sizes for n={}: train={}, valid={}, test={}.".format(
                n, train_size, valid_size, test_size
            )
        )
    return train_size, valid_size, test_size


def _bandwidth_range_from_config(config):
    value = OmegaConf.select(config, "model.bandwidth_range", default=None)
    if value is None:
        return np.linspace(0.1, 3.0, 30)

    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None

    if isinstance(value, (list, tuple)):
        if len(value) == 3 and isinstance(value[2], int):
            return np.linspace(float(value[0]), float(value[1]), int(value[2]))
        return np.asarray(value, dtype=float)

    raise ValueError("model.bandwidth_range must be null, 'none', a grid list, or [start, stop, num].")


def _glr_params_from_config(config):
    params = OmegaConf.select(config, "model.glr_params", default=None)
    if params is None:
        return None
    return OmegaConf.to_container(params, resolve=True)


def _target_quantiles(config):
    target_quantiles = OmegaConf.select(config, "model.target_quantiles", default=None)
    if target_quantiles is None:
        raise ValueError("model.target_quantiles is required.")
    return [list(pair) for pair in OmegaConf.to_container(target_quantiles, resolve=True)]


def _validate_grid_keys(grid):
    unexpected_keys = sorted(set(grid) - ALLOWED_GRID_KEYS)
    if unexpected_keys:
        raise ValueError(
            "KOWCPI tuning only supports grid keys {}. Unexpected keys: {}".format(
                sorted(ALLOWED_GRID_KEYS),
                unexpected_keys,
            )
        )


def _run_single_trial(config, sequence_data):
    delta_threshold = float(config.tuning.get("delta_threshold", 0.0))
    target_quantiles = _target_quantiles(config)
    raw_y = _flatten(sequence_data["heldout_y"])
    raw_predictions = _flatten(sequence_data["heldout_predictions"])
    if len(raw_y) != len(raw_predictions):
        raise ValueError("heldout_y and heldout_predictions must have the same length.")

    train_size, valid_size, test_size = _split_sizes(
        len(raw_y),
        float(config.data.train_ratio),
        float(config.data.valid_ratio),
    )
    calibration_size = train_size + valid_size

    if bool(config.data.normalize):
        y_mean, y_std = _normalization_params(sequence_data["heldout_y"], train_size)
        y_for_residuals = _flatten(_normalize(sequence_data["heldout_y"], y_mean, y_std))
        predictions_for_residuals = _flatten(
            _normalize(sequence_data["heldout_predictions"], y_mean, y_std)
        )
        residual_normalized_std = _scalar_std(y_std)
        residual_normalization_params = (0.0, residual_normalized_std)
    else:
        y_for_residuals = raw_y
        predictions_for_residuals = raw_predictions
        residual_normalized_std = None
        residual_normalization_params = None

    residuals = y_for_residuals - predictions_for_residuals
    target_residual = torch.tensor(residuals[calibration_size:], dtype=torch.float32)
    target_y = torch.tensor(raw_y[calibration_size:], dtype=torch.float32)
    target_predictions = torch.tensor(raw_predictions[calibration_size:], dtype=torch.float32)

    block_size = int(OmegaConf.select(config, "model.past_window"))
    history_window = OmegaConf.select(
        config,
        "model.history_window",
        default=OmegaConf.select(config, "model.num_resid_used", default=None),
    )
    bandwidth = OmegaConf.select(config, "model.bandwidth", default=None)
    if isinstance(bandwidth, str) and bandwidth.lower() in {"none", "null"}:
        bandwidth = None

    evaluation_results = {
        tuple(confidence_pair): {
            "coverage": [],
            "interval_width": [],
            "winkler_score": [],
        }
        for confidence_pair in target_quantiles
    }
    selected_bandwidths = {}

    for confidence_pair in target_quantiles:
        pair_key = tuple(confidence_pair)
        target_coverage = max(pair_key) - min(pair_key)
        alpha = 1.0 - target_coverage
        use_beta_search = bool(OmegaConf.select(config, "model.use_beta_search", default=True))
        fixed_beta = None if use_beta_search else min(pair_key)
        bandwidth_range = None if bandwidth is not None else _bandwidth_range_from_config(config)
        estimator = KOWCPIResidualIntervalEstimator(
            bandwidth=bandwidth,
            kernel=OmegaConf.select(config, "model.kernel", default="epanechnikov"),
            bandwidth_range=bandwidth_range,
            bins=int(OmegaConf.select(config, "model.bins", default=5)),
            weigh_residuals=bool(OmegaConf.select(config, "model.weigh_residuals", default=False)),
            residual_weight_decay=float(
                OmegaConf.select(config, "model.residual_weight_decay", default=0.995)
            ),
            max_training_blocks=OmegaConf.select(config, "model.max_training_blocks", default=None),
            min_training_blocks=int(OmegaConf.select(config, "model.min_training_blocks", default=2)),
            fixed_beta=fixed_beta,
        )

        lo_values, hi_values = estimator.predict_residual_intervals(
            residuals=residuals,
            calibration_size=calibration_size,
            test_size=test_size,
            alpha=alpha,
            block_size=block_size,
            history_window=history_window,
            use_adaptive_window=bool(OmegaConf.select(config, "model.use_adaptive_window", default=False)),
            glr_params=_glr_params_from_config(config),
            update_with_test=bool(OmegaConf.select(config, "model.update_with_test", default=True)),
        )
        selected_bandwidths[str(pair_key)] = (
            None if estimator.bandwidth is None else float(estimator.bandwidth)
        )

        lo = torch.tensor(lo_values, dtype=torch.float32)
        hi = torch.tensor(hi_values, dtype=torch.float32)

        evaluation_results[pair_key]["coverage"].extend(
            compute_coverage(hi, lo, target_residual)
        )
        evaluation_results[pair_key]["interval_width"].extend(
            compute_interval_width(
                hi,
                lo,
                normalized_std=residual_normalized_std,
            )
        )
        evaluation_results[pair_key]["winkler_score"].extend(
            compute_winkler_score(
                hi,
                lo,
                target_y,
                target_predictions,
                pair_key,
                normalized_params=residual_normalization_params,
            )
        )

    pair_metrics, selection_score, positive_delta_coverage = summarize_evaluation_results(
        evaluation_results,
        target_quantiles,
        delta_threshold=delta_threshold,
    )

    return {
        "train_size": train_size,
        "valid_size": valid_size,
        "calibration_size": calibration_size,
        "test_size": test_size,
        "kernel": OmegaConf.select(config, "model.kernel", default="epanechnikov"),
        "past_window": block_size,
        "history_window": history_window,
        "use_beta_search": bool(OmegaConf.select(config, "model.use_beta_search", default=True)),
        "selected_bandwidths": selected_bandwidths,
        "pair_metrics": pair_metrics,
        "selection_score": selection_score,
        "positive_delta_coverage": positive_delta_coverage,
    }


def _aggregate_sequence_results(sequence_results: dict, target_quantiles: list) -> dict:
    if not sequence_results:
        raise ValueError("No sequence results were provided for aggregation.")

    ordered_results = list(sequence_results.values())
    pair_metrics = {}
    mean_selected_bandwidths = {}
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

        bandwidths = [
            item["selected_bandwidths"][pair_key]
            for item in ordered_results
            if item["selected_bandwidths"][pair_key] is not None
        ]
        mean_selected_bandwidths[pair_key] = (
            None if not bandwidths else float(np.mean(bandwidths))
        )

    return {
        "num_sequences_evaluated": len(sequence_results),
        "sequence_results": sequence_results,
        "mean_train_size": float(np.mean([item["train_size"] for item in ordered_results])),
        "mean_valid_size": float(np.mean([item["valid_size"] for item in ordered_results])),
        "mean_calibration_size": float(np.mean([
            item["calibration_size"] for item in ordered_results
        ])),
        "mean_test_size": float(np.mean([item["test_size"] for item in ordered_results])),
        "mean_selected_bandwidths": mean_selected_bandwidths,
        "pair_metrics": pair_metrics,
        "selection_score": float(np.mean([
            item["selection_score"] for item in ordered_results
        ])),
        "positive_delta_coverage": all(
            item["positive_delta_coverage"] for item in ordered_results
        ),
    }


def main():
    args = parse_args("kowcpi")
    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    base_config = OmegaConf.load(args.base_config)
    grid, tuning_cfg = load_grid(args.grid_config)
    _validate_grid_keys(grid)

    prediction_step = int(OmegaConf.select(base_config, "model.prediction_step", default=1))
    if prediction_step != 1:
        raise NotImplementedError("KOWCPI tuning currently supports prediction_step=1.")

    data = load_data(base_config.data.data_path)
    num_sequences = resolve_num_sequences(tuning_cfg)
    delta_threshold = resolve_delta_threshold(tuning_cfg)
    base_config.tuning = dict(tuning_cfg)
    target_quantiles = _target_quantiles(base_config)
    sequence_keys = choose_sequence_keys(
        data,
        args.sequence_key,
        args.sequence_index,
        num_sequences,
    )

    trials = []
    for trial_index, (trial_config, grid_values) in enumerate(iter_grid_configs(base_config, grid), start=1):
        print(f"[kowcpi] starting trial {trial_index} with grid_values={grid_values}", flush=True)
        set_global_seed(args.seed + trial_index)
        sequence_results = {
            sequence_key: _run_single_trial(trial_config, data[sequence_key])
            for sequence_key in sequence_keys
        }
        result = _aggregate_sequence_results(sequence_results, target_quantiles)
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
        "method": "kowcpi",
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
