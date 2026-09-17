import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from baselines.kowcpi.data import prepare_sequence
from baselines.kowcpi.model import KOWCPIResidualIntervalEstimator
from utils.plotting import plot_cp_prediction_intervals
from utils.reporting import (
    compute_coverage,
    compute_interval_width,
    compute_winkler_score,
    summarize_evaluation_results,
)
from utils.utils import load_data, read_setup, save_data


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


def _evaluation_result_template(target_quantiles):
    return {
        tuple(confidence_pair): {
            "coverage": [],
            "interval_width": [],
            "winkler_score": [],
            "upper_interval": [],
            "lower_interval": [],
            "target_y": [],
            "target_predictions": [],
        }
        for confidence_pair in target_quantiles
    }


def _validate_num_cores(num_cores):
    if num_cores is None:
        return 1
    num_cores = int(num_cores)
    if num_cores < 1:
        raise ValueError("num_cores must be a positive integer.")
    return num_cores


def _ordered_completed_log(log, sequence_keys):
    return {key: log[key] for key in sequence_keys if key in log}


def _run_kowcpi_sequence_worker(args):
    key, item, config_container, target_quantiles = args
    config = OmegaConf.create(config_container)
    return key, _run_kowcpi_sequence(key, item, config, target_quantiles)


def _run_kowcpi_sequence(key, item, config, target_quantiles):
    prepared = prepare_sequence(item, config, split="test")
    residuals = prepared["residuals"]
    raw_y = prepared["raw_y"]
    raw_predictions = prepared["raw_predictions"]
    calibration_size = prepared["calibration_size"]
    test_size = prepared["evaluation_size"]
    residual_normalized_std = prepared["residual_normalized_std"]
    residual_normalization_params = prepared["residual_normalization_params"]
    target_residual = torch.tensor(residuals[calibration_size:], dtype=torch.float32)
    target_y = torch.tensor(raw_y[calibration_size:], dtype=torch.float32)
    target_predictions = torch.tensor(raw_predictions[calibration_size:], dtype=torch.float32)

    block_size = OmegaConf.select(
        config,
        "model.past_window",
        default=OmegaConf.select(config, "model.window_size", default=5),
    )
    history_window = OmegaConf.select(
        config,
        "model.history_window",
        default=OmegaConf.select(config, "model.num_resid_used", default=None),
    )
    bandwidth = OmegaConf.select(config, "model.bandwidth", default=None)
    if isinstance(bandwidth, str) and bandwidth.lower() in {"none", "null"}:
        bandwidth = None

    evaluation_results = _evaluation_result_template(target_quantiles)

    for confidence_pair in target_quantiles:
        tuple_confidence_pair = tuple(confidence_pair)
        target_coverage = max(tuple_confidence_pair) - min(tuple_confidence_pair)
        alpha = 1.0 - target_coverage
        use_beta_search = bool(OmegaConf.select(config, "model.use_beta_search", default=True))
        fixed_beta = None if use_beta_search else min(tuple_confidence_pair)

        print("{} KOWCPI alpha {}".format(key, alpha))
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

        lo, hi = estimator.predict_residual_intervals(
            residuals=residuals,
            calibration_size=calibration_size,
            test_size=test_size,
            alpha=alpha,
            block_size=int(block_size),
            history_window=history_window,
            use_adaptive_window=bool(OmegaConf.select(config, "model.use_adaptive_window", default=False)),
            glr_params=_glr_params_from_config(config),
            update_with_test=bool(OmegaConf.select(config, "model.update_with_test", default=True)),
        )

        lo = torch.tensor(lo, dtype=torch.float32)
        hi = torch.tensor(hi, dtype=torch.float32)

        this_coverage = compute_coverage(hi, lo, target_residual)
        this_interval_width = compute_interval_width(
            hi,
            lo,
            normalized_std=residual_normalized_std,
        )
        this_winkler_score = compute_winkler_score(
            hi,
            lo,
            target_y,
            target_predictions,
            tuple_confidence_pair,
            normalized_params=residual_normalization_params,
        )

        result = evaluation_results[tuple_confidence_pair]
        result["upper_interval"].extend(hi.tolist())
        result["lower_interval"].extend(lo.tolist())
        result["coverage"].extend(this_coverage)
        result["interval_width"].extend(this_interval_width)
        result["winkler_score"].extend(this_winkler_score)
        result["target_y"].extend(target_y.flatten().tolist())
        result["target_predictions"].extend(target_predictions.flatten().tolist())
        if residual_normalization_params is not None:
            result["train_residuals_mu"] = residual_normalization_params[0]
            result["train_residuals_std"] = residual_normalization_params[1]

    for confidence_pair in target_quantiles:
        tuple_confidence_pair = tuple(confidence_pair)
        target_coverage = max(tuple_confidence_pair) - min(tuple_confidence_pair)
        result = evaluation_results[tuple_confidence_pair]
        avg_coverage = float(np.mean(result["coverage"]))
        avg_delta_coverage = avg_coverage - target_coverage
        avg_interval_width = float(np.mean(result["interval_width"]))
        avg_winkler_score = float(np.mean(result["winkler_score"]))

        print("{} avg coverage: {}".format(key, avg_coverage))
        print("{} avg delta coverage: {}".format(key, avg_delta_coverage))
        print("{} avg interval width: {}".format(key, avg_interval_width))
        print("{} avg winkler score: {}".format(key, avg_winkler_score))

        result["avg_coverage"] = avg_coverage
        result["avg_delta_coverage"] = avg_delta_coverage
        result["avg_interval_width"] = avg_interval_width
        result["avg_winkler_score"] = avg_winkler_score

    return {
        "evaluation_results": evaluation_results,
        "metadata": {
            "evaluation_split": "test",
            "evaluation_start": prepared["evaluation_start"],
            "evaluation_end": prepared["evaluation_end"],
            "boundaries": prepared["boundaries"],
        },
    }


def run_kowcpi(config_path, num_cores=1):
    config = OmegaConf.load(config_path)
    os.makedirs(config.saving_dir, exist_ok=True)
    num_cores = _validate_num_cores(num_cores)

    prediction_step = int(OmegaConf.select(config, "model.prediction_step", default=1))
    if prediction_step != 1:
        raise NotImplementedError("This KOWCPI baseline currently supports prediction_step=1.")

    data = load_data(config.data.data_path)
    base_predictor, data_type = read_setup(config.data.data_path)
    target_quantiles = _target_quantiles(config)

    print("Experiment setup")
    print("Method: KOWCPI")
    print("Base predictor: {}".format(base_predictor))
    print("Data: {}".format(data_type))
    print("{} independent sequences".format(len(data)))
    print("{} parallel sequence worker(s)".format(min(num_cores, max(len(data), 1))))

    log = {}
    sequence_keys = list(data.keys())
    log_path = os.path.join(config.saving_dir, "log.pkl")

    if num_cores == 1 or len(data) <= 1:
        for key, item in tqdm(data.items(), desc="repetition over independent sequences"):
            log[key] = _run_kowcpi_sequence(key, item, config, target_quantiles)
            save_data(log_path, _ordered_completed_log(log, sequence_keys))
    else:
        max_workers = min(num_cores, len(data))
        config_container = OmegaConf.to_container(config, resolve=True)
        worker_args = [
            (key, data[key], config_container, target_quantiles)
            for key in sequence_keys
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_kowcpi_sequence_worker, args) for args in worker_args]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="repetition over independent sequences",
            ):
                key, result = future.result()
                log[key] = result
                save_data(log_path, _ordered_completed_log(log, sequence_keys))

        log = _ordered_completed_log(log, sequence_keys)

    summary_results = summarize_evaluation_results(log, target_quantiles)

    for tuple_confidence_pair, summary in summary_results.items():
        print("Summary for confidence pair {}".format(tuple_confidence_pair))
        print(
            "avg_coverage mean: {}, std: {}".format(
                summary["avg_coverage_mean"],
                summary["avg_coverage_std"],
            )
        )
        print(
            "avg_delta_coverage mean: {}, std: {}".format(
                summary["avg_delta_coverage_mean"],
                summary["avg_delta_coverage_std"],
            )
        )
        print(
            "avg_interval_width mean: {}, std: {}".format(
                summary["avg_interval_width_mean"],
                summary["avg_interval_width_std"],
            )
        )
        print(
            "avg_winkler_score mean: {}, std: {}".format(
                summary["avg_winkler_score_mean"],
                summary["avg_winkler_score_std"],
            )
        )

    save_data(os.path.join(config.saving_dir, "summary_results.pkl"), summary_results)

    if OmegaConf.select(config, "plotting.plotting", default=False):
        plot_cp_prediction_intervals(
            log,
            target_quantiles,
            config.plotting.plotting_seq_len,
            os.path.join(config.saving_dir, "plots"),
        )
