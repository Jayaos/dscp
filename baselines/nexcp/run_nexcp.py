from omegaconf import OmegaConf

from utils.experiment_config import load_experiment_config
import os
import numpy as np
import torch
from tqdm import tqdm

from baselines.nexcp.model import estimate_residual_interval, context_slice
from utils.utils import load_data, save_data, read_setup
from utils.reporting import (
    compute_coverage,
    compute_interval_width,
    compute_winkler_score,
    summarize_evaluation_results,
)
from utils.plotting import plot_cp_prediction_intervals


def run_nexcp(config_path):
    config = load_experiment_config(config_path)
    os.makedirs(config.saving_dir, exist_ok=True)

    data = load_data(config.data.data_path)
    base_predictor, data_type = read_setup(config.data.data_path)

    target_quantiles = config.model.target_quantiles
    log = {}

    print("Experiment setup")
    print("Method: NexCP")
    print("Base predictor: {}".format(base_predictor))
    print("Data: {}".format(data_type))
    print("{} independent sequences".format(len(data)))

    for key, item in tqdm(data.items(), desc="repetition over independent sequences"):
        y = np.asarray(item["heldout_y"], dtype=float).reshape(-1)
        predictions = np.asarray(item["heldout_predictions"], dtype=float).reshape(-1)
        residuals = y - predictions

        calibration_size = int(np.floor(len(y) * config.data.calibration_ratio))

        if config.model.max_past is None:
            max_past = calibration_size
        else:
            max_past = config.model.max_past

        evaluation_results = {
            tuple(confidence_pair): {"coverage": [],
                                     "interval_width": [],
                                     "winkler_score": [],
                                     "upper_interval": [],
                                     "lower_interval": [],
                                     "target_y": [],
                                     "target_predictions": []}
            for confidence_pair in target_quantiles
        }

        for target_idx in tqdm(range(calibration_size, len(y)), desc="{} test points".format(key), leave=False):
            context_slice_idx = context_slice(target_idx, max_past)
            residual_history = residuals[context_slice_idx]
            target_y = torch.tensor([y[target_idx]], dtype=torch.float32)
            target_prediction = torch.tensor([predictions[target_idx]], dtype=torch.float32)
            target_residual = target_y - target_prediction

            for confidence_pair in target_quantiles:
                tuple_confidence_pair = tuple(confidence_pair)
                target_coverage = max(tuple_confidence_pair) - min(tuple_confidence_pair)
                alpha = 1.0 - target_coverage
                lo_value, hi_value = estimate_residual_interval(residual_history,
                                                                alpha,
                                                                config.model.rho)
                lo = torch.tensor([lo_value], dtype=torch.float32)
                hi = torch.tensor([hi_value], dtype=torch.float32)

                this_coverage = compute_coverage(hi, lo, target_residual)
                this_interval_width = compute_interval_width(hi, lo, normalized_std=None)
                this_winkler_score = compute_winkler_score(hi,
                                                           lo,
                                                           target_y,
                                                           target_prediction,
                                                           tuple_confidence_pair,
                                                           normalized_params=None)

                evaluation_results[tuple_confidence_pair]["upper_interval"].extend(hi.tolist())
                evaluation_results[tuple_confidence_pair]["lower_interval"].extend(lo.tolist())
                evaluation_results[tuple_confidence_pair]["coverage"].extend(this_coverage)
                evaluation_results[tuple_confidence_pair]["interval_width"].extend(this_interval_width)
                evaluation_results[tuple_confidence_pair]["winkler_score"].extend(this_winkler_score)
                evaluation_results[tuple_confidence_pair]["target_y"].extend(target_y.tolist())
                evaluation_results[tuple_confidence_pair]["target_predictions"].extend(target_prediction.tolist())

        for confidence_pair in target_quantiles:
            tuple_confidence_pair = tuple(confidence_pair)
            target_alpha = max(tuple_confidence_pair) - min(tuple_confidence_pair)
            avg_coverage = np.mean(evaluation_results[tuple_confidence_pair]["coverage"])
            avg_delta_coverage = avg_coverage - target_alpha
            avg_interval_width = np.mean(evaluation_results[tuple_confidence_pair]["interval_width"])
            avg_winkler_score = np.mean(evaluation_results[tuple_confidence_pair]["winkler_score"])
            print("{} avg coverage: {}".format(key, avg_coverage))
            print("{} avg delta coverage: {}".format(key, avg_delta_coverage))
            print("{} avg interval width: {}".format(key, avg_interval_width))
            print("{} avg winkler score: {}".format(key, avg_winkler_score))
            evaluation_results[tuple_confidence_pair]["avg_coverage"] = avg_coverage
            evaluation_results[tuple_confidence_pair]["avg_delta_coverage"] = avg_delta_coverage
            evaluation_results[tuple_confidence_pair]["avg_interval_width"] = avg_interval_width
            evaluation_results[tuple_confidence_pair]["avg_winkler_score"] = avg_winkler_score

        log[key] = {"evaluation_results": evaluation_results}
        save_data(os.path.join(config.saving_dir, "log.pkl"), log)

    summary_results = summarize_evaluation_results(log, target_quantiles)

    for tuple_confidence_pair, summary in summary_results.items():
        print("Summary for confidence pair {}".format(tuple_confidence_pair))
        print("avg_coverage mean: {}, std: {}".format(
            summary["avg_coverage_mean"],
            summary["avg_coverage_std"])
        )
        print("avg_delta_coverage mean: {}, std: {}".format(
            summary["avg_delta_coverage_mean"],
            summary["avg_delta_coverage_std"])
        )
        print("avg_interval_width mean: {}, std: {}".format(
            summary["avg_interval_width_mean"],
            summary["avg_interval_width_std"])
        )
        print("avg_winkler_score mean: {}, std: {}".format(
            summary["avg_winkler_score_mean"],
            summary["avg_winkler_score_std"])
        )

    save_data(os.path.join(config.saving_dir, "summary_results.pkl"), summary_results)

    if OmegaConf.select(config, "plotting.plotting", default=False):
        plot_cp_prediction_intervals(log,
                                     target_quantiles,
                                     config.plotting.plotting_seq_len,
                                     os.path.join(config.saving_dir, "plots"))
