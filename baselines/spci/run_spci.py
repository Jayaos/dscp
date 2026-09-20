from omegaconf import OmegaConf

from utils.experiment_config import load_experiment_config
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from baselines.spci.data import prepare_spci_data
from baselines.spci.intervals import build_interval_plan, select_intervals
from baselines.spci.model import build_quantile_forest
from utils.utils import load_data, save_data, read_setup
from utils.reporting import compute_coverage, compute_interval_width, compute_winkler_score, summarize_evaluation_results
from utils.plotting import plot_cp_prediction_intervals


def run_spci_experiment(config_path):

    config = load_experiment_config(config_path)
    if config.get("seed", None) is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
    os.makedirs(config.saving_dir, exist_ok=True)
    interval_plan = build_interval_plan(config.model)

    # load data
    data = load_data(config.data.data_path) # load predictor results here
    base_predictor, data_type = read_setup(config.data.data_path)
    cpd = prepare_spci_data(data, config)

    log = dict()

    print("Experiment setup")
    print("Method: SPCI")
    print("Base predictor: {}".format(base_predictor))
    print("Data: {}".format(data_type))
    print("{} independent sequences".format(len(cpd.dataset)))
    
    for key, item in tqdm(cpd.dataset.items(), desc="repetition over independent sequences"):

        train_dataset = item["train_dataset"]
        test_dataset = item["test_dataset"]
        train_strided_x, train_strided_residual, train_strided_y, \
            train_target_x, train_target_residual, train_target_y, train_target_preds = train_dataset[:]
        strided_residual = train_strided_residual.numpy() # (data_size, window_size)
        target_residual = train_target_residual.flatten().numpy()  # (data_size, )

        # model init
        qrf = build_quantile_forest(
            config, len(train_dataset), interval_plan.quantiles
        )
            
        # train Quantile Random Forest
        qrf.fit(strided_residual, target_residual)

        # predict residuals using the trained Quantile Random Forest
        test_strided_x, test_strided_residual, test_strided_y, \
            test_target_x, test_target_residual, test_target_y, test_target_preds = test_dataset[:]
        test_strided_residual = test_strided_residual.numpy()
        pred_quantile_values = np.asarray(qrf.predict(test_strided_residual), dtype=np.float32)
        if pred_quantile_values.shape != (len(interval_plan.quantiles), len(test_dataset)):
            raise ValueError("The quantile forest returned an unexpected prediction shape.")
        intervals = select_intervals(pred_quantile_values, interval_plan)

        evaluation_results = {
            tuple(confidence_pair): {"coverage": [], 
                                     "interval_width": [],
                                     "winkler_score" : [],
                                     "upper_interval" : [],
                                     "lower_interval" : [],
                                     "target_y" : [],
                                     "target_predictions" : []}
            for confidence_pair in config.model.target_quantiles
        }

        if config.data.normalize:
            residuals_noramlized_mu = cpd.data[key]["train_residuals_mu"]
            residuals_noramlized_std = cpd.data[key]["train_residuals_std"]

        # evaluation
        for confidence_pair in config.model.target_quantiles:
            tuple_confidence_pair = tuple(confidence_pair)
            interval = intervals[tuple_confidence_pair]
            hi = torch.as_tensor(interval.upper, dtype=torch.float32)
            lo = torch.as_tensor(interval.lower, dtype=torch.float32)
            evaluation_results[tuple_confidence_pair].update(interval.metadata())

            this_coverage = compute_coverage(hi, lo, test_target_residual)

            if config.data.normalize:
                this_interval_width = compute_interval_width(hi,
                                                             lo,
                                                             normalized_std=residuals_noramlized_std)
                this_winkler_score = compute_winkler_score(hi,
                                                           lo,
                                                           test_target_y,
                                                           test_target_preds,
                                                           interval.score_quantiles,
                                                           normalized_params=(residuals_noramlized_mu,
                                                                              residuals_noramlized_std))
            else:
                this_interval_width = compute_interval_width(hi,
                                                             lo,
                                                             normalized_std=None)
                this_winkler_score = compute_winkler_score(hi,
                                                           lo,
                                                           test_target_y,
                                                           test_target_preds,
                                                           interval.score_quantiles,
                                                           normalized_params=None)

            evaluation_results[tuple_confidence_pair]["upper_interval"].extend(hi.tolist())
            evaluation_results[tuple_confidence_pair]["lower_interval"].extend(lo.tolist())
            evaluation_results[tuple_confidence_pair]["coverage"].extend(this_coverage)
            evaluation_results[tuple_confidence_pair]["interval_width"].extend(this_interval_width)
            evaluation_results[tuple_confidence_pair]["winkler_score"].extend(this_winkler_score)
            evaluation_results[tuple_confidence_pair]["target_y"].extend(test_target_y.flatten().tolist())
            evaluation_results[tuple_confidence_pair]["target_predictions"].extend(test_target_preds.flatten().tolist())

            if config.data.normalize:
                evaluation_results[tuple_confidence_pair]["train_residuals_mu"] = residuals_noramlized_mu
                evaluation_results[tuple_confidence_pair]["train_residuals_std"] = residuals_noramlized_std
            
        for confidence_pair in config.model.target_quantiles:
            tuple_confidence_pair = tuple(confidence_pair)
            target_alpha = max(tuple_confidence_pair) - min(tuple_confidence_pair)
            avg_coverage = np.mean(evaluation_results[tuple_confidence_pair]["coverage"])
            avg_delta_coverage = avg_coverage - target_alpha
            avg_interval_width = np.mean(evaluation_results[tuple_confidence_pair]["interval_width"])
            avg_winkler_score = np.mean(evaluation_results[tuple_confidence_pair]["winkler_score"])
            print("avg coverage: {}".format(avg_coverage))
            print("avg delta coverage: {}".format(avg_delta_coverage))
            print("avg interval width: {}".format(avg_interval_width))
            print("avg winkler score: {}".format(avg_winkler_score))
            evaluation_results[tuple_confidence_pair]["avg_coverage"] = avg_coverage
            evaluation_results[tuple_confidence_pair]["avg_delta_coverage"] = avg_delta_coverage
            evaluation_results[tuple_confidence_pair]["avg_interval_width"] = avg_interval_width
            evaluation_results[tuple_confidence_pair]["avg_winkler_score"] = avg_winkler_score

        log[key] = {"evaluation_results" : evaluation_results}
        save_data(os.path.join(config.saving_dir, "log.pkl"), log)

    summary_results = summarize_evaluation_results(log, config.model.target_quantiles)

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

    if config.plotting.plotting:
        plot_cp_prediction_intervals(log, 
                                     config.model.target_quantiles, 
                                     config.plotting.plotting_seq_len,
                                     os.path.join(config.saving_dir, "plots"))
                    
