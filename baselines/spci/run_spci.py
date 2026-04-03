from omegaconf import OmegaConf
import os
import copy
import torch
import numpy as np
from tqdm import tqdm
from sklearn_quantile import RandomForestQuantileRegressor, SampleRandomForestQuantileRegressor
from dscp.data import ConformalPredictionData
from utils.utils import load_data, save_data, read_setup, flatten_and_sort
from utils.utils import compute_coverage, compute_interval_width, compute_winkler_score


def run_spci_experiment(config_path):

    config = OmegaConf.load(config_path)
    os.makedirs(config.saving_dir, exist_ok=True)

    # load data
    data = load_data(config.data.data_path) # load predictor results here
    base_predictor, data_type = read_setup(config.data.data_path)
    cpd = ConformalPredictionData(data)
    cpd.prepare_quantile_prediction_datasets(config.model.window_size, 
                             config.model.prediction_step, 
                             config.data.train_ratio, 
                             config.data.valid_ratio, 
                             config.data.normalize)

    log = dict()

    print("Experiment setup")
    print("Method: SPCI")
    print("Base predictor: {}".format(base_predictor))
    print("Data: {}".format(data_type))
    print("{} independent sequences".format(len(cpd.dataset)))
    
    for key, item in tqdm(cpd.dataset.items(), desc="repetition over independent sequences"):

        target_quantiles = np.array(flatten_and_sort(config.model.target_quantiles)) # sorted array of target quantile levels

        # SPCI does not need validation set, therefore combine train and validation set
        train_dataset = item["train_dataset"]
        valid_dataset = item["valid_dataset"]
        test_dataset = item["test_dataset"]
        train_strided_x, train_strided_residual, train_strided_y, \
            train_target_residual, train_target_y, train_target_preds = train_dataset[:]
        valid_strided_x, valid_strided_residual, valid_strided_y, \
            valid_target_residual, valid_target_y, valid_target_preds = valid_dataset[:]
        strided_residual = torch.cat([train_strided_residual, valid_strided_residual], dim=0).numpy() # (data_size, window_size)
        target_residual = torch.cat([train_target_residual, valid_target_residual], dim=0).flatten().numpy()  # (data_size, )

        # model init
        # target_quantiles must be sorted array
        if len(train_dataset) > 10000:
            qrf = SampleRandomForestQuantileRegressor(n_estimators=config.model.n_estimators,
                                                      max_depth=config.model.max_depth,
                                                      criterion=config.model.criterion,
                                                      n_jobs=-1,
                                                      q=target_quantiles)
        else:
            qrf = RandomForestQuantileRegressor(n_estimators=config.model.n_estimators,
                                                max_depth=config.model.max_depth,
                                                criterion=config.model.criterion,
                                                n_jobs=-1,
                                                q=target_quantiles)
            
        # train Quantile Random Forest
        qrf.fit(strided_residual, target_residual)

        # predict residuals using the trained Quantile Random Forest
        test_strided_x, test_strided_residual, test_strided_y, \
            test_target_residual, test_target_y, test_target_preds = test_dataset[:]
        test_strided_residual = test_strided_residual.numpy()
        pred_quantile_value = qrf.predict(test_strided_residual) # (len(target_quantiles), test_size)

        evaluation_results = dict()

        # evaluation
        for i in range(len(config.model.target_quantiles)):
            confidence_pair = (float(target_quantiles[-(i+1)]),float(target_quantiles[i]))
            evaluation_results[confidence_pair] = {"coverage": [], 
                                                   "interval_width": [],
                                                   "winkler_score" : [],
                                                   "upper_interval" : [],
                                                   "lower_interval" : [],
                                                   "y" : [],
                                                   "predictions" : []}
            hi = pred_quantile_value[-(i+1), :] # (batch_size,)
            lo = pred_quantile_value[i, :] # (batch_size,)

            coverage = compute_coverage(hi, lo, test_target_residual)
            interval_width = compute_interval_width(hi, lo, normalized_std=None)
            winkler_score = compute_winkler_score(hi, lo, 
                                                  test_target_y, 
                                                  test_target_preds, 
                                                  confidence_pair, 
                                                  normalized_params=None)

            evaluation_results[confidence_pair]["upper_interval"] = hi
            evaluation_results[confidence_pair]["lower_interval"] = lo
            evaluation_results[confidence_pair]["coverage"] = coverage
            evaluation_results[confidence_pair]["interval_width"] = interval_width
            evaluation_results[confidence_pair]["winkler_score"].extend(winkler_score)
            evaluation_results[confidence_pair]["y"].extend(test_target_y.flatten().tolist())
            evaluation_results[confidence_pair]["predictions"].extend(test_target_preds.flatten().tolist())
            
        for confidence_pair in config.model.target_quantiles:
            avg_coverage = np.mean(evaluation_results[tuple(confidence_pair)]["coverage"])
            avg_interval_width = np.mean(evaluation_results[tuple(confidence_pair)]["interval_width"])
            avg_winkler_score = np.mean(evaluation_results[tuple(confidence_pair)]["winkler_score"])
            print("avg coverage: {}".format(avg_coverage))
            print("avg interval width: {}".format(avg_interval_width))
            print("avg winkler score: {}".format(avg_winkler_score))
            evaluation_results[tuple(confidence_pair)]["avg_coverage"] = avg_coverage
            evaluation_results[tuple(confidence_pair)]["avg_interval_width"] = avg_interval_width
            evaluation_results[tuple(confidence_pair)]["avg_winkler_score"] = avg_winkler_score

        log[key] = {"evaluation_results" : evaluation_results}
        save_data(os.path.join(config.saving_dir, "log.pkl"), log)
                    