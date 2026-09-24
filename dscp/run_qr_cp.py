from omegaconf import OmegaConf

from utils.experiment_config import load_experiment_config
import torch
import os
import copy
import numpy as np
from tqdm import tqdm
from dscp.models.qr_transformer import QuantileRegressionTransformer
from dscp.models.qr_rnn import QuantileRegressionRNN
from dscp.loss import compute_loss_quantile_regression_transformer, compute_loss_quantile_regression_rnn
from dscp.qr_reporting import QuantileCrossingTracker, summarize_qr_results, write_excluded_points
from dscp.data import ConformalPredictionData
from utils.utils import load_data, save_data, read_setup, generate_strided_feature, get_interval_quantile_indices
from utils.reporting import compute_coverage, compute_interval_width, compute_winkler_score, construct_interval_endpoints, summarize_evaluation_results
from utils.plotting import plot_cp_prediction_intervals
from torch.utils.data import DataLoader


def run_transformer_quantile_regression_cp(config_path):

    config = load_experiment_config(config_path)
    os.makedirs(config.saving_dir, exist_ok=True)
    _, pair_to_indices = get_interval_quantile_indices(config.model.target_quantiles)
    exclude_crossings = config.model.get("head_type", "nondecreasing") == "independent"

    # load data
    data = load_data(config.data.data_path) # load predictor results here
    base_predictor, data_type = read_setup(config.data.data_path)
    cpd = ConformalPredictionData(data)
    cpd.prepare_quantile_regression_datasets(config.model.window_size, 
                                             config.model.prediction_step, 
                                             config.data.train_ratio, 
                                             config.data.valid_ratio, 
                                             normalize=config.data.normalize)
    device = config.device
    log = dict()

    print("Experiment setup")
    print("Method: Quantile Regression - Transformer")
    print("Base predictor: {}".format(base_predictor))
    print("Data: {}".format(data_type))
    print("{} independent sequences".format(len(cpd.dataset)))
    
    for key, item in tqdm(cpd.dataset.items(), desc="repetition over independent sequences"):

        train_dataset = item["train_dataset"]
        valid_dataset = item["valid_dataset"]
        test_dataset = item["test_dataset"]

        train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.training.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

        dim_x = train_dataset.strided_x.shape[-1]
        if config.data.strided_features == "xry":
            dim_feature = dim_x + 2 # add 1 for residual and y as input
        elif config.data.strided_features == "xr":
            dim_feature = dim_x + 1 # add 1 for residual as input
        elif config.data.strided_features == "r":
            dim_feature = 1 # only residual is used for feature
        else:
            raise ValueError("wrong strided features specified")

        # model init
        if config.model.use_current_feature:
            # utilizing up to x_{t-1} and x_t to predict r_t
            qr_transformer = QuantileRegressionTransformer(dim_feature, 
                                                    config.model.dim_model, 
                                                    config.model.num_heads,
                                                    config.model.dim_model*4, 
                                                    config.model.num_layers, 
                                                    config.model.target_quantiles,
                                                    config.model.prediction_step, 
                                                    config.model.dropout,
                                                    current_feature_dim=dim_x,
                                                    head_type=config.model.get("head_type", "nondecreasing"))
        else:
            # utilizing up to x_{t-1} to predict r_t
            qr_transformer = QuantileRegressionTransformer(dim_feature, 
                                                    config.model.dim_model, 
                                                    config.model.num_heads,
                                                    config.model.dim_model*4, 
                                                    config.model.num_layers, 
                                                    config.model.target_quantiles,
                                                    config.model.prediction_step, 
                                                    config.model.dropout,
                                                    current_feature_dim=0,
                                                    head_type=config.model.get("head_type", "nondecreasing"))

        qr_transformer.to(device)
        optimizer = torch.optim.AdamW(qr_transformer.parameters(), 
                                      lr=config.training.learning_rate) # TODO: params for adamW?

        train_loss = []
        valid_loss = []
        best_loss = np.inf
        
        for i in tqdm(range(config.training.epochs), desc="training epochs"):

            loss_sum = 0.
            qr_transformer.train()
            for strided_x, strided_residual, strided_y, target_x, target_residual, _, _ in tqdm(train_dataloader):
                # strided_x : (batch_size, window, dim)
                # strided_residual : (batch_size, window)
                # strided_y : (batch_size, window)
                # target_residual : (batch_size, 1)
                # target_x : (batch, 1, current_feature_dim)

                optimizer.zero_grad()
                strided_feature = generate_strided_feature(strided_x, 
                                                           strided_residual, 
                                                           strided_y,
                                                           config.data.strided_features)
                strided_feature = strided_feature.to(device)
                target_residual = target_residual.to(device)
                target_x = target_x.to(device)

                if config.model.use_current_feature:
                    loss = compute_loss_quantile_regression_transformer(qr_transformer, 
                                                                        strided_feature, 
                                                                        target_residual, 
                                                                        config.model.target_quantiles,
                                                                        target_x)
                else:
                    loss = compute_loss_quantile_regression_transformer(qr_transformer, 
                                                                        strided_feature, 
                                                                        target_residual, 
                                                                        config.model.target_quantiles)    
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            train_loss.append(loss_sum/len(train_dataloader))
            print("training loss at epoch {}: {}".format(i+1, loss_sum/len(train_dataloader)))

            loss_sum = 0.
            qr_transformer.eval()
            for strided_x, strided_residual, strided_y, target_x, target_residual, _, _ in tqdm(valid_dataloader):

                strided_feature = generate_strided_feature(strided_x, 
                                                           strided_residual, 
                                                           strided_y,
                                                           config.data.strided_features)
                strided_feature = strided_feature.to(device)
                target_residual = target_residual.to(device)
                target_x = target_x.to(device)
                with torch.no_grad():
                    if config.model.use_current_feature:
                        loss = compute_loss_quantile_regression_transformer(qr_transformer, 
                                                                            strided_feature, 
                                                                            target_residual, 
                                                                            config.model.target_quantiles,
                                                                            target_x)
                    else:
                        loss = compute_loss_quantile_regression_transformer(qr_transformer, 
                                                                            strided_feature, 
                                                                            target_residual, 
                                                                            config.model.target_quantiles)
                loss_sum += loss.item()

            epoch_valid_loss = loss_sum/len(valid_dataloader)
            valid_loss.append(loss_sum/len(valid_dataloader))
            print("validation loss at epoch {}: {}".format(i+1, loss_sum/len(valid_dataloader)))

            if epoch_valid_loss < best_loss:
                best_loss = epoch_valid_loss
                best_epoch = i+1
                best_model = copy.deepcopy(qr_transformer.state_dict())

            if config.training.early_stop:
                if (i+1-best_epoch) >= config.training.early_stop:
                    # if the loss did not decrease for (early_stop) epoch in a row, stop training
                    break

        # TODO: additional training with validation dataset?
                
        # evaluate on the test data
        qr_transformer.load_state_dict(best_model)
        evaluation_results = {
            tuple(confidence_pair): {"coverage": [], 
                                     "interval_width": [],
                                     "winkler_score" : [],
                                     "upper_interval" : [],
                                     "lower_interval" : [],
                                     "upper_residual_quantile" : [],
                                     "lower_residual_quantile" : [],
                                     "target_y" : [],
                                     "target_predictions" : []}
            for confidence_pair in config.model.target_quantiles
        }

        crossing_tracker = QuantileCrossingTracker(
            config, key, len(cpd.data[key]["heldout_y"]), len(test_dataset),
        ) if exclude_crossings else None

        if config.data.normalize:
            residuals_noramlized_mu = cpd.data[key]["train_residuals_mu"]
            residuals_noramlized_std = cpd.data[key]["train_residuals_std"]

        qr_transformer.eval()
        for strided_x, strided_residual, strided_y, target_x, target_residual, target_y, target_predictions in tqdm(test_dataloader):
            
            with torch.no_grad():
                strided_feature = generate_strided_feature(strided_x, 
                                                           strided_residual, 
                                                           strided_y,
                                                           config.data.strided_features)
                strided_feature = strided_feature.to(device)
                target_x = target_x.to(device)

                if config.model.use_current_feature:
                    # (batch_size, num_unique_quantiles)
                    pred_quantile_values = qr_transformer.get_predicted_quantile_values(qr_transformer, 
                                                                                        strided_feature, 
                                                                                        target_x)
                else:
                    # (batch_size, num_unique_quantiles)
                    pred_quantile_values = qr_transformer.get_predicted_quantile_values(qr_transformer, 
                                                                                        strided_feature)
                    
            if config.device != "cpu":
                pred_quantile_values = pred_quantile_values.cpu().detach()

            if crossing_tracker is not None:
                pred_quantile_values, target_residual, target_y, target_predictions = crossing_tracker.filter_batch(
                    pred_quantile_values, target_residual, target_y, target_predictions,
                )

            for confidence_pair in config.model.target_quantiles:
                tuple_confidence_pair = tuple(confidence_pair)
                hi_idx, lo_idx = pair_to_indices[tuple_confidence_pair]
                hi = pred_quantile_values[:, hi_idx] # (batch_size,)
                lo = pred_quantile_values[:, lo_idx] # (batch_size,)
                this_coverage = compute_coverage(hi, lo, target_residual)

                if config.data.normalize:
                    this_interval_width = compute_interval_width(hi, 
                                                                 lo, 
                                                                 normalized_std=residuals_noramlized_std)
                    this_winkler_score = compute_winkler_score(hi,
                                                               lo, 
                                                               target_y, 
                                                               target_predictions, 
                                                               tuple_confidence_pair, 
                                                               normalized_params=(residuals_noramlized_mu, 
                                                                                  residuals_noramlized_std))
                else:
                    this_interval_width = compute_interval_width(hi, 
                                                                 lo, 
                                                                 normalized_std=None)
                    this_winkler_score = compute_winkler_score(hi,
                                                               lo, 
                                                               target_y, 
                                                               target_predictions, 
                                                               tuple_confidence_pair, 
                                                               normalized_params=None)

                upper_interval, lower_interval = construct_interval_endpoints(
                    hi,
                    lo,
                    target_predictions,
                    normalized_params=(residuals_noramlized_mu,
                                       residuals_noramlized_std)
                    if config.data.normalize else None,
                )
                
                evaluation_results[tuple_confidence_pair]["upper_interval"].extend(upper_interval.tolist())
                evaluation_results[tuple_confidence_pair]["lower_interval"].extend(lower_interval.tolist())
                evaluation_results[tuple_confidence_pair]["upper_residual_quantile"].extend(hi.tolist())
                evaluation_results[tuple_confidence_pair]["lower_residual_quantile"].extend(lo.tolist())
                evaluation_results[tuple_confidence_pair]["coverage"].extend(this_coverage)
                evaluation_results[tuple_confidence_pair]["interval_width"].extend(this_interval_width)
                evaluation_results[tuple_confidence_pair]["winkler_score"].extend(this_winkler_score)
                evaluation_results[tuple_confidence_pair]["target_y"].extend(target_y.flatten().tolist())
                evaluation_results[tuple_confidence_pair]["target_predictions"].extend(target_predictions.flatten().tolist())

                if config.data.normalize:
                    evaluation_results[tuple_confidence_pair]["train_residuals_mu"] = residuals_noramlized_mu
                    evaluation_results[tuple_confidence_pair]["train_residuals_std"] = residuals_noramlized_std
        
        for confidence_pair in config.model.target_quantiles:
            tuple_confidence_pair = tuple(confidence_pair)
            target_alpha = max(tuple_confidence_pair) - min(tuple_confidence_pair)
            result = evaluation_results[tuple_confidence_pair]
            has_predictions = bool(result["coverage"])
            avg_coverage = np.mean(result["coverage"]) if has_predictions else None
            avg_delta_coverage = avg_coverage - target_alpha if has_predictions else None
            avg_interval_width = np.mean(result["interval_width"]) if has_predictions else None
            avg_winkler_score = np.mean(result["winkler_score"]) if has_predictions else None
            print("avg coverage: {}".format(avg_coverage))
            print("avg delta coverage: {}".format(avg_delta_coverage))
            print("avg interval width: {}".format(avg_interval_width))
            print("avg winkler score: {}".format(avg_winkler_score))
            evaluation_results[tuple_confidence_pair]["avg_coverage"] = avg_coverage
            evaluation_results[tuple_confidence_pair]["avg_delta_coverage"] = avg_delta_coverage
            evaluation_results[tuple_confidence_pair]["avg_interval_width"] = avg_interval_width
            evaluation_results[tuple_confidence_pair]["avg_winkler_score"] = avg_winkler_score

        log[key] = {"train_loss" : train_loss,
                    "valid_loss" : valid_loss,
                    "evaluation_results" : evaluation_results}
        if crossing_tracker is not None:
            log[key]["metadata"] = crossing_tracker.finalize(evaluation_results)

        torch.save(best_model, os.path.join(config.saving_dir, key + '_model.pt'))
        save_data(os.path.join(config.saving_dir, "log.pkl"), log)

    if exclude_crossings:
        summary_results = summarize_qr_results(log, config.model.target_quantiles)
        write_excluded_points(log, config.saving_dir)
    else:
        summary_results = summarize_evaluation_results(log, config.model.target_quantiles)

    for tuple_confidence_pair, summary in summary_results.items():
        print("Summary for confidence pair {}".format(tuple_confidence_pair))
        if exclude_crossings:
            print("Test points: evaluated={}, excluded={}, total={}".format(
                summary["evaluated_points"], summary["excluded_points_count"], summary["total_points"],
            ))
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
        plotted = {key: item for key, item in log.items()
                   if any(result["coverage"] for result in item["evaluation_results"].values())}
        plot_cp_prediction_intervals(plotted,
                                        config.model.target_quantiles, 
                                        config.plotting.plotting_seq_len,
                                        os.path.join(config.saving_dir, "plots"))
                    

def run_rnn_quantile_regression_cp(config_path):

    config = load_experiment_config(config_path)
    os.makedirs(config.saving_dir, exist_ok=True)
    _, pair_to_indices = get_interval_quantile_indices(config.model.target_quantiles)
    exclude_crossings = config.model.get("head_type", "nondecreasing") == "independent"

    # load data
    data = load_data(config.data.data_path) # load predictor results here
    base_predictor, data_type = read_setup(config.data.data_path)
    cpd = ConformalPredictionData(data)
    cpd.prepare_quantile_regression_datasets(config.model.window_size, 
                                             config.model.prediction_step, 
                                             config.data.train_ratio, 
                                             config.data.valid_ratio, 
                                             normalize=config.data.normalize)
    device = config.device
    log = dict()

    print("Experiment setup")
    print("Method: Quantile Regression - RNN")
    print("Base predictor: {}".format(base_predictor))
    print("Data: {}".format(data_type))
    print("{} independent sequences".format(len(cpd.dataset)))
    
    for key, item in tqdm(cpd.dataset.items(), desc="repetition over independent sequences"):

        train_dataset = item["train_dataset"]
        valid_dataset = item["valid_dataset"]
        test_dataset = item["test_dataset"]

        train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.training.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

        dim_x = train_dataset.strided_x.shape[-1]
        if config.data.strided_features == "xry":
            dim_feature = dim_x + 2 # add 1 for residual and y as input
        elif config.data.strided_features == "xr":
            dim_feature = dim_x + 1 # add 1 for residual as input
        elif config.data.strided_features == "r":
            dim_feature = 1 # only residual is used for feature
        else:
            raise ValueError("wrong strided features specified")

        # model init
        if config.model.use_current_feature:
            # utilizing up to x_{t-1} and x_t to predict r_t
            qr_rnn = QuantileRegressionRNN(config.model.rnn_type,
                                           dim_feature,
                                           config.model.dim_model,
                                           config.model.num_layers,
                                           config.model.target_quantiles,
                                           config.model.prediction_step,
                                           config.model.dropout,
                                           current_feature_dim=dim_x,
                                           head_type=config.model.get("head_type", "nondecreasing"))
        else:
            # utilizing up to x_{t-1} to predict r_t
            qr_rnn = QuantileRegressionRNN(config.model.rnn_type,
                                           dim_feature,
                                           config.model.dim_model,
                                           config.model.num_layers,
                                           config.model.target_quantiles,
                                           config.model.prediction_step,
                                           config.model.dropout,
                                           current_feature_dim=0,
                                           head_type=config.model.get("head_type", "nondecreasing"))

        qr_rnn.to(device)
        optimizer = torch.optim.AdamW(qr_rnn.parameters(), 
                                      lr=config.training.learning_rate) 

        train_loss = []
        valid_loss = []
        best_loss = np.inf
        
        for i in tqdm(range(config.training.epochs), desc="training epochs"):

            loss_sum = 0.
            qr_rnn.train()
            for strided_x, strided_residual, strided_y, target_x, target_residual, _, _ in tqdm(train_dataloader):
                # strided_x : (batch_size, window, dim)
                # strided_residual : (batch_size, window)
                # strided_y : (batch_size, window)
                # target_residual : (batch_size, 1)
                # target_x : (batch, 1, current_feature_dim)

                optimizer.zero_grad()
                strided_feature = generate_strided_feature(strided_x, 
                                                            strided_residual, 
                                                            strided_y,
                                                            config.data.strided_features)
                strided_feature = strided_feature.to(device)
                target_residual = target_residual.to(device)
                target_x = target_x.to(device)

                if config.model.use_current_feature:
                    loss = compute_loss_quantile_regression_rnn(qr_rnn, 
                                                                strided_feature, 
                                                                target_residual, 
                                                                config.model.target_quantiles,
                                                                target_x)
                else:
                    loss = compute_loss_quantile_regression_rnn(qr_rnn, 
                                                                strided_feature, 
                                                                target_residual, 
                                                                config.model.target_quantiles)    
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            train_loss.append(loss_sum/len(train_dataloader))
            print("training loss at epoch {}: {}".format(i+1, loss_sum/len(train_dataloader)))

            loss_sum = 0.
            qr_rnn.eval()
            for strided_x, strided_residual, strided_y, target_x, target_residual, _, _ in tqdm(valid_dataloader):

                strided_feature = generate_strided_feature(strided_x, 
                                                            strided_residual, 
                                                            strided_y,
                                                            config.data.strided_features)
                strided_feature = strided_feature.to(device)
                target_residual = target_residual.to(device)
                target_x = target_x.to(device)
                with torch.no_grad():
                    if config.model.use_current_feature:
                        loss = compute_loss_quantile_regression_rnn(qr_rnn, 
                                                                    strided_feature, 
                                                                    target_residual, 
                                                                    config.model.target_quantiles,
                                                                    target_x)
                    else:
                        loss = compute_loss_quantile_regression_rnn(qr_rnn, 
                                                                    strided_feature, 
                                                                    target_residual, 
                                                                    config.model.target_quantiles)
                loss_sum += loss.item()

            epoch_valid_loss = loss_sum/len(valid_dataloader)
            valid_loss.append(loss_sum/len(valid_dataloader))
            print("validation loss at epoch {}: {}".format(i+1, loss_sum/len(valid_dataloader)))

            if epoch_valid_loss < best_loss:
                best_loss = epoch_valid_loss
                best_epoch = i+1
                best_model = copy.deepcopy(qr_rnn.state_dict())

            if config.training.early_stop:
                if (i+1-best_epoch) >= config.training.early_stop:
                    # if the loss did not decrease for (early_stop) epoch in a row, stop training
                    break

        # TODO: additional training with validation dataset?
                
        # evaluate on the test data
        qr_rnn.load_state_dict(best_model)
        evaluation_results = {
            tuple(confidence_pair): {"coverage": [], 
                                     "interval_width": [],
                                     "winkler_score" : [],
                                     "upper_interval" : [],
                                     "lower_interval" : [],
                                     "upper_residual_quantile" : [],
                                     "lower_residual_quantile" : [],
                                     "target_y" : [],
                                     "target_predictions" : []}
            for confidence_pair in config.model.target_quantiles
        }

        crossing_tracker = QuantileCrossingTracker(
            config, key, len(cpd.data[key]["heldout_y"]), len(test_dataset),
        ) if exclude_crossings else None

        if config.data.normalize:
            residuals_noramlized_mu = cpd.data[key]["train_residuals_mu"]
            residuals_noramlized_std = cpd.data[key]["train_residuals_std"]

        qr_rnn.eval()
        for strided_x, strided_residual, strided_y, target_x, target_residual, target_y, target_predictions in tqdm(test_dataloader):
            
            with torch.no_grad():
                strided_feature = generate_strided_feature(strided_x, 
                                                            strided_residual, 
                                                            strided_y,
                                                            config.data.strided_features)
                strided_feature = strided_feature.to(device)
                target_x = target_x.to(device)

                if config.model.use_current_feature:
                    # (batch_size, num_unique_quantiles)
                    pred_quantile_values = qr_rnn.get_predicted_quantile_values(qr_rnn, strided_feature, target_x)
                else:
                    # (batch_size, num_unique_quantiles)
                    pred_quantile_values = qr_rnn.get_predicted_quantile_values(qr_rnn, strided_feature)

            if config.device != "cpu":
                pred_quantile_values = pred_quantile_values.cpu().detach()

            if crossing_tracker is not None:
                pred_quantile_values, target_residual, target_y, target_predictions = crossing_tracker.filter_batch(
                    pred_quantile_values, target_residual, target_y, target_predictions,
                )

            for confidence_pair in config.model.target_quantiles:
                tuple_confidence_pair = tuple(confidence_pair)
                hi_idx, lo_idx = pair_to_indices[tuple_confidence_pair]
                hi = pred_quantile_values[:, hi_idx] # (batch_size,)
                lo = pred_quantile_values[:, lo_idx] # (batch_size,)
                this_coverage = compute_coverage(hi, lo, target_residual)

                if config.data.normalize:
                    this_interval_width = compute_interval_width(hi, 
                                                                 lo, 
                                                                 normalized_std=residuals_noramlized_std)
                    this_winkler_score = compute_winkler_score(hi,
                                                               lo, 
                                                               target_y, 
                                                               target_predictions, 
                                                               tuple_confidence_pair, 
                                                               normalized_params=(residuals_noramlized_mu, 
                                                                                  residuals_noramlized_std))
                else:
                    this_interval_width = compute_interval_width(hi, 
                                                                 lo, 
                                                                 normalized_std=None)
                    this_winkler_score = compute_winkler_score(hi,
                                                               lo, 
                                                               target_y, 
                                                               target_predictions, 
                                                               tuple_confidence_pair, 
                                                               normalized_params=None)

                upper_interval, lower_interval = construct_interval_endpoints(
                    hi,
                    lo,
                    target_predictions,
                    normalized_params=(residuals_noramlized_mu,
                                       residuals_noramlized_std)
                    if config.data.normalize else None,
                )
                
                evaluation_results[tuple_confidence_pair]["upper_interval"].extend(upper_interval.tolist())
                evaluation_results[tuple_confidence_pair]["lower_interval"].extend(lower_interval.tolist())
                evaluation_results[tuple_confidence_pair]["upper_residual_quantile"].extend(hi.tolist())
                evaluation_results[tuple_confidence_pair]["lower_residual_quantile"].extend(lo.tolist())
                evaluation_results[tuple_confidence_pair]["coverage"].extend(this_coverage)
                evaluation_results[tuple_confidence_pair]["interval_width"].extend(this_interval_width)
                evaluation_results[tuple_confidence_pair]["winkler_score"].extend(this_winkler_score)
                evaluation_results[tuple_confidence_pair]["target_y"].extend(target_y.flatten().tolist())
                evaluation_results[tuple_confidence_pair]["target_predictions"].extend(target_predictions.flatten().tolist())

                if config.data.normalize:
                    evaluation_results[tuple_confidence_pair]["train_residuals_mu"] = residuals_noramlized_mu
                    evaluation_results[tuple_confidence_pair]["train_residuals_std"] = residuals_noramlized_std
        
        for confidence_pair in config.model.target_quantiles:
            tuple_confidence_pair = tuple(confidence_pair)
            target_alpha = max(tuple_confidence_pair) - min(tuple_confidence_pair)
            result = evaluation_results[tuple_confidence_pair]
            has_predictions = bool(result["coverage"])
            avg_coverage = np.mean(result["coverage"]) if has_predictions else None
            avg_delta_coverage = avg_coverage - target_alpha if has_predictions else None
            avg_interval_width = np.mean(result["interval_width"]) if has_predictions else None
            avg_winkler_score = np.mean(result["winkler_score"]) if has_predictions else None
            print("avg coverage: {}".format(avg_coverage))
            print("avg delta coverage: {}".format(avg_delta_coverage))
            print("avg interval width: {}".format(avg_interval_width))
            print("avg winkler score: {}".format(avg_winkler_score))
            evaluation_results[tuple_confidence_pair]["avg_coverage"] = avg_coverage
            evaluation_results[tuple_confidence_pair]["avg_delta_coverage"] = avg_delta_coverage
            evaluation_results[tuple_confidence_pair]["avg_interval_width"] = avg_interval_width
            evaluation_results[tuple_confidence_pair]["avg_winkler_score"] = avg_winkler_score

        log[key] = {"train_loss" : train_loss,
                    "valid_loss" : valid_loss,
                    "evaluation_results" : evaluation_results}
        if crossing_tracker is not None:
            log[key]["metadata"] = crossing_tracker.finalize(evaluation_results)

        torch.save(best_model, os.path.join(config.saving_dir, key + '_model.pt'))
        save_data(os.path.join(config.saving_dir, "log.pkl"), log)

    if exclude_crossings:
        summary_results = summarize_qr_results(log, config.model.target_quantiles)
        write_excluded_points(log, config.saving_dir)
    else:
        summary_results = summarize_evaluation_results(log, config.model.target_quantiles)

    for tuple_confidence_pair, summary in summary_results.items():
        print("Summary for confidence pair {}".format(tuple_confidence_pair))
        if exclude_crossings:
            print("Test points: evaluated={}, excluded={}, total={}".format(
                summary["evaluated_points"], summary["excluded_points_count"], summary["total_points"],
            ))
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
        plotted = {key: item for key, item in log.items()
                   if any(result["coverage"] for result in item["evaluation_results"].values())}
        plot_cp_prediction_intervals(plotted,
                                        config.model.target_quantiles, 
                                        config.plotting.plotting_seq_len,
                                        os.path.join(config.saving_dir, "plots"))



