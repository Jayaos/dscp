from omegaconf import OmegaConf
import torch
import os
import copy
import numpy as np
from tqdm import tqdm
from dscp.models.transformer_predictor import TransformerPredictor
from dscp.models.rnn_predictor import RNNPredictor
from dscp.models.local_cp import LocalConformalPrediction
from dscp.loss import compute_loss_transformer_predictor, compute_loss_rnn_predictor
from dscp.data import ConformalPredictionData
from utils.utils import load_data, save_data, read_setup, flatten, generate_strided_feature
from utils.reporting import compute_coverage, compute_interval_width, compute_winkler_score, summarize_evaluation_results
from utils.plotting import plot_cp_prediction_intervals
from torch.utils.data import DataLoader, ConcatDataset


def run_transformer_local_cp(config_path):

    config = OmegaConf.load(config_path)
    os.makedirs(config.saving_dir, exist_ok=True)

    # load data
    data = load_data(config.data.data_path) # load predictor results here
    base_predictor, data_type = read_setup(config.data.data_path)
    cpd = ConformalPredictionData(data)
    cpd.prepare_quantile_regression_datasets(config.model.window_size, 
                              config.model.prediction_step, 
                              config.data.train_ratio, 
                              config.data.valid_ratio, 
                              config.data.normalize)
    device = config.device
    target_quantiles = flatten(config.model.target_quantiles)
    log = dict()

    print("Experiment setup")
    print("Method: Local Conformal Prediction - Transformer")
    print("Base predictor: {}".format(base_predictor))
    print("Data: {}".format(data_type))
    print("{} independent sequences".format(len(cpd.dataset)))
    
    for key, item in tqdm(cpd.dataset.items(), desc="repetition over independent sequences"):

        train_dataset = item["train_dataset"]
        valid_dataset = item["valid_dataset"]
        test_dataset = item["test_dataset"]

        train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.training.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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
        transformer_predictor = TransformerPredictor(dim_feature, 
                                                     config.model.dim_model, 
                                                     config.model.num_head,
                                                     config.model.dim_model * 4, 
                                                     config.model.num_layer,
                                                     config.model.prediction_step, 
                                                     config.model.dropout)
        transformer_predictor.to(device)
        optimizer = torch.optim.AdamW(transformer_predictor.parameters(), 
                                      lr=config.training.learning_rate) # TODO: params for adamW?

        train_loss = []
        valid_loss = []
        best_loss = np.inf
        for i in tqdm(range(config.training.epochs), desc="training epochs"):

            loss_sum = 0.
            transformer_predictor.train()
            for strided_x, strided_residual, strided_y, target_x, target_residual, _, _ in tqdm(train_dataloader):

                optimizer.zero_grad()
                strided_feature = generate_strided_feature(strided_x, 
                                                           strided_residual, 
                                                           strided_y,
                                                           config.data.strided_features)
                strided_feature = strided_feature.to(device)
                target_residual = target_residual.to(device)
                loss = compute_loss_transformer_predictor(transformer_predictor, 
                                                          strided_feature, 
                                                          target_residual)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            epoch_train_loss = loss_sum/len(train_dataloader)
            train_loss.append(epoch_train_loss)
            print("training loss at epoch {}: {}".format(i+1, epoch_train_loss))

            loss_sum = 0.
            transformer_predictor.eval()
            for strided_x, strided_residual, strided_y, target_x, target_residual, _, _ in tqdm(valid_dataloader):

                strided_feature = generate_strided_feature(strided_x, 
                                                           strided_residual, 
                                                           strided_y,
                                                           config.data.strided_features)

                strided_feature = strided_feature.to(device)
                target_residual = target_residual.to(device)
                with torch.no_grad():
                    loss = compute_loss_transformer_predictor(transformer_predictor, 
                                                              strided_feature, 
                                                              target_residual)
                loss_sum += loss.item()

            epoch_valid_loss = loss_sum/len(valid_dataloader)
            valid_loss.append(epoch_valid_loss)
            print("validation loss at epoch {}: {}".format(i+1, epoch_valid_loss))

            if epoch_valid_loss < best_loss:
                best_loss = epoch_valid_loss
                best_epoch = i+1
                best_model = copy.deepcopy(transformer_predictor.state_dict())

            if config.training.early_stop:
                if (i+1-best_epoch) >= config.training.early_stop:
                    # if the loss did not decrease for (early_stop) epoch in a row, stop training
                    break
        
        # training transformer predictor has been completed
        # evaluate on the test data
        transformer_predictor.load_state_dict(best_model)

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

        # NOTE: test_dataloader batch size must be 1 in the current setup
        tv_dataset = ConcatDataset([train_dataset, valid_dataset])
        tv_dataloader = DataLoader(tv_dataset, batch_size=config.training.batch_size, shuffle=False)

        with torch.no_grad():
            tv_repr, tv_residual = transformer_predictor.encode_dataloader(transformer_predictor,
                                                                           tv_dataloader,
                                                                           config.data.strided_features,
                                                                           device)

        if len(tv_repr) > config.model.calibration_size:
            tv_repr = tv_repr[-config.model.calibration_size:]
            tv_residual = tv_residual[-config.model.calibration_size:]

        past_test_repr = []
        past_test_residual = []

        test_data_count = 0
        transformer_predictor.eval()
        for strided_x, strided_residual, strided_y, \
            target_x, target_residual, target_y, target_predictions in tqdm(test_dataloader):

            # construct query representation
            with torch.no_grad():
                strided_feature = generate_strided_feature(strided_x, 
                                                           strided_residual, 
                                                           strided_y,
                                                           config.data.strided_features)
                strided_feature = strided_feature.to(device)
                o = transformer_predictor.encode(transformer_predictor, strided_feature)
                query_repr = o[:, -1, :] # representation of the last timestep, (query_size, dim_model)

            tv_idx = max((config.model.calibration_size - test_data_count), 0)
            
            calib_repr_parts = []
            calib_residual_parts = []

            if tv_idx > 0 and len(tv_repr) > 0:
                calib_repr_parts.append(tv_repr[-tv_idx:])
                calib_residual_parts.append(tv_residual[-tv_idx:])

            if len(past_test_repr) > 0:
                calib_repr_parts.append(torch.vstack(past_test_repr[-config.model.calibration_size:]))
                calib_residual_parts.append(torch.vstack(past_test_residual[-config.model.calibration_size:]))

            calib_repr = torch.vstack(calib_repr_parts)
            calib_residual = torch.vstack(calib_residual_parts)
                 
            test_data_count += 1

            local_cp = LocalConformalPrediction(calib_repr,
                                                calib_residual, 
                                                config.model.similarity_fn, 
                                                config.model.temperature,
                                                device)
            
            # (len(target_quantiles), query_size)
            pred_quantile_values = local_cp.approximate_quantile(query_repr, 
                                                                 target_quantiles,
                                                                 config.model.sampling_num)
            
            if config.device != "cpu":
                pred_quantile_values = pred_quantile_values.cpu().detach()

            for j, confidence_pair in enumerate(config.model.target_quantiles):
                tuple_confidence_pair = tuple(confidence_pair)
                hi = pred_quantile_values[2*j+0, :] 
                lo = pred_quantile_values[2*j+1, :] 
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
                    
                evaluation_results[tuple_confidence_pair]["upper_interval"].extend(hi.tolist())
                evaluation_results[tuple_confidence_pair]["lower_interval"].extend(lo.tolist())
                evaluation_results[tuple_confidence_pair]["coverage"].extend(this_coverage)
                evaluation_results[tuple_confidence_pair]["interval_width"].extend(this_interval_width)
                evaluation_results[tuple_confidence_pair]["winkler_score"].extend(this_winkler_score)
                evaluation_results[tuple_confidence_pair]["target_y"].extend(target_y.flatten().tolist())
                evaluation_results[tuple_confidence_pair]["target_predictions"].extend(target_predictions.flatten().tolist())

                if config.data.normalize:
                    evaluation_results[tuple_confidence_pair]["train_residuals_mu"] = residuals_noramlized_mu
                    evaluation_results[tuple_confidence_pair]["train_residuals_std"] = residuals_noramlized_std

            past_test_repr.append(query_repr.detach().cpu())
            past_test_residual.append(strided_residual[:, -1].detach().cpu().reshape(-1, 1))

            if len(past_test_repr) > config.model.calibration_size:
                past_test_repr = past_test_repr[-config.model.calibration_size:]
                past_test_residual = past_test_residual[-config.model.calibration_size:]
        
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

        log[key] = {"train_loss" : train_loss,
                    "valid_loss" : valid_loss,
                    "evaluation_results" : evaluation_results}
        
        torch.save(best_model, os.path.join(config.saving_dir, key + '_model.pt'))
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


def run_rnn_local_cp(config_path):

    config = OmegaConf.load(config_path)
    os.makedirs(config.saving_dir, exist_ok=True)

    # load data
    data = load_data(config.data.data_path) # load predictor results here
    base_predictor, data_type = read_setup(config.data.data_path)
    cpd = ConformalPredictionData(data)
    cpd.prepare_quantile_regression_datasets(config.model.window_size,
                              config.model.prediction_step,
                              config.data.train_ratio,
                              config.data.valid_ratio,
                              config.data.normalize)
    device = config.device
    target_quantiles = flatten(config.model.target_quantiles)
    log = dict()

    print("Experiment setup")
    print("Method: Local Conformal Prediction - RNN")
    print("Base predictor: {}".format(base_predictor))
    print("Data: {}".format(data_type))
    print("{} independent sequences".format(len(cpd.dataset)))

    for key, item in tqdm(cpd.dataset.items(), desc="repetition over independent sequences"):

        train_dataset = item["train_dataset"]
        valid_dataset = item["valid_dataset"]
        test_dataset = item["test_dataset"]

        train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.training.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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
        rnn_predictor = RNNPredictor(config.model.rnn_type,
                                     dim_feature,
                                     config.model.dim_model,
                                     config.model.num_layer,
                                     config.model.prediction_step,
                                     config.model.dropout)
        rnn_predictor.to(device)
        optimizer = torch.optim.AdamW(rnn_predictor.parameters(),
                                      lr=config.training.learning_rate) # TODO: params for adamW?

        train_loss = []
        valid_loss = []
        best_loss = np.inf
        for i in tqdm(range(config.training.epochs), desc="training epochs"):

            loss_sum = 0.
            rnn_predictor.train()
            for strided_x, strided_residual, strided_y, target_x, target_residual, _, _ in tqdm(train_dataloader):

                optimizer.zero_grad()
                strided_feature = generate_strided_feature(strided_x,
                                                           strided_residual,
                                                           strided_y,
                                                           config.data.strided_features)
                strided_feature = strided_feature.to(device)
                target_residual = target_residual.to(device)
                loss = compute_loss_rnn_predictor(rnn_predictor,
                                                  strided_feature,
                                                  target_residual)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            epoch_train_loss = loss_sum/len(train_dataloader)
            train_loss.append(epoch_train_loss)
            print("training loss at epoch {}: {}".format(i+1, epoch_train_loss))

            loss_sum = 0.
            rnn_predictor.eval()
            for strided_x, strided_residual, strided_y, target_x, target_residual, _, _ in tqdm(valid_dataloader):

                strided_feature = generate_strided_feature(strided_x,
                                                           strided_residual,
                                                           strided_y,
                                                           config.data.strided_features)

                strided_feature = strided_feature.to(device)
                target_residual = target_residual.to(device)
                with torch.no_grad():
                    loss = compute_loss_rnn_predictor(rnn_predictor,
                                                      strided_feature,
                                                      target_residual)
                loss_sum += loss.item()

            epoch_valid_loss = loss_sum/len(valid_dataloader)
            valid_loss.append(epoch_valid_loss)
            print("validation loss at epoch {}: {}".format(i+1, epoch_valid_loss))

            if epoch_valid_loss < best_loss:
                best_loss = epoch_valid_loss
                best_epoch = i+1
                best_model = copy.deepcopy(rnn_predictor.state_dict())

            if config.training.early_stop:
                if (i+1-best_epoch) >= config.training.early_stop:
                    # if the loss did not decrease for (early_stop) epoch in a row, stop training
                    break

        # training RNN predictor has been completed
        # evaluate on the test data
        rnn_predictor.load_state_dict(best_model)

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

        # NOTE: test_dataloader batch size must be 1 in the current setup
        tv_dataset = ConcatDataset([train_dataset, valid_dataset])
        tv_dataloader = DataLoader(tv_dataset, batch_size=config.training.batch_size, shuffle=False)

        with torch.no_grad():
            tv_repr, tv_residual = rnn_predictor.encode_dataloader(rnn_predictor,
                                                                   tv_dataloader,
                                                                   config.data.strided_features,
                                                                   device)

        if len(tv_repr) > config.model.calibration_size:
            tv_repr = tv_repr[-config.model.calibration_size:]
            tv_residual = tv_residual[-config.model.calibration_size:]

        past_test_repr = []
        past_test_residual = []

        test_data_count = 0
        rnn_predictor.eval()
        for strided_x, strided_residual, strided_y, \
            target_x, target_residual, target_y, target_predictions in tqdm(test_dataloader):

            # construct query representation
            with torch.no_grad():
                strided_feature = generate_strided_feature(strided_x,
                                                           strided_residual,
                                                           strided_y,
                                                           config.data.strided_features)
                strided_feature = strided_feature.to(device)
                o = rnn_predictor.encode(rnn_predictor, strided_feature)
                query_repr = o[:, -1, :] # representation of the last timestep, (query_size, dim_model)

            tv_idx = max((config.model.calibration_size - test_data_count), 0)

            calib_repr_parts = []
            calib_residual_parts = []

            if tv_idx > 0 and len(tv_repr) > 0:
                calib_repr_parts.append(tv_repr[-tv_idx:])
                calib_residual_parts.append(tv_residual[-tv_idx:])

            if len(past_test_repr) > 0:
                calib_repr_parts.append(torch.vstack(past_test_repr[-config.model.calibration_size:]))
                calib_residual_parts.append(torch.vstack(past_test_residual[-config.model.calibration_size:]))

            calib_repr = torch.vstack(calib_repr_parts)
            calib_residual = torch.vstack(calib_residual_parts)

            test_data_count += 1

            local_cp = LocalConformalPrediction(calib_repr,
                                                calib_residual,
                                                config.model.similarity_fn,
                                                config.model.temperature,
                                                device)

            # (len(target_quantiles), query_size)
            pred_quantile_values = local_cp.approximate_quantile(query_repr,
                                                                 target_quantiles,
                                                                 config.model.sampling_num)

            if config.device != "cpu":
                pred_quantile_values = pred_quantile_values.cpu().detach()

            for j, confidence_pair in enumerate(config.model.target_quantiles):
                tuple_confidence_pair = tuple(confidence_pair)
                hi = pred_quantile_values[2*j+0, :]
                lo = pred_quantile_values[2*j+1, :]
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

                evaluation_results[tuple_confidence_pair]["upper_interval"].extend(hi.tolist())
                evaluation_results[tuple_confidence_pair]["lower_interval"].extend(lo.tolist())
                evaluation_results[tuple_confidence_pair]["coverage"].extend(this_coverage)
                evaluation_results[tuple_confidence_pair]["interval_width"].extend(this_interval_width)
                evaluation_results[tuple_confidence_pair]["winkler_score"].extend(this_winkler_score)
                evaluation_results[tuple_confidence_pair]["target_y"].extend(target_y.flatten().tolist())
                evaluation_results[tuple_confidence_pair]["target_predictions"].extend(target_predictions.flatten().tolist())

                if config.data.normalize:
                    evaluation_results[tuple_confidence_pair]["train_residuals_mu"] = residuals_noramlized_mu
                    evaluation_results[tuple_confidence_pair]["train_residuals_std"] = residuals_noramlized_std

            past_test_repr.append(query_repr.detach().cpu())
            past_test_residual.append(strided_residual[:, -1].detach().cpu().reshape(-1, 1))

            if len(past_test_repr) > config.model.calibration_size:
                past_test_repr = past_test_repr[-config.model.calibration_size:]
                past_test_residual = past_test_residual[-config.model.calibration_size:]

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

        log[key] = {"train_loss" : train_loss,
                    "valid_loss" : valid_loss,
                    "evaluation_results" : evaluation_results}

        torch.save(best_model, os.path.join(config.saving_dir, key + '_model.pt'))
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

