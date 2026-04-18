from omegaconf import OmegaConf
import torch
import os
import copy
import numpy as np
from tqdm import tqdm
from baselines.hopcpt.model import HopfieldNet
from baselines.hopcpt.loss import compute_hopfield_net_loss
from dscp.data import ConformalPredictionData, initialize_valid_dataloader, initialize_test_dataloader
from utils.utils import load_data, save_data, read_setup
from utils.utils import generate_feature_hopcpt_training, generate_feature_hopcpt_test, estimate_hopcpt_residual_interval
from utils.reporting import compute_coverage, compute_interval_width, compute_winkler_score, summarize_evaluation_results
from utils.plotting import plot_cp_prediction_intervals


def run_hopcpt(config_path):

    config = OmegaConf.load(config_path)
    os.makedirs(config.saving_dir, exist_ok=True)

    # load data
    data = load_data(config.data.data_path) # load predictor results here
    base_predictor, data_type = read_setup(config.data.data_path)
    cpd = ConformalPredictionData(data)

    cpd.prepare_hopcpt_datasets(config.model.prediction_step,
                                config.model.y_lags,
                                config.data.train_ratio,
                                config.data.valid_ratio,
                                config.data.normalize,
                                config.model.use_absolute_residual)

    device = config.device
    target_quantiles = config.model.target_quantiles
    selection_confidence_pair = tuple(target_quantiles[0]) # this is used for validation
    selection_target_coverage = max(selection_confidence_pair) - min(selection_confidence_pair)
    log = dict()

    print("Experiment setup")
    print("Method: HopCPT")
    print("Base predictor: {}".format(base_predictor))
    print("Data: {}".format(data_type))
    print("{} independent sequences".format(len(cpd.data)))
    
    for key, data in tqdm(cpd.data.items(), desc="repetition over independent sequences"):

        train_size = data["train_size"]
        valid_size = data["valid_size"]
        test_size = data["test_size"]
        valid_dataloader_size = data["valid_dataloader_size"]
        test_dataloader_size = data["test_dataloader_size"]

        dim_feature = data["heldout_context"].shape[-1]
        dim_context_encoding = config.model.dim_context_encoding
        if dim_context_encoding is None or dim_context_encoding == "auto":
            dim_context_encoding = dim_feature
        dim_hopfield_hidden = config.model.dim_hopfield_hidden
        if dim_hopfield_hidden is None or dim_hopfield_hidden == "auto":
            dim_hopfield_hidden = dim_context_encoding

        # model initialization
        hopfield_net = HopfieldNet(dim_feature, 
                                   dim_context_encoding, 
                                   dim_hopfield_hidden, 
                                   config.model.beta,
                                   config.model.use_temporal_encoding)
        hopfield_net.to(device)
        optimizer = torch.optim.AdamW(hopfield_net.parameters(), 
                                      lr=config.training.learning_rate) # TODO: params for adamW?

        train_loss = []
        valid_delta_coverages = []
        valid_interval_widths = []
        best_interval_width = np.inf
        best_delta_coverage = -np.inf
        best_epoch = 0
        best_model = copy.deepcopy(hopfield_net.state_dict())

        for i in tqdm(range(config.training.epochs), desc="training epochs"):

            hopfield_net.train()
            optimizer.zero_grad()
            memory_feature = generate_feature_hopcpt_training(data["heldout_train_context"])
            memory_feature = memory_feature.to(device) # (1, memory_length, feature_dim)
            memory_residual = torch.from_numpy(data["heldout_train_residual"]).to(torch.float32).to(device)
            loss = compute_hopfield_net_loss(hopfield_net,
                                             memory_feature,
                                             memory_residual)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            print("training loss at epoch {}: {}".format(i+1, loss.item()))

            if (i+1) % config.training.validation_epochs == 0:

                # evaluate on the validation set
                hopfield_net.eval()
                this_coverages = []
                this_interval_widths = []

                valid_context_dataloader, valid_target_y_dataloader, \
                    valid_residual_dataloader, valid_prediction_dataloader = initialize_valid_dataloader(data, 
                                                                                                        train_size, 
                                                                                                        valid_size, 
                                                                                                        config.model.prediction_step, 
                                                                                                        config.model.memory_size,
                                                                                                        config.data.normalize)

                for j in range(valid_dataloader_size):

                    strided_context, target_context = next(valid_context_dataloader)
                    _, target_y = next(valid_target_y_dataloader)
                    strided_residual, target_residual = next(valid_residual_dataloader)
                    _, target_predictions = next(valid_prediction_dataloader)
                    
                    memory_feature, query_feature = generate_feature_hopcpt_test(strided_context,
                                                                                 target_context)
                    memory_feature = memory_feature.to(device) # (batch_size, memeory_length, feature_dim)
                    query_feature = query_feature.to(device) # (batch_size, 1, feature_dim)
                    with torch.no_grad():
                        # (batch_size, 1, 1, memory_length)
                        association_matrix = hopfield_net.obtain_association_matrix(memory_feature, query_feature)
                    lo, hi = estimate_hopcpt_residual_interval(association_matrix,
                                                         strided_residual,
                                                         selection_confidence_pair,
                                                         config.model.sampling_num,
                                                         config.model.use_absolute_residual)
                    this_coverage = compute_coverage(hi, lo, target_residual)
                    this_interval_width = compute_interval_width(hi, lo, normalized_std=None)
                    this_coverages.extend(this_coverage)
                    this_interval_widths.extend(this_interval_width)

                this_delta_coverage = np.mean(this_coverages) - selection_target_coverage
                valid_delta_coverages.append(this_delta_coverage)
                this_avg_interval_width = np.mean(this_interval_widths)
                valid_interval_widths.append(this_avg_interval_width)
                print("delta coverage on validation set at epoch {}: {}".format(i+1, this_delta_coverage))
                print("avg interval width on validation set at epoch {}: {}".format(i+1, this_avg_interval_width))

                valid_coverage = this_delta_coverage >= 0
                best_has_valid_coverage = best_delta_coverage >= 0
                if (valid_coverage and
                        (not best_has_valid_coverage or this_avg_interval_width < best_interval_width)) or \
                   (not valid_coverage and
                        not best_has_valid_coverage and this_delta_coverage > best_delta_coverage):
                    best_delta_coverage = this_delta_coverage
                    best_interval_width = this_avg_interval_width
                    best_epoch = i+1
                    best_model = copy.deepcopy(hopfield_net.state_dict())

        # load best model
        print("best model : epoch {}".format(best_epoch))
        hopfield_net.load_state_dict(best_model)

        test_context_dataloader, test_target_y_dataloader, \
            test_residual_dataloader, test_prediction_dataloader = initialize_test_dataloader(data, 
                                                                                             train_size, 
                                                                                             valid_size, 
                                                                                             config.model.prediction_step, 
                                                                                             config.model.memory_size,
                                                                                             config.data.normalize)

        evaluation_results = {
            tuple(confidence_pair): {"coverage": [],
                                     "interval_width": [],
                                     "winkler_score" : [],
                                     "upper_interval" : [],
                                     "lower_interval" : [],
                                     "target_y" : [],
                                     "target_predictions" : []}
            for confidence_pair in target_quantiles
        }

        hopfield_net.eval()
        for j in range(test_dataloader_size):

            strided_context, target_context = next(test_context_dataloader)
            _, target_y = next(test_target_y_dataloader)
            strided_residual, target_residual = next(test_residual_dataloader)
            _, target_predictions = next(test_prediction_dataloader)

            with torch.no_grad():
                memory_feature, query_feature = generate_feature_hopcpt_test(strided_context,
                                                                             target_context)
                memory_feature = memory_feature.to(device) # (batch_size, memeory_length, feature_dim)
                query_feature = query_feature.to(device) # (batch_size, 1, feature_dim)
                # (batch_size, 1, 1, memory_length)
                association_matrix = hopfield_net.obtain_association_matrix(memory_feature, query_feature)

            for confidence_pair in target_quantiles:
                tuple_confidence_pair = tuple(confidence_pair)
                lo, hi = estimate_hopcpt_residual_interval(association_matrix,
                                                           strided_residual,
                                                           tuple_confidence_pair,
                                                           config.model.sampling_num,
                                                           config.model.use_absolute_residual)
                this_coverage = compute_coverage(hi, lo, target_residual)
                this_interval_width = compute_interval_width(hi, lo, normalized_std=None)
                this_winkler_score = compute_winkler_score(hi, lo,
                                                           target_y,
                                                           target_predictions,
                                                           tuple_confidence_pair,
                                                           normalized_params=None)

                evaluation_results[tuple_confidence_pair]["upper_interval"].extend(hi.cpu().detach().tolist())
                evaluation_results[tuple_confidence_pair]["lower_interval"].extend(lo.cpu().detach().tolist())
                evaluation_results[tuple_confidence_pair]["coverage"].extend(this_coverage)
                evaluation_results[tuple_confidence_pair]["interval_width"].extend(this_interval_width)
                evaluation_results[tuple_confidence_pair]["winkler_score"].extend(this_winkler_score)
                evaluation_results[tuple_confidence_pair]["target_y"].extend(target_y.flatten().tolist())
                evaluation_results[tuple_confidence_pair]["target_predictions"].extend(target_predictions.flatten().tolist())

        for confidence_pair in target_quantiles:
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
                    "valid_delta_coverages" : valid_delta_coverages,
                    "valid_interval_widths" : valid_interval_widths,
                    "best_epoch" : best_epoch,
                    "best_delta_coverage" : best_delta_coverage,
                    "best_interval_width" : best_interval_width,
                    "evaluation_results" : evaluation_results}
        
        torch.save(best_model, os.path.join(config.saving_dir, key + '_model.pt'))
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
                                       
