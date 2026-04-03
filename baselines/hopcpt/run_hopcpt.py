from omegaconf import OmegaConf
import torch
import os
import copy
import numpy as np
from tqdm import tqdm
from baselines.hopcpt.model import HopfieldNet
from baselines.hopcpt.loss import compute_hopfield_net_loss
from dscp.data import ConformalPredictionData, initialize_valid_dataloader, initialize_test_dataloader
from utils.utils import load_data, save_data, read_setup, generate_feature_hopcpt_training, generate_feature_hopcpt_test
from utils.utils import compute_coverage, compute_interval_width, compute_winkler_score, estimate_quantile_values
from torch.utils.data import DataLoader


def run_hopcpt(config_path):

    config = OmegaConf.load(config_path)
    os.makedirs(config.saving_dir, exist_ok=True)

    # load data
    data = load_data(config.data.data_path) # load predictor results here
    base_predictor, data_type = read_setup(config.data.data_path)
    cpd = ConformalPredictionData(data)

    cpd.prepare_hopcpt_datasets(config.model.memory_size, 
                                config.model.prediction_step, 
                                config.data.train_ratio, 
                                config.data.valid_ratio, 
                                config.data.normalize)

    device = config.device
    target_quantile = 1-(config.model.alpha/2)
    log = dict()

    print("Experiment setup")
    print("Method: HopCPT")
    print("Base predictor: {}".format(base_predictor))
    print("Data: {}".format(data_type))
    print("{} independent sequences".format(len(cpd.dataset)))
    
    for key, item in tqdm(cpd.dataset.items(), desc="repetition over independent sequences"):

        dataset = cpd.data[key]
        valid_dataset = item["valid_dataset"]
        test_dataset = item["test_dataset"]
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.training.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

        if config.data.memory_features == "xy":
            dim_feature = test_dataset.strided_x.shape[-1] + 1 # add 1 for y as input
        elif config.data.memory_features == "x":
            dim_feature = test_dataset.strided_x.shape[-1]
        else:
            raise ValueError("wrong strided features specified")

        # model initialization
        hopfield_net = HopfieldNet(dim_feature, 
                                   config.model.dim_context_encoding, 
                                   config.model.dim_hopfield_hidden, 
                                   config.model.beta)
        hopfield_net.to(device)
        optimizer = torch.optim.AdamW(hopfield_net.parameters(), 
                                      lr=config.training.learning_rate) # TODO: params for adamW?

        train_loss = []
        valid_delta_coverages = []
        valid_interval_widths = []
        best_interval_width = np.inf

        for i in tqdm(range(config.training.epochs), desc="training epochs"):

            hopfield_net.train()
            optimizer.zero_grad()
            memory_feature = generate_feature_hopcpt_training(dataset["heldout_train_x"],
                                                              dataset["heldout_train_y"],
                                                              config.data.memory_features)
            memory_feature = memory_feature.to(device) # (1, memeory_length, feature_dim)
            memory_residual = torch.from_numpy(dataset["heldout_train_residual"]).to(torch.float32).to(device)
            #print("memory_feature : {}".format(memory_feature.shape))
            #print("memory_residual : {}".format(memory_residual.shape))
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
                for strided_x, strided_y, target_x, \
                    target_predictions, strided_residual, target_residual, target_y in tqdm(valid_dataloader):
                    
                    memory_feature, query_feature = generate_feature_hopcpt_test(strided_x,
                                                                strided_y,
                                                                target_x,
                                                                target_predictions,
                                                                config.data.memory_features)
                    memory_feature = memory_feature.to(device) # (batch_size, memeory_length, feature_dim)
                    query_feature = query_feature.to(device) # (batch_size, 1, feature_dim)
                    # (batch_size, 1, 1, memory_length)
                    association_matrix = hopfield_net.obtain_association_matrix(memory_feature, query_feature)
                    # (1, batch_size)
                    estimated_quantile_values = estimate_quantile_values(association_matrix,
                                                                         strided_residual,
                                                                         target_quantile, 
                                                                         config.model.sampling_num)
                    
                    hi = estimated_quantile_values # (batch_size,)
                    lo = -estimated_quantile_values # (batch_size,)
                    this_coverage = compute_coverage(hi, lo, target_residual)
                    this_interval_width = compute_interval_width(hi, lo, normalized_std=None)
                    this_coverages.extend(this_coverage)
                    this_interval_widths.extend(this_interval_width)

                this_delta_coverage = np.mean(this_coverages) - (1-config.model.alpha)
                valid_delta_coverages.append(this_delta_coverage)
                this_avg_interval_width = np.mean(this_interval_widths)
                valid_interval_widths.append(this_avg_interval_width)
                print("delta coverage on validation set at epoch {}: {}".format(i+1, this_delta_coverage))
                print("avg interval width on validation set at epoch {}: {}".format(i+1, this_avg_interval_width))

                if this_delta_coverage > 0 and this_avg_interval_width < best_interval_width:
                    best_interval_width = this_avg_interval_width
                    best_epoch = i+1
                    best_model = copy.deepcopy(hopfield_net.state_dict())

        # load best model
        print("best model : epoch {}".format(best_epoch))
        hopfield_net.load_state_dict(best_model)

        evaluation_results = {"coverage": [], 
                              "interval_width": [],
                              "winkler_score" : [],
                              "upper_interval" : [],
                              "lower_interval" : [],
                              "target_y" : [],
                              "target_predictions" : []
                              }

        hopfield_net.eval()
        for strided_x, strided_y, target_x, \
            target_predictions, strided_residual, target_residual, target_y in tqdm(test_dataloader):
            
            memory_feature, query_feature = generate_feature_hopcpt_test(strided_x,
                                                                         strided_y,
                                                                         target_x,
                                                                         target_predictions,
                                                                         config.data.memory_features)
            memory_feature = memory_feature.to(device) # (batch_size, memeory_length, feature_dim)
            query_feature = query_feature.to(device) # (batch_size, 1, feature_dim)
            # (batch_size, 1, 1, memory_length)
            association_matrix = hopfield_net.obtain_association_matrix(memory_feature, query_feature)
            # (1, batch_size)
            estimated_quantile_values = estimate_quantile_values(association_matrix,
                                                                 strided_residual,
                                                                 target_quantile, 
                                                                 config.model.sampling_num)
            
            hi = estimated_quantile_values # (batch_size,)
            lo = -estimated_quantile_values # (batch_size,)
            this_coverage = compute_coverage(hi, lo, target_residual)
            this_interval_width = compute_interval_width(hi, lo, normalized_std=None)
            this_winkler_score = compute_winkler_score(hi, lo, 
                                                       target_y, 
                                                       target_predictions, 
                                                       config.model.alpha, 
                                                       normalized_params=None)
                
            evaluation_results["upper_interval"].extend(hi.cpu().detach().tolist())
            evaluation_results["lower_interval"].extend(lo.cpu().detach().tolist())
            evaluation_results["coverage"].extend(this_coverage)
            evaluation_results["interval_width"].extend(this_interval_width)
            evaluation_results["winkler_score"].extend(this_winkler_score)
            evaluation_results["target_y"].extend(target_y)
            evaluation_results["target_predictions"].extend(target_predictions)

        avg_coverage = np.mean(evaluation_results["coverage"])
        avg_interval_width = np.mean(evaluation_results["interval_width"])
        avg_winkler_score = np.mean(evaluation_results["winkler_score"])
        print("avg coverage: {}".format(avg_coverage))
        print("avg interval width: {}".format(avg_interval_width))
        print("avg winkler score: {}".format(avg_winkler_score))
        evaluation_results["avg_coverage"] = avg_coverage
        evaluation_results["avg_interval_width"] = avg_interval_width
        evaluation_results["avg_winkler_score"] = avg_winkler_score

        log[key] = {"train_loss" : train_loss,
                    "valid_delta_coverages" : valid_delta_coverages,
                    "evaluation_results" : evaluation_results}
        
        torch.save(best_model, os.path.join(config.saving_dir, key + '_model.pt'))
        save_data(os.path.join(config.saving_dir, "log.pkl"), log)
                                       

def run_hopcpt_max_memory(config_path):

    config = OmegaConf.load(config_path)
    os.makedirs(config.saving_dir, exist_ok=True)

    # load data
    data = load_data(config.data.data_path) # load predictor results here
    base_predictor, data_type = read_setup(config.data.data_path)
    cpd = ConformalPredictionData(data)

    cpd.prepare_hopcpt_datasets_max_memory(config.model.memory_size, 
                                           config.model.prediction_step, 
                                           config.data.train_ratio, 
                                           config.data.valid_ratio, 
                                           config.data.normalize)

    device = config.device
    target_quantile = 1-(config.model.alpha/2)
    log = dict()

    print("Experiment setup")
    print("Method: HopCPT")
    print("Base predictor: {}".format(base_predictor))
    print("Data: {}".format(data_type))
    print("{} independent sequences".format(len(cpd.dataset)))
    
    for key, data in tqdm(cpd.data.items(), desc="repetition over independent sequences"):

        train_size = data["train_size"]
        valid_size = data["valid_size"]
        test_size = data["test_size"]
        valid_dataloader_size = data["valid_dataloader_size"]
        test_dataloader_size = data["test_dataloader_size"]

        if config.data.memory_features == "xy":
            dim_feature = data["heldout_train_x"].shape[-1] + 1 # add 1 for y as input
        elif config.data.memory_features == "x":
            dim_feature = data["heldout_train_x"].shape[-1]
        else:
            raise ValueError("wrong strided features specified")

        # model initialization
        hopfield_net = HopfieldNet(dim_feature, 
                                   config.model.dim_context_encoding, 
                                   config.model.dim_hopfield_hidden, 
                                   config.model.beta)
        hopfield_net.to(device)
        optimizer = torch.optim.AdamW(hopfield_net.parameters(), 
                                      lr=config.training.learning_rate) # TODO: params for adamW?

        train_loss = []
        valid_delta_coverages = []
        valid_interval_widths = []
        best_interval_width = np.inf

        for i in tqdm(range(config.training.epochs), desc="training epochs"):

            hopfield_net.train()
            optimizer.zero_grad()
            memory_feature = generate_feature_hopcpt_training(data["heldout_train_x"],
                                                              data["heldout_train_y"],
                                                              config.data.memory_features)
            memory_feature = memory_feature.to(device) # (1, memeory_length, feature_dim)
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

                valid_x_dataloader, valid_y_dataloader, \
                    valid_target_y_dataloader, valid_residual_dataloader, \
                        valid_prediction_dataloader = initialize_valid_dataloader(data, 
                                                                                 train_size, 
                                                                                 valid_size, 
                                                                                 config.model.prediction_step, 
                                                                                 config.model.memory_size,
                                                                                 config.data.normalize)

                for j in range(valid_dataloader_size):

                    strided_x, target_x = next(valid_x_dataloader)
                    strided_y, _ = next(valid_y_dataloader)
                    _, target_y = next(valid_target_y_dataloader)
                    strided_residual, target_residual = next(valid_residual_dataloader)
                    _, target_predictions = next(valid_prediction_dataloader)
                    
                    memory_feature, query_feature = generate_feature_hopcpt_test(strided_x,
                                                                                 strided_y,
                                                                                 target_x,
                                                                                 target_predictions,
                                                                                 config.data.memory_features)
                    print(memory_feature.shape)
                    memory_feature = memory_feature.to(device) # (batch_size, memeory_length, feature_dim)
                    query_feature = query_feature.to(device) # (batch_size, 1, feature_dim)
                    # (batch_size, 1, 1, memory_length)
                    association_matrix = hopfield_net.obtain_association_matrix(memory_feature, query_feature)
                    # (1, batch_size)
                    estimated_quantile_values = estimate_quantile_values(association_matrix,
                                                                         strided_residual,
                                                                         target_quantile, 
                                                                         config.model.sampling_num)
                    
                    hi = estimated_quantile_values # (batch_size,)
                    lo = -estimated_quantile_values # (batch_size,)
                    this_coverage = compute_coverage(hi, lo, target_residual)
                    this_interval_width = compute_interval_width(hi, lo, normalized_std=None)
                    this_coverages.extend(this_coverage)
                    this_interval_widths.extend(this_interval_width)

                this_delta_coverage = np.mean(this_coverages) - (1-config.model.alpha)
                valid_delta_coverages.append(this_delta_coverage)
                this_avg_interval_width = np.mean(this_interval_widths)
                valid_interval_widths.append(this_avg_interval_width)
                print("delta coverage on validation set at epoch {}: {}".format(i+1, this_delta_coverage))
                print("avg interval width on validation set at epoch {}: {}".format(i+1, this_avg_interval_width))

                if this_delta_coverage > 0 and this_avg_interval_width < best_interval_width:
                    best_interval_width = this_avg_interval_width
                    best_epoch = i+1
                    best_model = copy.deepcopy(hopfield_net.state_dict())

        # load best model
        print("best model : epoch {}".format(best_epoch))
        hopfield_net.load_state_dict(best_model)

        test_x_dataloader, test_y_dataloader, \
            test_target_y_dataloader, test_residual_dataloader, \
                test_prediction_dataloader = initialize_test_dataloader(data, 
                                                                        train_size, 
                                                                        valid_size, 
                                                                        config.model.prediction_step, 
                                                                        config.model.memory_size,
                                                                        config.data.normalize)

        evaluation_results = {"coverage": [], 
                              "interval_width": [],
                              "winkler_score" : [],
                              "upper_interval" : [],
                              "lower_interval" : [],
                              "target_y" : [],
                              "target_predictions" : []
                              }

        hopfield_net.eval()
        for j in range(test_dataloader_size):

            strided_x, target_x = next(test_x_dataloader)
            strided_y, _ = next(test_y_dataloader)
            _, target_y = next(test_target_y_dataloader)
            strided_residual, target_residual = next(test_residual_dataloader)
            _, target_predictions = next(test_prediction_dataloader)
            
            memory_feature, query_feature = generate_feature_hopcpt_test(strided_x,
                                                                         strided_y,
                                                                         target_x,
                                                                         target_predictions,
                                                                         config.data.memory_features)
            memory_feature = memory_feature.to(device) # (batch_size, memeory_length, feature_dim)
            query_feature = query_feature.to(device) # (batch_size, 1, feature_dim)
            # (batch_size, 1, 1, memory_length)
            association_matrix = hopfield_net.obtain_association_matrix(memory_feature, query_feature)
            # (1, batch_size)
            estimated_quantile_values = estimate_quantile_values(association_matrix,
                                                                 strided_residual,
                                                                 target_quantile, 
                                                                 config.model.sampling_num)
            
            hi = estimated_quantile_values # (batch_size,)
            lo = -estimated_quantile_values # (batch_size,)
            this_coverage = compute_coverage(hi, lo, target_residual)
            this_interval_width = compute_interval_width(hi, lo, normalized_std=None)
            this_winkler_score = compute_winkler_score(hi, lo, 
                                                       target_y, 
                                                       target_predictions, 
                                                       config.model.alpha, 
                                                       normalized_params=None)
                
            evaluation_results["upper_interval"].extend(hi.cpu().detach().tolist())
            evaluation_results["lower_interval"].extend(lo.cpu().detach().tolist())
            evaluation_results["coverage"].extend(this_coverage)
            evaluation_results["interval_width"].extend(this_interval_width)
            evaluation_results["winkler_score"].extend(this_winkler_score)
            evaluation_results["target_y"].extend(target_y)
            evaluation_results["target_predictions"].extend(target_predictions)

        avg_coverage = np.mean(evaluation_results["coverage"])
        avg_interval_width = np.mean(evaluation_results["interval_width"])
        avg_winkler_score = np.mean(evaluation_results["winkler_score"])
        print("avg coverage: {}".format(avg_coverage))
        print("avg interval width: {}".format(avg_interval_width))
        print("avg winkler score: {}".format(avg_winkler_score))
        evaluation_results["avg_coverage"] = avg_coverage
        evaluation_results["avg_interval_width"] = avg_interval_width
        evaluation_results["avg_winkler_score"] = avg_winkler_score

        log[key] = {"train_loss" : train_loss,
                    "valid_delta_coverages" : valid_delta_coverages,
                    "evaluation_results" : evaluation_results}
        
        torch.save(best_model, os.path.join(config.saving_dir, key + '_model.pt'))
        save_data(os.path.join(config.saving_dir, "log.pkl"), log)
                                       
