import torch
import numpy as np
from torch.utils.data import Dataset
from utils.utils import to_strided_feature, to_strided_residual, chronological_split_fixed_test
from utils.utils import build_hopcpt_context_features
from utils.utils import normalize_array_with_params, compute_mean_std


class ConformalPredictionData:
    """
    Data class to load data for conformal prediction
    """

    def __init__(self, data):
        super(ConformalPredictionData, self).__init__()
        self.data = data
        self.dataset = dict()

    def prepare_quantile_regression_datasets(self, 
                                             past_window, 
                                             prediction_steps, 
                                             train_ratio, 
                                             valid_ratio, 
                                             normalize=False):
        """
        prepare dscp dataset to predict t only with past_window context of features and residuals.

        shifted_concatenation only changes how strided feature is constructed.

        """

        for key, item in self.data.items():
            
            # compute residuals
            heldout_residuals = (item["heldout_y"] - item["heldout_predictions"]).flatten()
            self.data[key].update({"heldout_residuals" : heldout_residuals})

            heldout_size = len(item["heldout_y"])
            train_size = int(np.floor(heldout_size*train_ratio))
            valid_size = int(np.ceil(heldout_size*valid_ratio))
            test_size = heldout_size - (train_size+valid_size)

            if normalize:
                # normalize variables that will be used for sequence model prediction
                # NOTE: normalize should be conducted only with training set
                
                train_x_mu, train_x_std = compute_mean_std(item["heldout_x"][:train_size])
                heldout_x = normalize_array_with_params(item["heldout_x"], train_x_mu, train_x_std)

                train_residuals_mu, train_residuals_std = compute_mean_std(item["heldout_residuals"][:train_size])
                heldout_residuals = normalize_array_with_params(item["heldout_residuals"], train_residuals_mu, train_residuals_std)

                train_y_mu, train_y_std = compute_mean_std(item["heldout_y"][:train_size])
                heldout_y = normalize_array_with_params(item["heldout_y"], train_y_mu, train_y_std)

            else:
                heldout_x = item["heldout_x"]
                heldout_residuals = item["heldout_residuals"]
                heldout_y = item["heldout_y"]

            strided_x, target_x = to_strided_feature(heldout_x, 
                                                     past_window, 
                                                     prediction_steps,
                                                     return_target=True) # (seq_len-window_len, window_len, dim), (seq_len-window_len, 1, dim)
            strided_y = to_strided_feature(heldout_y, 
                                           past_window, 
                                           prediction_steps) # (seq_len-window_len, window_len, dim)
            # residual_target: (seq_len-window_len-pred_horizon+1, pred_horizon)
            strided_residual, target_residual = to_strided_residual(heldout_residuals, 
                                                                    past_window, 
                                                                    prediction_steps)
            # target_y does not need to be normalized
            _, target_y = to_strided_residual(item["heldout_y"], 
                                              past_window, 
                                              prediction_steps)
            _, target_predictions = to_strided_residual(item["heldout_predictions"],
                                                        past_window, 
                                                        prediction_steps)
            
            if normalize:
                self.data[key].update({"strided_x" : strided_x,
                                       "target_x" : target_x,
                                       "strided_residual" : strided_residual, 
                                        "strided_y" : strided_y,
                                        "target_residual" : target_residual,
                                        "target_y" : target_y,
                                        "target_predictions" : target_predictions,
                                        "heldout_x_normalized" : heldout_x,
                                        "train_x_mu" : train_x_mu,
                                        "train_x_std" : train_x_std,
                                        "heldout_residuals" : heldout_residuals,
                                        "train_residuals_mu" : train_residuals_mu,
                                        "train_residuals_std" : train_residuals_std,
                                        "heldout_y_normalized" : heldout_y,
                                        "train_y_mu" : train_y_mu,
                                        "train_y_std" : train_y_std})
            else:
                self.data[key].update({"strided_x" : strided_x,
                                       "target_x" : target_x,
                                        "strided_residual" : strided_residual,
                                        "strided_y" : strided_y,
                                        "target_residual" : target_residual,
                                        "target_y" : target_y,
                                        "target_predictions" : target_predictions,
                                        "heldout_residuals" : heldout_residuals})
            
            train_split, valid_split, test_split = chronological_split_fixed_test([strided_x,
                                                                                   strided_residual,
                                                                                   strided_y,
                                                                                   target_x,
                                                                                   target_residual,
                                                                                   target_y,
                                                                                   target_predictions],
                                                                                   valid_size, 
                                                                                   test_size)

            self.dataset[key] = {"train_dataset" : QuantileRegressionDataset(train_split[0], 
                                                                             train_split[1], 
                                                                             train_split[2], 
                                                                             train_split[3], 
                                                                             train_split[4],
                                                                             train_split[5],
                                                                             train_split[6]),
                                 "valid_dataset" : QuantileRegressionDataset(valid_split[0], 
                                                                             valid_split[1], 
                                                                             valid_split[2],
                                                                             valid_split[3],
                                                                             valid_split[4],
                                                                             valid_split[5],
                                                                             valid_split[6]),
                                 "test_dataset" : QuantileRegressionDataset(test_split[0], 
                                                                            test_split[1], 
                                                                            test_split[2],
                                                                            test_split[3],
                                                                            test_split[4],
                                                                            test_split[5],
                                                                            test_split[6])}

    def prepare_hopcpt_datasets(self, 
                                prediction_steps, 
                                y_lags,
                                train_ratio, 
                                valid_ratio, 
                                normalize=False,
                                predict_absolute_residual=True,
                                conformal_absolute_residual=False):

        for key, item in self.data.items():
            
            # compute residuals
            heldout_signed_residuals = (item["heldout_y"] - item["heldout_predictions"]).flatten()
            heldout_train_residuals = heldout_signed_residuals
            heldout_conformal_residuals = heldout_signed_residuals
            if predict_absolute_residual:
                heldout_train_residuals = np.abs(heldout_train_residuals)
            if conformal_absolute_residual:
                heldout_conformal_residuals = np.abs(heldout_conformal_residuals)
            self.data[key].update({"heldout_residuals" : heldout_conformal_residuals,
                                   "heldout_signed_residuals" : heldout_signed_residuals,
                                   "heldout_train_residuals" : heldout_train_residuals})
            
            heldout_size = len(item["heldout_y"])
            train_size = int(np.floor(heldout_size*train_ratio))
            valid_size = int(np.ceil(heldout_size*valid_ratio))
            test_size = heldout_size - (train_size+valid_size)

            if normalize:
                # normalize variables that will be used for sequence model prediction
                # NOTE: normalize should be conducted only with training set
                heldout_size = len(item["heldout_y"])
                train_size = int(np.floor(heldout_size*train_ratio))

                train_x_mu, train_x_std = compute_mean_std(item["heldout_x"][:train_size])
                heldout_x = normalize_array_with_params(item["heldout_x"], train_x_mu, train_x_std)

                #train_residuals_mu, train_residuals_std = compute_mean_std(item["heldout_residuals"][:train_size])
                #heldout_residuals = normalize_array_with_params(item["heldout_residuals"], train_residuals_mu, train_residuals_std)

                train_y_mu, train_y_std = compute_mean_std(item["heldout_y"][:train_size])
                heldout_y = normalize_array_with_params(item["heldout_y"], train_y_mu, train_y_std)

                # use mu and std of y for predicted y since this is going to be used for query
                heldout_predictions = normalize_array_with_params(item["heldout_predictions"], train_y_mu, train_y_std)

            else:
                heldout_x = item["heldout_x"]
                heldout_train_residuals = item["heldout_train_residuals"]
                heldout_conformal_residuals = item["heldout_residuals"]
                heldout_y = item["heldout_y"]
                heldout_predictions = item["heldout_predictions"]

            train_x = heldout_x[:train_size]
            valid_x = heldout_x[train_size:train_size+valid_size]
            train_y = heldout_y[:train_size]
            valid_y = heldout_y[train_size:train_size+valid_size]
            train_residual = heldout_train_residuals[y_lags:train_size]
            valid_residual = heldout_conformal_residuals[train_size:train_size+valid_size]

            heldout_context = build_hopcpt_context_features(heldout_x,
                                                            heldout_y,
                                                            heldout_predictions,
                                                            y_lags)
            heldout_context_residuals = heldout_conformal_residuals[y_lags:]
            heldout_target_y = np.asarray(item["heldout_y"])[y_lags:]
            heldout_target_predictions = np.asarray(item["heldout_predictions"])[y_lags:]

            if train_size <= y_lags:
                raise ValueError("train split must contain more observations than y_lags.")
            train_context = heldout_context[:train_size-y_lags]

            if normalize:
                self.data[key].update({"heldout_train_x" : train_x,
                                       "heldout_valid_x" : valid_x,
                                       "heldout_train_y" : train_y,
                                       "heldout_valid_y" : valid_y,
                                       "heldout_train_context" : train_context,
                                       "heldout_train_residual" : train_residual,
                                       "heldout_valid_residual" : valid_residual,
                                       "heldout_context" : heldout_context,
                                       "heldout_context_residuals" : heldout_context_residuals,
                                       "heldout_target_y" : heldout_target_y,
                                       "heldout_target_predictions" : heldout_target_predictions,
                                       "heldout_x_normalized" : heldout_x,
                                       "heldout_train_x_mu" : train_x_mu,
                                       "heldout_train_x_std" : train_x_std,
                                       "heldout_residuals" : heldout_conformal_residuals,
                                       "heldout_signed_residuals" : heldout_signed_residuals,
                                       "heldout_train_residuals" : heldout_train_residuals,
                                       "heldout_y_normalized" : heldout_y,
                                       "heldout_train_y_mu" : train_y_mu,
                                       "heldout_train_y_std" : train_y_std,
                                       "heldout_predictions" : heldout_predictions,
                                       "y_lags" : y_lags,
                                       "train_size" : train_size,
                                       "valid_size" : valid_size,
                                       "test_size" : test_size,
                                       "valid_dataloader_size" : valid_size - prediction_steps + 1,
                                       "test_dataloader_size" : test_size - prediction_steps + 1
                                       })
            else:
                self.data[key].update({"heldout_train_x" : train_x,
                                       "heldout_valid_x" : valid_x,
                                       "heldout_train_y" : train_y,
                                       "heldout_valid_y" : valid_y,
                                       "heldout_train_context" : train_context,
                                       "heldout_train_residual" : train_residual,
                                       "heldout_valid_residual" : valid_residual,
                                       "heldout_context" : heldout_context,
                                       "heldout_context_residuals" : heldout_context_residuals,
                                       "heldout_target_y" : heldout_target_y,
                                       "heldout_target_predictions" : heldout_target_predictions,
                                       "heldout_residuals" : heldout_conformal_residuals,
                                       "heldout_signed_residuals" : heldout_signed_residuals,
                                       "heldout_train_residuals" : heldout_train_residuals,
                                       "heldout_predictions" : heldout_predictions,
                                       "y_lags" : y_lags,
                                       "train_size" : train_size,
                                       "valid_size" : valid_size,
                                       "test_size" : test_size,
                                       "valid_dataloader_size" : valid_size - prediction_steps + 1,
                                       "test_dataloader_size" : test_size - prediction_steps + 1
                                       })
                

def initialize_valid_dataloader(data, train_size, valid_size, prediction_steps, max_memory_size, normalize):
    """
    initialize dataloader, which is a generator
    """
    del normalize
    y_lags = data["y_lags"]
    start_idx = train_size - y_lags
    end_idx = train_size + valid_size - y_lags

    valid_context_generator = prefix_gen(data["heldout_context"][:end_idx],
                                         start_idx,
                                         prediction_steps,
                                         max_memory=max_memory_size)
    valid_target_y_generator = prefix_gen(data["heldout_target_y"][:end_idx],
                                          start_idx,
                                          prediction_steps,
                                          max_memory=max_memory_size)
    valid_residual_generator = prefix_gen(data["heldout_context_residuals"][:end_idx],
                                          start_idx,
                                          prediction_steps,
                                          max_memory=max_memory_size)
    valid_prediction_generator = prefix_gen(data["heldout_target_predictions"][:end_idx],
                                            start_idx,
                                            prediction_steps,
                                            max_memory=max_memory_size)
    
    return (valid_context_generator, valid_target_y_generator, valid_residual_generator, valid_prediction_generator)


def initialize_test_dataloader(data, train_size, valid_size, prediction_steps, max_memory_size, normalize):
    """
    initialize dataloader, which is a generator
    """
    del normalize
    y_lags = data["y_lags"]
    start_idx = train_size + valid_size - y_lags
    
    test_context_generator = prefix_gen(data["heldout_context"],
                                        start_idx,
                                        prediction_steps,
                                        max_memory=max_memory_size)
    test_target_y_generator = prefix_gen(data["heldout_target_y"],
                                         start_idx,
                                         prediction_steps,
                                         max_memory=max_memory_size)
    test_residual_generator = prefix_gen(data["heldout_context_residuals"],
                                         start_idx,
                                         prediction_steps,
                                         max_memory=max_memory_size)
    test_prediction_generator = prefix_gen(data["heldout_target_predictions"],
                                           start_idx,
                                           prediction_steps,
                                           max_memory=max_memory_size)
    
    return (test_context_generator, test_target_y_generator, test_residual_generator, test_prediction_generator)

                

def prefix_gen(x, start_idx, prediction_steps=1, max_memory=None):
    """
    x: array/tensor of shape (num_data, feature_dim)
    start_idx: int, first i to start generating from
    prediction_steps: int, length of the future target window
    max_memory: int or None, if set bounds past length to <= max_memory

    Yields:
        past  : (L, feature_dim) where L == i (or <= max_memory)
        future: (prediction_steps, feature_dim)
    """
    N = x.shape[0]

    if not (0 <= start_idx < N):
        raise ValueError("start_idx must be in [0, num_data-1].")
    if prediction_steps <= 0:
        raise ValueError("prediction_steps must be positive.")
    if max_memory is not None and max_memory <= 0:
        raise ValueError("max_memory must be positive or None.")

    # last i must satisfy i + prediction_steps <= N
    last_i = N - prediction_steps
    if start_idx > last_i:
        raise ValueError("start_idx is too large: no room for prediction_steps targets.")

    for i in range(start_idx, last_i + 1):
        past_start = 0 if max_memory is None else max(0, i - max_memory)
        past = x[past_start:i]                 # (L, d), aligns with your x[:i]
        future = x[i:i + prediction_steps]     # (prediction_steps, d)
        yield torch.tensor(past).to(torch.float32), torch.tensor(future).to(torch.float32)


class QuantileRegressionDataset(Dataset):
    """
    Dataset class for quantile regression using sequence models
    """
    
    def __init__(self, strided_x, strided_residual, strided_y, target_x, target_residual, target_y, target_predictions):
        self.strided_x = torch.from_numpy(strided_x.copy()).to(torch.float32)
        self.strided_residual = torch.from_numpy(strided_residual.copy()).to(torch.float32)
        self.strided_y = torch.from_numpy(strided_y.copy()).to(torch.float32)
        self.target_x = torch.from_numpy(target_x.copy()).to(torch.float32)
        self.target_residual = torch.from_numpy(target_residual.copy()).to(torch.float32)
        self.target_y = torch.from_numpy(target_y.copy()).to(torch.float32)
        self.target_predictions = torch.from_numpy(target_predictions.copy()).to(torch.float32)

    def __len__(self):
        return len(self.target_residual)

    def __getitem__(self,idx):
        return self.strided_x[idx], self.strided_residual[idx], self.strided_y[idx], \
            self.target_x[idx], self.target_residual[idx], self.target_y[idx], self.target_predictions[idx]
    

class HopCPTTestDataset(Dataset):
    """
    Dataset class for conditional CDF approximation for HopCPT
    """

    def __init__(self, strided_context, target_context, target_predictions, strided_residual, target_residual, target_y):
        self.strided_context = torch.from_numpy(strided_context.copy()).to(torch.float32)
        self.target_context = torch.from_numpy(target_context.copy()).to(torch.float32)
        self.target_predictions = torch.from_numpy(target_predictions.copy()).to(torch.float32)
        self.strided_residual = torch.from_numpy(strided_residual.copy()).to(torch.float32)
        self.target_residual = torch.from_numpy(target_residual.copy()).to(torch.float32)
        self.target_y = torch.from_numpy(target_y.copy()).to(torch.float32)

    def __len__(self):
        return len(self.target_residual)

    def __getitem__(self,idx):
        return self.strided_context[idx], self.target_context[idx], self.target_predictions[idx], \
            self.strided_residual[idx], self.target_residual[idx], self.target_y[idx]

