from base_predictor.data import BasePredictorData
from utils.utils import save_data, normalize_array, normalize_array_with_params, denormalize_array
import torch
import numpy as np
from tqdm import tqdm
import os
import copy
from torch.utils.data import Dataset, DataLoader


class LSTM(torch.nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers):
        super(LSTM, self).__init__()

        self.input_embedding = torch.nn.Linear(input_dim, embedding_dim)
        self.lstm_layers = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_embedding(x) 
        output, (_,_) = self.lstm_layers(x)

        return self.output_fc(output)
    

class LSTMPredictor:
    """
    Global rolling one-step LSTM predictor using past covariates and targets.
    """

    def __init__(self, data: BasePredictorData, embedding_dim, hidden_dim, num_layers, train_ratio, window_length):
        super(LSTMPredictor, self).__init__()
        self.covariate_dim = next(iter(data.data.items()))[1]["x"].shape[-1]
        self.input_dim = self.covariate_dim + 1
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = LSTM(self.input_dim, embedding_dim, hidden_dim, 1, num_layers)
        self.data = data.data
        self.data_type = data.data_type
        self.train_ratio = train_ratio
        self.window_length = window_length
        self.data_processed = dict()
        self.predictions = dict()
        self._process_data(window_length)

    def _process_data(self, window_length):
        print("{} time series identified".format(len(self.data)))
        print("merging {} time series to train a single LSTM predictor".format(len(self.data)))

        for key, item in self.data.items():
            x = item["x"]
            y = item["y"]
            train_x, heldout_x = _split_before(x, self.train_ratio)
            train_y, heldout_y = _split_before(y, self.train_ratio)

            self.data_processed[key] = {"train_x" : train_x,
                                        "heldout_x" : heldout_x,
                                        "train_y" : train_y,
                                        "heldout_y" : heldout_y
            }

    def _prepare_fit_data(self, fit_train_ratio):
        """Create per-sequence chronological train/validation windows.

        Normalization parameters are estimated globally, but only from each
        sequence's inner-training prefix.  The parameters are then held fixed
        when normalizing inner-validation and outer held-out observations.
        """

        self.fit_train_ratio = fit_train_ratio
        normalization_x = []
        normalization_y = []

        for key, item in self.data_processed.items():
            train_x = item["train_x"]
            train_y = item["train_y"]

            if len(train_x) != len(train_y):
                raise ValueError(
                    "Invalid fitting data for {}: x and y must have the same length".format(key)
                )

            try:
                inner_train_end = _inner_split_index(
                    len(train_y),
                    self.window_length,
                    fit_train_ratio,
                )
            except ValueError as exc:
                raise ValueError("Invalid inner split for {}: {}".format(key, exc)) from exc

            normalization_x.append(train_x[:inner_train_end])
            normalization_y.append(train_y[:inner_train_end])
            item["inner_train_end"] = inner_train_end

        merged_inner_train_x = np.concatenate(normalization_x, axis=0)
        merged_inner_train_y = np.concatenate(normalization_y, axis=0)
        _, (train_x_mu, train_x_std) = normalize_array(merged_inner_train_x)
        _, (train_y_mu, train_y_std) = normalize_array(merged_inner_train_y)

        for item in self.data_processed.values():
            train_x = item["train_x"]
            heldout_x = item["heldout_x"]
            train_y = item["train_y"]
            heldout_y = item["heldout_y"]

            normalized_train_x = normalize_array_with_params(train_x, train_x_mu, train_x_std)
            normalized_heldout_x = normalize_array_with_params(heldout_x, train_x_mu, train_x_std)
            normalized_train_y = normalize_array_with_params(train_y, train_y_mu, train_y_std)
            normalized_heldout_y = normalize_array_with_params(heldout_y, train_y_mu, train_y_std)

            inner_train_end = item["inner_train_end"]
            train_input_seq, train_y_seq = _make_sequence_prediction_data(
                normalized_train_x[:inner_train_end],
                normalized_train_y[:inner_train_end],
                self.window_length,
            )
            valid_input_seq, valid_y_seq = _make_heldout_sequence_prediction_data(
                normalized_train_x[:inner_train_end],
                normalized_train_x[inner_train_end:],
                normalized_train_y[:inner_train_end],
                normalized_train_y[inner_train_end:],
                self.window_length,
            )

            item.update({"normalized_train_x" : normalized_train_x,
                                             "normalized_heldout_x" : normalized_heldout_x,
                                             "normalized_train_y" : normalized_train_y,
                                             "normalized_heldout_y" : normalized_heldout_y,
                                             "train_input_seq" : train_input_seq,
                                             "train_y_seq" : train_y_seq,
                                             "valid_input_seq" : valid_input_seq,
                                             "valid_y_seq" : valid_y_seq,
                                             "train_x_mu" : train_x_mu,
                                             "train_x_std" : train_x_std,
                                             "train_y_mu" : train_y_mu, 
                                             "train_y_std" : train_y_std})
        
    def fit_predict(self, train_ratio, batch_size, learning_rate, max_epoch, early_stop, seed=2026, device=0):
        """
        Fit the global LSTM and make fixed-weight rolling one-step predictions.

        ``train_ratio`` is the chronological inner-training fraction applied
        independently to every sequence's outer fitting prefix.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(max_epoch, int) or max_epoch <= 0:
            raise ValueError("max_epoch must be a positive integer")
        if not isinstance(early_stop, int) or early_stop <= 0:
            raise ValueError("early_stop must be a positive integer")

        self._prepare_fit_data(train_ratio)

        # Pool the already-separated per-sequence training and validation
        # windows independently. Validation remains ordered so that its loss
        # can be averaged within each sequence before averaging across series.
        merged_train_input_seq = []
        merged_train_y_seq = []
        merged_valid_input_seq = []
        merged_valid_y_seq = []
        valid_sequence_lengths = []

        for _, item in tqdm(self.data_processed.items()):
            merged_train_input_seq.append(item["train_input_seq"])
            merged_train_y_seq.append(item["train_y_seq"])
            merged_valid_input_seq.append(item["valid_input_seq"])
            merged_valid_y_seq.append(item["valid_y_seq"])
            valid_sequence_lengths.append(len(item["valid_y_seq"]))

        merged_train_input_seq = np.vstack(merged_train_input_seq)
        merged_train_y_seq = np.vstack(merged_train_y_seq)
        merged_valid_input_seq = np.vstack(merged_valid_input_seq)
        merged_valid_y_seq = np.vstack(merged_valid_y_seq)
        merged_train_input_seq, merged_train_y_seq = _shuffle_in_unison(
            merged_train_input_seq,
            merged_train_y_seq,
            seed,
        )

        train_dataset = LSTMPredictorDataset(merged_train_input_seq, merged_train_y_seq)
        valid_dataset = LSTMPredictorDataset(merged_valid_input_seq, merged_valid_y_seq)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        optim = torch.optim.Adam(self.lstm.parameters(), lr=learning_rate)

        best_loss = float("inf")
        best_epoch = 0
        best_model = copy.deepcopy(self.lstm.state_dict())
        epochs_without_improvement = 0
        self.lstm.to(device)
        for e in tqdm(range(max_epoch)):
            # training
            self.lstm.train()
            squared_error_sum = 0.0
            target_count = 0
            for x_batch, y_batch in tqdm(train_dataloader):

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                out = self.lstm(x_batch) # (batch_size, window_length, 1)
                preds = out[:,-1,:] # (batch_size, 1)
                loss = torch.nn.functional.mse_loss(preds, y_batch)
                optim.zero_grad()
                loss.backward()
                optim.step()
                squared_error_sum += torch.nn.functional.mse_loss(
                    preds.detach(),
                    y_batch,
                    reduction="sum",
                ).item()
                target_count += y_batch.numel()

            epoch_train_loss = squared_error_sum / target_count
            print("train loss at epoch {} : {}".format(e+1, epoch_train_loss))

            self.lstm.eval()
            valid_squared_errors = []
            for x_batch, y_batch in tqdm(valid_dataloader):

                with torch.no_grad():
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    out = self.lstm(x_batch) # (batch_size, window_length, 1)
                    preds = out[:,-1,:] # (batch_size, 1)
                    per_example_error = (preds - y_batch).square().reshape(
                        y_batch.shape[0],
                        -1,
                    ).mean(dim=1)
                    valid_squared_errors.append(per_example_error.cpu().numpy())

            epoch_valid_loss = _mean_per_sequence_mse(
                np.concatenate(valid_squared_errors),
                valid_sequence_lengths,
            )
            print("valid loss at epoch {} : {}".format(e+1, epoch_valid_loss))
            
            if epoch_valid_loss < best_loss:
                best_loss = epoch_valid_loss
                best_epoch = e+1
                best_model = copy.deepcopy(self.lstm.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stop:
                # if the loss did not decrease for (early_stop) epoch in a row, stop training
                break
        
        self.best_epoch = best_epoch
        self.best_validation_loss = best_loss
        print("best model at epoch {}".format(best_epoch))
        print("making predictions on the heldout data")
        mse_list = []
        mae_list = []
        self.lstm.load_state_dict(best_model)
        self.lstm.eval()
        for key, item in tqdm(self.data_processed.items()):

            normalized_train_x = item["normalized_train_x"]
            normalized_heldout_x = item["normalized_heldout_x"]
            normalized_train_y = item["normalized_train_y"]
            normalized_heldout_y = item["normalized_heldout_y"]
            y_mu = item["train_y_mu"]
            y_std = item["train_y_std"]

            heldout_input_seq, heldout_y_seq = _make_heldout_sequence_prediction_data(
                normalized_train_x,
                normalized_heldout_x,
                normalized_train_y,
                normalized_heldout_y,
                self.window_length,
            )
            heldout_dataset = LSTMPredictorDataset(heldout_input_seq, heldout_y_seq)
            heldout_dataloader = DataLoader(heldout_dataset, batch_size=batch_size, shuffle=False)

            predictions = []
            series_targets = []
            for x_batch, y_batch in tqdm(heldout_dataloader):

                with torch.no_grad():
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    out = self.lstm(x_batch)
                    preds = out[:,-1,:]
                    denormalized_y_batch = denormalize_array(y_batch, y_mu, y_std)
                    denormalized_preds = denormalize_array(preds, y_mu, y_std)
                    predictions.append(denormalized_preds)
                    series_targets.append(denormalized_y_batch)

            series_predictions = torch.vstack(predictions)
            series_targets = torch.vstack(series_targets)

            mse = torch.nn.functional.mse_loss(series_predictions, series_targets)
            mae = torch.nn.functional.l1_loss(series_predictions, series_targets)
            mse_list.append(mse.item())
            mae_list.append(mae.item())

            self.predictions[key] = series_predictions.cpu()

        self.average_mse = np.mean(mse_list)
        self.average_mae = np.mean(mae_list)
        print("average MSE : {}".format(self.average_mse))
        print("average MAE : {}".format(self.average_mae))

    def save(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)
        data_record_save_path = os.path.join(save_dir, f"lstm_{self.data_type}_data.pkl")
        data_record = dict()

        for key, item in tqdm(self.data_processed.items()):

            data_record[key] = {"train_x" : item["train_x"],
                                "heldout_x" : item["heldout_x"],
                                "train_y" : item["train_y"].reshape(-1, 1),
                                "heldout_y" : item["heldout_y"].reshape(-1, 1),
                                "heldout_predictions" : self.predictions[key].detach().cpu().numpy()}
            
        predictor_results_save_path = os.path.join(save_dir, f"lstm_{self.data_type}_results.pkl")
        predictor_results = {"window_length" : self.window_length,
                             "prediction_step" : 1,
                             "predictor_train_ratio" : self.train_ratio,
                             "fit_train_ratio" : self.fit_train_ratio,
                             "embedding_dim" : self.embedding_dim,
                             "hidden_dim" : self.hidden_dim,
                             "num_layers" : self.num_layers,
                             "evaluation_mode" : "rolling_one_step",
                             "uses_lagged_targets" : True,
                             "normalization_scope" : "pooled_inner_training_prefixes",
                             "validation_strategy" : "per_sequence_chronological_tail",
                             "validation_aggregation" : "mean_per_sequence_mse",
                             "best_epoch" : self.best_epoch,
                             "best_validation_loss" : self.best_validation_loss,
                             "average_mse" : self.average_mse,
                             "average_mae" : self.average_mae}

        save_data(data_record_save_path, data_record)
        save_data(predictor_results_save_path, predictor_results)
    

class LSTMPredictorDataset(Dataset):
    def __init__(self, X, Y):
        """
        X: (N, T, D)
        Y: (N, 1)
        """
        self.X = torch.from_numpy(X).to(torch.float32)
        self.Y = torch.from_numpy(Y).to(torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    

def _shuffle_in_unison(x, y, seed=None):
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if seed is not None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(x))
    else:
        idx = np.random.permutation(len(x))

    return x[idx], y[idx]


def _inner_split_index(sequence_length, window_length, fit_train_ratio):
    """Return the raw-time boundary for a per-sequence fitting split."""

    if not isinstance(sequence_length, (int, np.integer)) or sequence_length <= 0:
        raise ValueError("sequence length must be a positive integer")
    if not isinstance(window_length, (int, np.integer)) or window_length <= 0:
        raise ValueError("window_length must be a positive integer")
    if not np.isscalar(fit_train_ratio) or not np.isfinite(fit_train_ratio):
        raise ValueError("fit_train_ratio must be a finite scalar in (0, 1)")
    if not 0 < fit_train_ratio < 1:
        raise ValueError("fit_train_ratio must be in (0, 1)")

    inner_train_end = int(sequence_length * fit_train_ratio)
    if inner_train_end <= window_length:
        raise ValueError(
            "inner training portion must contain more than window_length observations"
        )
    if inner_train_end >= sequence_length:
        raise ValueError("inner validation portion must contain at least one observation")

    return inner_train_end


def _mean_per_sequence_mse(squared_errors, sequence_lengths):
    """Average per-example squared errors within, then across, sequences."""

    squared_errors = np.asarray(squared_errors, dtype=np.float64)
    if squared_errors.ndim != 1:
        raise ValueError("squared_errors must be one-dimensional")

    sequence_lengths = tuple(sequence_lengths)
    if not sequence_lengths:
        raise ValueError("sequence_lengths must contain at least one sequence")
    if any(not isinstance(length, (int, np.integer)) or length <= 0 for length in sequence_lengths):
        raise ValueError("every validation sequence must contain at least one example")
    if sum(sequence_lengths) != len(squared_errors):
        raise ValueError("sequence_lengths must account for every squared error")

    sequence_mse = []
    start = 0
    for length in sequence_lengths:
        end = start + length
        sequence_mse.append(squared_errors[start:end].mean())
        start = end

    loss = float(np.mean(sequence_mse))
    if not np.isfinite(loss):
        raise ValueError("validation loss must be finite")
    return loss


def _split_before(arr, split_ratio):

    length = arr.shape[0]
    split_idx = int(length*split_ratio)

    return arr[:split_idx], arr[split_idx:]


def _make_sequence_prediction_data(x, y, k):
    """
    x: np.ndarray of shape (T, D)
    y: np.ndarray of shape (T, 1) or (T,)
    k: window length

    Returns:
        X_seq: (T-k, k, D+1), containing paired past covariates and targets
        Y_seq: (T-k, 1)

    Sample j predicts y[j+k] from x[j:j+k] and y[j:j+k]. The target being
    predicted is therefore never included in its own input window.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if not isinstance(k, int) or k <= 0:
        raise ValueError("window length must be a positive integer")
    if x.ndim != 2:
        raise ValueError("x must have shape (T, D)")

    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError("y must have shape (T,) or (T, 1)")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of observations")

    T, D = x.shape
    if T <= k:
        print(
            "Warning: sequence length ({}) does not exceed window length ({}); "
            "returning empty sequence data.".format(T, k)
        )
        input_dtype = np.result_type(x.dtype, y.dtype)
        return (
            np.empty((0, k, D + 1), dtype=input_dtype),
            np.empty((0, 1), dtype=y.dtype),
        )

    inputs = np.concatenate([x, y], axis=-1)
    X_seq = np.stack([inputs[j:j+k] for j in range(T - k)], axis=0)
    Y_seq = y[k:]

    return X_seq, Y_seq


def _make_heldout_sequence_prediction_data(
    train_x,
    heldout_x,
    train_y,
    heldout_y,
    k,
):
    """Build rolling one-step held-out examples using only previously observed y."""

    train_x = np.asarray(train_x)
    heldout_x = np.asarray(heldout_x)
    train_y = np.asarray(train_y)
    heldout_y = np.asarray(heldout_y)

    if len(train_x) != len(train_y):
        raise ValueError("train_x and train_y must have the same length")
    if len(heldout_x) != len(heldout_y):
        raise ValueError("heldout_x and heldout_y must have the same length")
    if len(train_y) < k:
        raise ValueError("training history must contain at least window_length observations")
    if len(heldout_y) == 0:
        raise ValueError("heldout data must contain at least one observation")

    context_x = np.concatenate([train_x[-k:], heldout_x], axis=0)
    context_y = np.concatenate([train_y[-k:], heldout_y], axis=0)

    return _make_sequence_prediction_data(context_x, context_y, k)
