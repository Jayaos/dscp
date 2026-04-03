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
    Global LSTM predictor
    """

    def __init__(self, data: BasePredictorData, embedding_dim, hidden_dim, num_layers, train_ratio, window_length):
        super(LSTMPredictor, self).__init__()
        self.input_dim = next(iter(data.data.items()))[1]["x"].shape[-1]
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

        merged_train_x = []
        merged_train_y = []

        for key, item in self.data.items():
            x = item["x"]
            y = item["y"]
            train_x, heldout_x = _split_before(x, self.train_ratio)
            train_y, heldout_y = _split_before(y, self.train_ratio)

            merged_train_x.append(train_x)
            merged_train_y.append(train_y)

            self.data_processed[key] = {"train_x" : train_x,
                                        "heldout_x" : heldout_x,
                                        "train_y" : train_y,
                                        "heldout_y" : heldout_y
            }

        # normalized using global data
        merged_train_x = np.concatenate(merged_train_x, axis=0)
        merged_train_y = np.concatenate(merged_train_y, axis=0)
        _, (train_x_mu, train_x_std) = normalize_array(merged_train_x)
        _, (train_y_mu, train_y_std) = normalize_array(merged_train_y)

        for key, item in self.data_processed.items():
            train_x = item["train_x"]
            heldout_x = item["heldout_x"]
            train_y = item["train_y"]
            heldout_y = item["heldout_y"]

            normalized_train_x = normalize_array_with_params(train_x, train_x_mu, train_x_std)
            normalized_heldout_x = normalize_array_with_params(heldout_x, train_x_mu, train_x_std)
            normalized_train_y = normalize_array_with_params(train_y, train_y_mu, train_y_std)
            normalized_heldout_y = normalize_array_with_params(heldout_y, train_y_mu, train_y_std)

            train_x_seq, train_y_seq = _make_sequence_prediction_data(normalized_train_x, 
                                                                      normalized_train_y, 
                                                                      window_length)

            self.data_processed[key].update({"normalized_train_x" : normalized_train_x,
                                             "normalized_heldout_x" : normalized_heldout_x,
                                             "normalized_train_y" : normalized_train_y,
                                             "normalized_heldout_y" : normalized_heldout_y,
                                             "train_x_seq" : train_x_seq,
                                             "train_y_seq" : train_y_seq,
                                             "train_x_mu" : train_x_mu,
                                             "train_x_std" : train_x_std,
                                             "train_y_mu" : train_y_mu, 
                                             "train_y_std" : train_y_std})
        
    def fit_predict(self, train_ratio, batch_size, learning_rate, max_epoch, early_stop, seed=2026, device=0):
        """
        fit LSTM predictor on data and make point prediction
        """
        # merge data first to train a global model
        merged_train_x_seq = []
        merged_train_y_seq = []

        for key, item in tqdm(self.data_processed.items()):
            merged_train_x_seq.append(item["train_x_seq"])
            merged_train_y_seq.append(item["train_y_seq"])

        merged_train_x_seq = np.vstack(merged_train_x_seq)
        merged_train_y_seq = np.vstack(merged_train_y_seq)
        merged_train_x_seq = _shuffle_array(merged_train_x_seq, seed)
        merged_train_y_seq = _shuffle_array(merged_train_y_seq, seed)
        train_x_seq, valid_x_seq = _split_before(merged_train_x_seq, train_ratio)
        train_y_seq, valid_y_seq = _split_before(merged_train_y_seq, train_ratio)

        train_dataset = LSTMPredictorDataset(train_x_seq, train_y_seq)
        valid_dataset = LSTMPredictorDataset(valid_x_seq, valid_y_seq)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        optim = torch.optim.Adam(self.lstm.parameters(), lr=learning_rate)

        best_loss = torch.inf
        self.lstm.to(device)
        for e in tqdm(range(max_epoch)):
            # training
            self.lstm.train()
            loss_sum = 0.
            for x_batch, y_batch in tqdm(train_dataloader):

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                out = self.lstm(x_batch) # (batch_size, window_length, 1)
                preds = out[:,-1,:] # (batch_size, 1)
                loss = torch.nn.functional.mse_loss(preds, y_batch)
                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_sum += loss.item()

            epoch_train_loss = loss_sum/len(train_dataloader)
            print("train loss at epoch {} : {}".format(e+1, epoch_train_loss))

            self.lstm.eval()
            loss_sum = 0.
            for x_batch, y_batch in tqdm(valid_dataloader):

                with torch.no_grad():
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    out = self.lstm(x_batch) # (batch_size, window_length, 1)
                    preds = out[:,-1,:] # (batch_size, 1)
                    loss = torch.nn.functional.mse_loss(preds, y_batch)
                    loss_sum += loss.item()

            epoch_valid_loss = loss_sum/len(valid_dataloader)
            print("valid loss at epoch {} : {}".format(e+1, epoch_valid_loss))
            
            if epoch_valid_loss < best_loss:
                best_loss = epoch_valid_loss
                best_epoch = e+1
                best_model = copy.deepcopy(self.lstm.state_dict())

            if (e+1-best_epoch) >= early_stop:
                # if the loss did not decrease for (early_stop) epoch in a row, stop training
                break
        
        print("best model at epoch {}".format(best_epoch))
        print("making predictions on the heldout data")
        mse_list = []
        mae_list = []
        self.lstm.load_state_dict(best_model)
        self.lstm.eval()
        for key, item in tqdm(self.data_processed.items()):

            normalized_heldout_x = item["normalized_heldout_x"]
            normalized_heldout_y = item["normalized_heldout_y"]
            y_mu = item["train_y_mu"]
            y_std = item["train_y_std"]

            heldout_x_seq, heldout_y_seq = _make_sequence_prediction_data(normalized_heldout_x, 
                                                                          normalized_heldout_y, 
                                                                          self.window_length)
            heldout_dataset = LSTMPredictorDataset(heldout_x_seq, heldout_y_seq)
            heldout_dataloader = DataLoader(heldout_dataset, batch_size=batch_size, shuffle=False)

            predictions = []
            for x_batch, y_batch in tqdm(heldout_dataloader):

                with torch.no_grad():
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    out = self.lstm(x_batch)
                    preds = out[:,-1,:]
                    denormalized_y_batch = denormalize_array(y_batch, y_mu, y_std)
                    denormalized_preds = denormalize_array(preds, y_mu, y_std)
                    predictions.append(denormalized_preds)
    
                    mse = torch.nn.functional.mse_loss(denormalized_preds, denormalized_y_batch)
                    mse_list.append(mse.item())
                    mae = torch.nn.functional.l1_loss(denormalized_preds, denormalized_y_batch)
                    mae_list.append(mae.item())

            self.predictions[key] = torch.vstack(predictions)

        self.average_mse = np.mean(mse_list)
        self.average_mae = np.mean(mae_list)
        print("average MSE : {}".format(self.average_mse))
        print("average MAE : {}".format(self.average_mae))

    def save(self, save_dir):

        data_record_save_path = os.path.join(save_dir + "lstm_" + str(self.data_type) + "_data.pkl")
        data_record = dict()

        for key, item in tqdm(self.data_processed.items()):

            data_record[key] = {"normalized_train_x" : item["normalized_train_x"],
                                "train_x" : item["train_x"],
                                "normalized_heldout_x" : item["normalized_heldout_x"],
                                "heldout_x" : item["heldout_x"],
                                "normalized_train_y" : item["normalized_train_y"],
                                "train_y" : item["train_y"],
                                "normalized_heldout_y" : item["normalized_heldout_y"],
                                "heldout_y" : item["heldout_y"],
                                "train_x_seq" : item["train_x_seq"],
                                "train_y_seq" : item["train_y_seq"],
                                "train_x_mu" : item["train_x_mu"],
                                "train_x_std" : item["train_x_std"],
                                "train_y_mu" : item["train_y_mu"], 
                                "train_y_std" : item["train_y_std"],
                                "heldout_predictions" : self.predictions[key]}
            
        predictor_results_save_path = os.path.join(save_dir + "lstm_" + str(self.data_type) + "_results.pkl")
        predictor_results = {"window_length" : self.window_length,
                             "embedding_dim" : self.embedding_dim,
                             "hidden_dim" : self.hidden_dim,
                             "num_layers" : self.num_layers,
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
    

def _shuffle_array(arr, seed=None):

    if seed is not None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(arr))
    else:
        idx = np.random.permutation(len(arr))

    return arr[idx]


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
        X_seq: (T-k, k, D)
        Y_seq: (T-k, 1)
    """

    if y.ndim == 1:
        y = y[:, None]

    T, D = x.shape
    # Build X_seq
    # j ranges 0..T-k, window is x[j:j+k]
    X_seq = np.stack([x[j:j+k] for j in range(T - k + 1)], axis=0)  # (T-k+1, k, D)

    # Build Y_seq: y[k-1], y[k], ..., y[T-1]
    Y_seq = y[k-1:]  # (T-k+1, 1)

    return X_seq, Y_seq