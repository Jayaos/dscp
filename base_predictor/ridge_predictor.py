import os

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from base_predictor.data import BasePredictorData
from utils.utils import save_data


class RidgeRegressionPredictor:
    """
    Ridge point predictor for simulation-style datasets where y_t = f(x_t).
    """

    def __init__(self, data: BasePredictorData, train_ratio, min_alpha, max_alpha, num_alphas):
        self.data = data.data
        self.data_type = data.data_type
        self.train_ratio = train_ratio
        self.alphas = np.linspace(min_alpha, max_alpha, num_alphas)
        self.data_split = {}
        self.models = {}
        self.predictions = {}
        self.selected_alphas = {}

        self._process_data()

    def _process_data(self):
        print("{} time series identified".format(len(self.data)))

        for key, item in self.data.items():
            x = np.asarray(item["x"], dtype=np.float32)
            y = np.asarray(item["y"], dtype=np.float32)

            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if y.ndim != 1:
                y = y.reshape(-1)
            if len(x) != len(y):
                raise ValueError(f"x and y length mismatch for {key}: {len(x)} != {len(y)}")

            split_idx = int(np.floor(len(y) * self.train_ratio))
            if split_idx <= 0 or split_idx >= len(y):
                raise ValueError(
                    f"train_ratio={self.train_ratio} leaves an empty split for {key} with length {len(y)}."
                )

            self.data_split[key] = {
                "train_x": x[:split_idx],
                "heldout_x": x[split_idx:],
                "train_y": y[:split_idx],
                "heldout_y": y[split_idx:],
            }

    def fit_predict(self):
        mse_list = []
        mae_list = []

        for key, item in tqdm(self.data_split.items()):
            print("fitting ridge regression model on {}".format(key))
            model = RidgeCV(alphas=self.alphas)
            model.fit(item["train_x"], item["train_y"])

            print("making predictions on heldout set...")
            predictions = model.predict(item["heldout_x"]).astype(np.float32)
            mse_heldout = mean_squared_error(item["heldout_y"], predictions)
            mae_heldout = mean_absolute_error(item["heldout_y"], predictions)

            mse_list.append(mse_heldout)
            mae_list.append(mae_heldout)
            self.predictions[key] = predictions
            self.models[key] = model
            self.selected_alphas[key] = float(model.alpha_)

        self.average_mse = float(np.mean(mse_list))
        self.average_mae = float(np.mean(mae_list))
        print("average MSE over {} sequences : {}".format(len(self.data), self.average_mse))
        print("average MAE over {} sequences : {}".format(len(self.data), self.average_mae))

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"ridge_{self.data_type}_data.pkl")
        data_record = {}

        for key, item in tqdm(self.data_split.items()):
            data_record[key] = {
                "train_x": item["train_x"],
                "heldout_x": item["heldout_x"],
                "train_y": item["train_y"],
                "heldout_y": item["heldout_y"],
                "heldout_predictions": self.predictions[key],
            }

        predictor_results_save_path = os.path.join(save_dir, f"ridge_{self.data_type}_results.pkl")
        predictor_results = {
            "train_ratio": self.train_ratio,
            "alphas": self.alphas,
            "selected_alphas": self.selected_alphas,
            "average_mse": self.average_mse,
            "average_mae": self.average_mae,
        }

        save_data(save_path, data_record)
        save_data(predictor_results_save_path, predictor_results)
