
from base_predictor.data import BasePredictorData
from utils.utils import save_data
from chronos import Chronos2Pipeline
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os


class ChronosPredictor:
    """
    Global Chronos-2 predictor
    """

    def __init__(self, data: BasePredictorData, device):
        super(ChronosPredictor, self).__init__()
        self.input_dim = next(iter(data.data.items()))[1]["x"].shape[-1]
        self.chronos2 = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            dtype=torch.float32,
            device_map=device)
        self.data = data.data
        self.data_type = data.data_type
        self.predictions = dict()

    def predict(self, window_length, prediction_length):

        self.window_length = window_length
        self.prediction_length = prediction_length

        print("{} time series identified".format(len(self.data)))

        mse_list = []
        mae_list = []
        for key, data in self.data.items():

            x = data["x"]
            y = data["y"]
            iter_num = int(np.ceil((y.shape[0] - window_length) / prediction_length))
            
            predictions = []
            for i in tqdm(range(iter_num)):

                context_x = x[i*prediction_length:i*prediction_length+window_length,:]
                future_x = x[i*prediction_length+window_length:i*prediction_length+window_length+prediction_length,:]
                context_y = y[i*prediction_length:i*prediction_length+window_length]
                future_y = y[i*prediction_length+window_length:i*prediction_length+window_length+prediction_length]

                context_df, future_df = _build_context_future_df(context_x, future_x, context_y)
                pred_df = self.chronos2.predict_df(context_df,
                                                   future_df=future_df,
                                                   prediction_length=future_x.shape[0],  # Number of steps to forecast
                                                   quantile_levels=[0.1, 0.5, 0.9],  # Quantile for probabilistic forecast
                                                   id_column="id",  # Column identifying different time series
                                                   timestamp_column="timestamp",  # Column with datetime information
                                                   target="target",  # Column(s) with time series values to predict
                )

                predictions.extend(pred_df["predictions"].to_list())

                mse = torch.nn.functional.mse_loss(torch.from_numpy(pred_df["predictions"].to_numpy()), 
                                                   torch.from_numpy(future_y).to(torch.float32))
                mse_list.append(mse.item())
                mae = torch.nn.functional.l1_loss(torch.from_numpy(pred_df["predictions"].to_numpy()), 
                                                   torch.from_numpy(future_y).to(torch.float32))
                mae_list.append(mae.item())
            
            self.predictions[key] = predictions

        self.average_mse = np.mean(mse_list)
        self.average_mae = np.mean(mae_list)
        print("average MSE : {}".format(self.average_mse))
        print("average MAE : {}".format(self.average_mae))

    def save(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)
        data_record_save_path = os.path.join(save_dir, f"chronos_{self.data_type}_data.pkl")
        data_record = dict()

        for key, data in self.data.items():
            data_record[key] = {"heldout_x" : data["x"][self.window_length:,:],
                                "heldout_y" : data["y"][self.window_length:],
                                "heldout_predictions" : self.predictions[key]}

        predictor_results_save_path = os.path.join(save_dir, f"chronos_{self.data_type}_results.pkl")
        predictor_results = {"window_length" : self.window_length,
                             "prediction_length" : self.prediction_length,
                             "average_mse" : self.average_mse,
                             "average_mae" : self.average_mae}

        save_data(data_record_save_path, data_record)
        save_data(predictor_results_save_path, predictor_results)


def _build_context_future_df(context_x, future_x, context_y):
    
    t = np.arange(context_x.shape[0])
    t_future = np.arange(context_x.shape[0], context_x.shape[0]+future_x.shape[0])

    context_df = pd.DataFrame({
        "id": "ts_0",
        "timestamp": t,
        "target": context_y,
    })

    feature_dim = context_x.shape[1]
    for j in range(feature_dim):
        context_df[f"feat_{j}"] = context_x[:, j]

    future_df = pd.DataFrame({
        "id": "ts_0",
        "timestamp": t_future,
    })

    for j in range(feature_dim):
        future_df[f"feat_{j}"] = future_x[:, j]

    return context_df, future_df

