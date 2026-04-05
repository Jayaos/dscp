from base_predictor.data import BasePredictorData
from utils.utils import save_data
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from darts import TimeSeries, concatenate
from darts.models import LinearRegressionModel
from darts.metrics import mse, mae


class LinearRegressionPredictor:
    # TODO: prediction plot function
    def __init__(self, data: BasePredictorData, train_ratio, past_window, prediction_step):
        super(LinearRegressionPredictor, self).__init__()
        self.data = data.data
        self.data_type = data.data_type
        self.train_ratio = train_ratio
        self.past_window = past_window
        self.prediction_step = prediction_step
        self.data_darts_format = dict()
        self.models = dict()
        self.predictions = dict()

        self._process_data()

    def _process_data(self):
        print("{} time series identified".format(len(self.data)))

        for key, item in self.data.items():
            x = item["x"]
            y = item["y"]
            idx = pd.RangeIndex(start=0, stop=len(y), step=1)

            x_ts = TimeSeries.from_times_and_values(idx, x)
            y_ts = TimeSeries.from_times_and_values(idx, y)
            train_x, heldout_x = x_ts.split_before(self.train_ratio)
            train_y, heldout_y = y_ts.split_before(self.train_ratio)

            self.data_darts_format[key] = {"train_x" : train_x,
                                           "heldout_x" : heldout_x,
                                           "train_y" : train_y,
                                           "heldout_y" : heldout_y}
    
    def fit_predict(self):
        """
        fit linear regression predictor on data and make point prediction
        """
        mse_list = []
        mae_list = []

        for key, item in tqdm(self.data_darts_format.items()):
            print("fitting linear regression model on {}".format(key))
            model = LinearRegressionModel(lags=self.past_window, 
                                          # if its None, not utilizing past y as features
                                          lags_past_covariates=self.past_window,
                                          output_chunk_length=self.prediction_step)
            
            model.fit(item["train_y"], past_covariates=item["train_x"])

            print("making predictions on heldout set...")
            all_x = concatenate([item["train_x"], item["heldout_x"]], axis="time")
            past_x = all_x[len(item["train_x"])-self.past_window:]
            predictions = model.predict(len(item["heldout_y"]), past_covariates=past_x)
            mse_heldout = mse(item["heldout_y"], predictions)
            mae_heldout = mae(item["heldout_y"], predictions)

            mse_list.append(mse_heldout)
            mae_list.append(mae_heldout)
            self.predictions[key] = predictions
            self.models[key] = model
        
        self.average_mse = np.mean(mse_list)
        self.average_mae = np.mean(mae_list)
        print("average MSE over {} sequences : {}".format(len(self.data), self.average_mse))
        print("average MAE over {} sequences : {}".format(len(self.data), self.average_mae))

    def save(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"lr_{self.data_type}_data.pkl")
        data_record = dict()
        
        for key, item in tqdm(self.data_darts_format.items()):

            data_record[key] = {"train_x" : item["train_x"].values(),
                                "heldout_x" : item["heldout_x"].values(),
                                "train_y" : item["train_y"].values(),
                                "heldout_y" : item["heldout_y"].values(),
                                "heldout_predictions" : self.predictions[key].values()}
            
        predictor_results_save_path = os.path.join(save_dir, f"lr_{self.data_type}_results.pkl")
        predictor_results = {"past_window" : self.past_window,
                             "prediction_step" : self.prediction_step,
                             "average_mse" : self.average_mse,
                             "average_mae" : self.average_mae}

        save_data(save_path, data_record)
        save_data(predictor_results_save_path, predictor_results)

