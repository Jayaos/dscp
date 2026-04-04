from base_predictor.data import BasePredictorData
from utils.utils import save_data
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from darts import TimeSeries, concatenate
from darts.models import LightGBMModel
from darts.metrics import mse, mae


class LightGBMPredictor:
    # TODO: prediction plot function
    def __init__(self, data: BasePredictorData, train_ratio):
        super(LightGBMPredictor, self).__init__()
        self.data = data.data
        self.data_type = data.data_type
        self.train_ratio = train_ratio
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

            print(train_x.components)

            self.data_darts_format[key] = {"train_x" : train_x,
                                      "heldout_x" : heldout_x,
                                      "train_y" : train_y,
                                      "heldout_y" : heldout_y}
            
    def fit_predict(self, past_window, prediction_step):
        """
        fit lightGBM predictor on data and make point prediction
        """
        mse_list = []
        mae_list = []

        for key, item in tqdm(self.data_darts_format.items()):
            print("fitting lightGBM model on {}".format(key))
            model = LightGBMModel(lags=past_window,
                                  lags_past_covariates=past_window,
                                  output_chunk_length=prediction_step,
                                  verbose=-1)
            
            model.fit(item["train_y"], past_covariates=item["train_x"])

            print("making predictions on heldout set...")
            all_x = concatenate([item["train_x"], item["heldout_x"]], axis="time")
            past_x = all_x[len(item["train_x"])-past_window:]
            print(past_x.components)
            
            predictions = model.predict(len(item["heldout_y"]), 
                                        past_covariates=past_x,
                                        show_warnings=False) 
            # show_warinings=False to turn of 
            mse_heldout = mse(item["heldout_y"], predictions)
            mae_heldout = mae(item["heldout_y"], predictions)

            mse_list.append(mse_heldout)
            mae_list.append(mae_heldout)

            self.predictions[key] = predictions
            self.models[key] = model

        print("average MSE over {} sequences : {}".format(len(self.data), np.mean(mse_list)))
        print("average MAE over {} sequences : {}".format(len(self.data), np.mean(mae_list)))

    def save(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"lightgbm_{self.data_type}_data.pkl")
        data_record = dict()
        
        for key, item in tqdm(self.data_darts_format.items()):

            data_record[key] = {"train_x" : item["train_x"].values(),
                                "heldout_x" : item["heldout_x"].values(),
                                "train_y" : item["train_y"].values(),
                                "heldout_y" : item["heldout_y"].values(),
                                "heldout_predictions" : self.predictions[key].values()}
            
        save_data(save_path, data_record)
    
