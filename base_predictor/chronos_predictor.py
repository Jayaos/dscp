
from base_predictor.data import BasePredictorData
from utils.utils import save_data
from chronos import Chronos2Pipeline
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from contextlib import nullcontext
from importlib.metadata import version

from base_predictor.chronos_features import (
    ChronosFeatureCapture, combine_feature_blocks, export_chronos_head,
)


class ChronosPredictor:
    """
    Global Chronos-2 predictor using historical targets and past-only covariates.
    """

    def __init__(self, data: BasePredictorData, device, model_revision=None):
        super(ChronosPredictor, self).__init__()
        self.input_dim = next(iter(data.data.items()))[1]["x"].shape[-1]
        self.chronos2 = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            **({"revision": model_revision} if model_revision else {}),
            dtype=torch.float32,
            device_map=device)
        self.data = data.data
        self.data_type = data.data_type
        self.predictions = dict()
        self.model_revision = model_revision

    def predict(self, window_length, prediction_length, save_ctssf=False):
        """Generate rolling block forecasts without forecast-time covariates.

        Every forecast uses only the targets and covariates strictly before its
        origin.  Covariates at the timestamps being predicted are intentionally
        not passed to Chronos.
        """

        if window_length < 1 or prediction_length < 1:
            raise ValueError("Window and prediction lengths must be positive.")
        self.window_length = window_length
        self.prediction_length = prediction_length
        self.ctssf_features = {} if save_ctssf else None
        if save_ctssf:
            self.chronos2.model.eval()
            if prediction_length > self.chronos2.model_prediction_length:
                raise ValueError("CT-SSF export does not support autoregressive long-horizon unrolling.")
            self.ctssf_head = export_chronos_head(self.chronos2.model, {
                "model_id": "amazon/chronos-2",
                "requested_revision": getattr(self, "model_revision", None),
                "resolved_revision": getattr(self.chronos2.model.config, "_commit_hash", None),
                "chronos_forecasting_version": version("chronos-forecasting"),
                "torch_version": str(torch.__version__),
                "window_length": int(window_length),
                "prediction_length": int(prediction_length),
                "uses_future_covariates": False,
            })
            capture = ChronosFeatureCapture(self.chronos2, self.ctssf_head)

        print("{} time series identified".format(len(self.data)))

        mse_list = []
        mae_list = []
        for key, data in self.data.items():

            x = data["x"]
            y = data["y"]
            if len(y) <= window_length:
                raise ValueError(f"Series {key!r} has no targets after the context window.")
            iter_num = int(np.ceil((y.shape[0] - window_length) / prediction_length))
            
            predictions = []
            series_targets = []
            feature_blocks = []
            for i in tqdm(range(iter_num)):

                context_x = x[i*prediction_length:i*prediction_length+window_length,:]
                context_y = y[i*prediction_length:i*prediction_length+window_length]
                future_y = y[i*prediction_length+window_length:i*prediction_length+window_length+prediction_length]

                context_df = _build_context_df(context_x, context_y)
                with (capture if save_ctssf else nullcontext()):
                    pred_df = self.chronos2.predict_df(
                        context_df,
                        prediction_length=future_y.shape[0],
                        quantile_levels=[0.5],
                        id_column="id",
                        timestamp_column="timestamp",
                        target="target",
                    )

                if save_ctssf:
                    feature_blocks.append(capture.export_block(
                        pred_df["predictions"].to_numpy(),
                        origin=i * prediction_length + window_length,
                        covariate_dim=x.shape[1],
                    ))

                predictions.extend(pred_df["predictions"].to_list())
                series_targets.extend(future_y.tolist())
            
            predictions_array = np.asarray(predictions, dtype=np.float32)
            targets_array = np.asarray(series_targets, dtype=np.float32)

            mse = torch.nn.functional.mse_loss(torch.from_numpy(predictions_array),
                                               torch.from_numpy(targets_array))
            mae = torch.nn.functional.l1_loss(torch.from_numpy(predictions_array),
                                              torch.from_numpy(targets_array))
            mse_list.append(mse.item())
            mae_list.append(mae.item())
            self.predictions[key] = predictions
            if save_ctssf:
                self.ctssf_features[key] = combine_feature_blocks(
                    feature_blocks, self.ctssf_head["head_sha256"],
                )

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
            if getattr(self, "ctssf_features", None) is not None:
                data_record[key]["ctssf"] = self.ctssf_features[key]

        predictor_results_save_path = os.path.join(save_dir, f"chronos_{self.data_type}_results.pkl")
        predictor_results = {"window_length" : self.window_length,
                             "prediction_length" : self.prediction_length,
                             "uses_future_covariates" : False,
                             "quantile_levels" : [0.5],
                             "average_mse" : self.average_mse,
                             "average_mae" : self.average_mae}

        save_data(data_record_save_path, data_record)
        save_data(predictor_results_save_path, predictor_results)
        if getattr(self, "ctssf_features", None) is not None:
            head_path = os.path.join(save_dir, f"chronos_{self.data_type}_ctssf_head.pt")
            torch.save(self.ctssf_head, head_path)
            print(f"saved CT-SSF features and median head: {head_path}")


def _build_context_df(context_x, context_y):
    
    t = np.arange(context_x.shape[0])

    context_df = pd.DataFrame({
        "id": "ts_0",
        "timestamp": t,
        "target": context_y,
    })

    feature_dim = context_x.shape[1]
    for j in range(feature_dim):
        context_df[f"feat_{j}"] = context_x[:, j]

    return context_df

