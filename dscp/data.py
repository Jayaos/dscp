import torch
import numpy as np
from torch.utils.data import Dataset
from utils.forecast_data import canonicalize_forecast_data
from utils.utils import (
    chronological_split_fixed_calibration_test,
    chronological_split_fixed_test,
    to_strided_feature,
    to_strided_residual,
)
from utils.utils import build_hopcpt_context_features
from utils.utils import normalize_array_with_params, compute_mean_std


class ConformalPredictionData:
    """
    Data class to load data for conformal prediction
    """

    def __init__(self, data):
        super(ConformalPredictionData, self).__init__()
        # Saved scalar forecasts may mix vectors and columns. Align both before
        # normalization or subtraction, including direct API and tuning calls.
        self.data = canonicalize_forecast_data(data)
        self.dataset = dict()

    def prepare_quantile_regression_datasets(self, 
                                             past_window, 
                                             prediction_steps, 
                                             train_ratio, 
                                             valid_ratio, 
                                             normalize=False,
                                             *,
                                             calibration_ratio=None,
                                             test_ratio=None,
                                             model_selection_valid_ratio=None):
        """
        Prepare sequence examples that predict future residuals from past context.

        Calls without optional split arguments retain the legacy chronological
        train | validation | test split. Providing calibration_ratio/test_ratio
        adds a dedicated calibration partition. Providing only
        model_selection_valid_ratio splits the nominal training prefix into
        model-fit and checkpoint-validation partitions and exposes the nominal
        validation partition as tuning_evaluation_dataset without exposing a
        test dataset. Providing all three optional ratios creates the nested
        Local-CP tuning split: the nominal validation partition is calibration,
        the nominal calibration partition is tuning evaluation, and the final
        test partition remains reserved and is not exposed as a dataset.
        Calibration and nested tuning splits are currently restricted to
        one-step prediction.

        """

        use_calibration_split = calibration_ratio is not None or test_ratio is not None
        use_model_selection_split = model_selection_valid_ratio is not None
        use_nested_local_split = use_calibration_split and use_model_selection_split

        if use_calibration_split:
            if calibration_ratio is None or test_ratio is None:
                raise ValueError(
                    "calibration_ratio and test_ratio must be provided together."
                )
            if prediction_steps != 1:
                raise ValueError(
                    "Local-CP calibration/tuning splits currently support "
                    "prediction_steps=1 only."
                )
            ratios = np.asarray(
                [train_ratio, valid_ratio, calibration_ratio, test_ratio],
                dtype=float,
            )
            if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0):
                raise ValueError(
                    "train, validation, calibration, and test ratios must all be "
                    "finite and strictly positive."
                )
            if not np.isclose(ratios.sum(), 1.0, rtol=0.0, atol=1e-8):
                raise ValueError(
                    "train_ratio + valid_ratio + calibration_ratio + test_ratio "
                    f"must equal 1; got {ratios.sum():.12g}."
                )

        if use_model_selection_split:
            if prediction_steps != 1:
                raise ValueError(
                    "The nested tuning split currently supports prediction_steps=1 only."
                )
            outer_ratios = np.asarray([train_ratio, valid_ratio], dtype=float)
            if (
                not np.all(np.isfinite(outer_ratios))
                or np.any(outer_ratios <= 0)
                or outer_ratios.sum() >= 1
            ):
                raise ValueError(
                    "train_ratio and valid_ratio must be finite, strictly positive, "
                    "and sum to less than 1 for nested tuning."
                )
            model_selection_valid_ratio = float(model_selection_valid_ratio)
            if (
                not np.isfinite(model_selection_valid_ratio)
                or not 0 < model_selection_valid_ratio < 1
            ):
                raise ValueError(
                    "model_selection_valid_ratio must be finite and strictly between 0 and 1."
                )

        for key, item in self.data.items():
            raw_heldout_y = np.asarray(item["heldout_y"])
            raw_heldout_predictions = np.asarray(item["heldout_predictions"])
            raw_heldout_residuals = (raw_heldout_y - raw_heldout_predictions).flatten()

            heldout_size = len(raw_heldout_y)
            if use_nested_local_split:
                raw_boundaries = heldout_size * np.cumsum(ratios[:3])
                nominal_train_size, outer_valid_end, outer_calibration_end = np.floor(
                    np.nextafter(raw_boundaries, np.inf)
                ).astype(int).tolist()
                calibration_size = outer_valid_end - nominal_train_size
                tuning_evaluation_size = outer_calibration_end - outer_valid_end
                test_size = heldout_size - outer_calibration_end
                train_size = int(
                    np.floor(
                        np.nextafter(
                            nominal_train_size * (1 - model_selection_valid_ratio),
                            np.inf,
                        )
                    )
                )
                model_selection_valid_size = nominal_train_size - train_size
                if min(
                    train_size,
                    model_selection_valid_size,
                    calibration_size,
                    tuning_evaluation_size,
                    test_size,
                ) <= 0:
                    raise ValueError(
                        f"Empty nested Local-CP tuning split for {key!r}: "
                        f"heldout_size={heldout_size}, fit={train_size}, "
                        f"checkpoint_valid={model_selection_valid_size}, "
                        f"calibration={calibration_size}, "
                        f"tuning={tuning_evaluation_size}, test={test_size}."
                    )
            elif use_model_selection_split:
                nominal_train_size = int(np.floor(heldout_size * train_ratio))
                tuning_size = int(np.ceil(heldout_size * valid_ratio))
                test_size = heldout_size - nominal_train_size - tuning_size
                train_size = int(
                    np.floor(
                        np.nextafter(
                            nominal_train_size * (1 - model_selection_valid_ratio),
                            np.inf,
                        )
                    )
                )
                model_selection_valid_size = nominal_train_size - train_size
                if min(
                    train_size,
                    model_selection_valid_size,
                    tuning_size,
                    test_size,
                ) <= 0:
                    raise ValueError(
                        f"Empty nested tuning split for {key!r}: "
                        f"heldout_size={heldout_size}, fit={train_size}, "
                        f"checkpoint_valid={model_selection_valid_size}, "
                        f"tuning={tuning_size}, "
                        f"test={test_size}."
                    )
            elif use_calibration_split:
                raw_boundaries = heldout_size * np.cumsum(ratios[:3])
                # Move each floating-point product one representable value
                # upward before floor so exact conceptual boundaries such as
                # 20 * (0.7 + 0.1) are not rounded from 16 down to 15.
                train_end, valid_end, calibration_end = np.floor(
                    np.nextafter(raw_boundaries, np.inf)
                ).astype(int).tolist()
                train_size = train_end
                valid_size = valid_end - train_end
                calibration_size = calibration_end - valid_end
                test_size = heldout_size - calibration_end
                if min(train_size, valid_size, calibration_size, test_size) <= 0:
                    raise ValueError(
                        f"Empty split for {key!r}: heldout_size={heldout_size}, "
                        f"train={train_size}, valid={valid_size}, "
                        f"calibration={calibration_size}, test={test_size}."
                    )
            else:
                train_size = int(np.floor(heldout_size*train_ratio))
                valid_size = int(np.ceil(heldout_size*valid_ratio))
                test_size = heldout_size - (train_size+valid_size)

            if normalize:
                # normalize variables that will be used for sequence model prediction
                # NOTE: normalize should be conducted only with training set
                
                train_x_mu, train_x_std = compute_mean_std(item["heldout_x"][:train_size])
                heldout_x = normalize_array_with_params(item["heldout_x"], train_x_mu, train_x_std)

                train_y_mu, train_y_std = compute_mean_std(raw_heldout_y[:train_size])
                heldout_y = normalize_array_with_params(raw_heldout_y, train_y_mu, train_y_std)
                heldout_predictions = normalize_array_with_params(raw_heldout_predictions, train_y_mu, train_y_std)

                # Residuals are computed after normalizing y and yhat.  For
                # metric code that converts residual intervals back to raw y
                # units, this is equivalent to residual_mu=0, residual_std=y_std.
                train_residuals_mu = np.zeros_like(train_y_mu)
                train_residuals_std = train_y_std + 1e-8

            else:
                heldout_x = item["heldout_x"]
                heldout_y = raw_heldout_y
                heldout_predictions = raw_heldout_predictions

            heldout_residuals = (heldout_y - heldout_predictions).flatten()
            self.data[key].update({"heldout_residuals" : heldout_residuals,
                                   "raw_heldout_residuals" : raw_heldout_residuals})

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
            # Keep target_y/target_predictions in raw units for reporting.
            _, target_y = to_strided_residual(raw_heldout_y, 
                                              past_window, 
                                              prediction_steps)
            _, target_predictions = to_strided_residual(raw_heldout_predictions,
                                                        past_window, 
                                                        prediction_steps)
            _, target_y_normalized = to_strided_residual(heldout_y,
                                                         past_window,
                                                         prediction_steps)
            _, target_predictions_normalized = to_strided_residual(heldout_predictions,
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
                                        "raw_heldout_residuals" : raw_heldout_residuals,
                                        "train_residuals_mu" : train_residuals_mu,
                                        "train_residuals_std" : train_residuals_std,
                                        "heldout_y_normalized" : heldout_y,
                                        "heldout_predictions_normalized" : heldout_predictions,
                                        "target_y_normalized" : target_y_normalized,
                                        "target_predictions_normalized" : target_predictions_normalized,
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
                                        "heldout_residuals" : heldout_residuals,
                                        "raw_heldout_residuals" : raw_heldout_residuals})
            
            arrays = [strided_x,
                      strided_residual,
                      strided_y,
                      target_x,
                      target_residual,
                      target_y,
                      target_predictions]
            if use_nested_local_split:
                train_split, valid_split, calibration_split, heldout_split = \
                    chronological_split_fixed_calibration_test(
                        arrays,
                        model_selection_valid_size,
                        calibration_size,
                        tuning_evaluation_size + test_size,
                    )
                tuning_evaluation_split = tuple(
                    array[:tuning_evaluation_size] for array in heldout_split
                )
            elif use_calibration_split or use_model_selection_split:
                fourth_split_size = (
                    calibration_size if use_calibration_split else tuning_size
                )
                train_split, valid_split, fourth_split, test_split = \
                    chronological_split_fixed_calibration_test(
                        arrays,
                        (
                            model_selection_valid_size
                            if use_model_selection_split
                            else valid_size
                        ),
                        fourth_split_size,
                        test_size,
                    )
            else:
                train_split, valid_split, test_split = chronological_split_fixed_test(
                    arrays,
                    valid_size,
                    test_size,
                )

            datasets = {
                "train_dataset": QuantileRegressionDataset(
                    train_split[0],
                    train_split[1],
                    train_split[2],
                    train_split[3],
                    train_split[4],
                    train_split[5],
                    train_split[6],
                ),
            }
            valid_dataset = QuantileRegressionDataset(
                valid_split[0],
                valid_split[1],
                valid_split[2],
                valid_split[3],
                valid_split[4],
                valid_split[5],
                valid_split[6],
            )
            valid_dataset_key = (
                "model_selection_valid_dataset"
                if use_model_selection_split
                else "valid_dataset"
            )
            datasets[valid_dataset_key] = valid_dataset
            if use_nested_local_split:
                datasets["calibration_dataset"] = QuantileRegressionDataset(
                    calibration_split[0],
                    calibration_split[1],
                    calibration_split[2],
                    calibration_split[3],
                    calibration_split[4],
                    calibration_split[5],
                    calibration_split[6],
                )
                datasets["tuning_evaluation_dataset"] = QuantileRegressionDataset(
                    tuning_evaluation_split[0],
                    tuning_evaluation_split[1],
                    tuning_evaluation_split[2],
                    tuning_evaluation_split[3],
                    tuning_evaluation_split[4],
                    tuning_evaluation_split[5],
                    tuning_evaluation_split[6],
                )
            elif use_calibration_split:
                datasets["calibration_dataset"] = QuantileRegressionDataset(
                    fourth_split[0],
                    fourth_split[1],
                    fourth_split[2],
                    fourth_split[3],
                    fourth_split[4],
                    fourth_split[5],
                    fourth_split[6],
                )
            elif use_model_selection_split:
                datasets["tuning_evaluation_dataset"] = QuantileRegressionDataset(
                    fourth_split[0],
                    fourth_split[1],
                    fourth_split[2],
                    fourth_split[3],
                    fourth_split[4],
                    fourth_split[5],
                    fourth_split[6],
                )

            if not use_model_selection_split:
                datasets["test_dataset"] = QuantileRegressionDataset(
                    test_split[0],
                    test_split[1],
                    test_split[2],
                    test_split[3],
                    test_split[4],
                    test_split[5],
                    test_split[6],
                )
            self.dataset[key] = datasets
            if use_nested_local_split:
                self.data[key].update({
                    "nominal_train_size": nominal_train_size,
                    "train_size": train_size,
                    "model_selection_valid_size": model_selection_valid_size,
                    "calibration_size": calibration_size,
                    "tuning_evaluation_size": tuning_evaluation_size,
                    "test_size": test_size,
                })
            elif use_calibration_split:
                self.data[key].update({
                    "train_size": train_size,
                    "valid_size": valid_size,
                    "calibration_size": calibration_size,
                    "test_size": test_size,
                })
            elif use_model_selection_split:
                self.data[key].update({
                    "nominal_train_size": nominal_train_size,
                    "train_size": train_size,
                    "model_selection_valid_size": model_selection_valid_size,
                    "tuning_evaluation_size": tuning_size,
                    "test_size": test_size,
                })

    def prepare_hopcpt_datasets(self, 
                                prediction_steps, 
                                y_lags,
                                train_ratio, 
                                valid_ratio, 
                                normalize=False,
                                predict_absolute_residual=True,
                                conformal_absolute_residual=False):

        for key, item in self.data.items():
            raw_heldout_y = np.asarray(item["heldout_y"])
            raw_heldout_predictions = np.asarray(item["heldout_predictions"])
            raw_heldout_signed_residuals = (raw_heldout_y - raw_heldout_predictions).flatten()

            heldout_size = len(raw_heldout_y)
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

                train_y_mu, train_y_std = compute_mean_std(item["heldout_y"][:train_size])
                heldout_y = normalize_array_with_params(item["heldout_y"], train_y_mu, train_y_std)

                # use mu and std of y for predicted y since this is going to be used for query
                heldout_predictions = normalize_array_with_params(item["heldout_predictions"], train_y_mu, train_y_std)

            else:
                heldout_x = item["heldout_x"]
                heldout_y = item["heldout_y"]
                heldout_predictions = item["heldout_predictions"]

            # Residuals must be computed in the same scale as the contexts.
            # When normalize=True, heldout_y and heldout_predictions are both
            # normalized with train_y_mu/train_y_std, so these residuals are too.
            heldout_signed_residuals = (heldout_y - heldout_predictions).flatten()
            heldout_train_residuals = heldout_signed_residuals
            heldout_conformal_residuals = heldout_signed_residuals
            if predict_absolute_residual:
                heldout_train_residuals = np.abs(heldout_train_residuals)
            if conformal_absolute_residual:
                heldout_conformal_residuals = np.abs(heldout_conformal_residuals)

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
            heldout_target_y = np.asarray(heldout_y)[y_lags:]
            heldout_target_predictions = np.asarray(heldout_predictions)[y_lags:]
            raw_heldout_target_y = raw_heldout_y[y_lags:]
            raw_heldout_target_predictions = raw_heldout_predictions[y_lags:]

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
                                       "raw_heldout_signed_residuals" : raw_heldout_signed_residuals,
                                       "raw_heldout_target_y" : raw_heldout_target_y,
                                       "raw_heldout_target_predictions" : raw_heldout_target_predictions,
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
                                       "raw_heldout_signed_residuals" : raw_heldout_signed_residuals,
                                       "raw_heldout_target_y" : raw_heldout_target_y,
                                       "raw_heldout_target_predictions" : raw_heldout_target_predictions,
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
    y_lags = data["y_lags"]
    start_idx = train_size - y_lags
    end_idx = train_size + valid_size - y_lags
    target_y_key = "raw_heldout_target_y" if normalize else "heldout_target_y"
    target_predictions_key = "raw_heldout_target_predictions" if normalize else "heldout_target_predictions"

    valid_context_generator = prefix_gen(data["heldout_context"][:end_idx],
                                         start_idx,
                                         prediction_steps,
                                         max_memory=max_memory_size)
    valid_target_y_generator = prefix_gen(data[target_y_key][:end_idx],
                                          start_idx,
                                          prediction_steps,
                                          max_memory=max_memory_size)
    valid_residual_generator = prefix_gen(data["heldout_context_residuals"][:end_idx],
                                          start_idx,
                                          prediction_steps,
                                          max_memory=max_memory_size)
    valid_prediction_generator = prefix_gen(data[target_predictions_key][:end_idx],
                                            start_idx,
                                            prediction_steps,
                                            max_memory=max_memory_size)
    
    return (valid_context_generator, valid_target_y_generator, valid_residual_generator, valid_prediction_generator)


def initialize_test_dataloader(data, train_size, valid_size, prediction_steps, max_memory_size, normalize):
    """
    initialize dataloader, which is a generator
    """
    y_lags = data["y_lags"]
    start_idx = train_size + valid_size - y_lags
    target_y_key = "raw_heldout_target_y" if normalize else "heldout_target_y"
    target_predictions_key = "raw_heldout_target_predictions" if normalize else "heldout_target_predictions"
    
    test_context_generator = prefix_gen(data["heldout_context"],
                                        start_idx,
                                        prediction_steps,
                                        max_memory=max_memory_size)
    test_target_y_generator = prefix_gen(data[target_y_key],
                                         start_idx,
                                         prediction_steps,
                                         max_memory=max_memory_size)
    test_residual_generator = prefix_gen(data["heldout_context_residuals"],
                                         start_idx,
                                         prediction_steps,
                                         max_memory=max_memory_size)
    test_prediction_generator = prefix_gen(data[target_predictions_key],
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

