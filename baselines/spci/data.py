"""Chronological train/test preparation with optional validation inside training."""

from numbers import Integral

import numpy as np

from dscp.data import ConformalPredictionData, QuantileRegressionDataset
from utils.utils import (
    compute_mean_std,
    normalize_array_with_params,
    to_strided_feature,
    to_strided_residual,
)


def _fraction(value, name):
    try:
        fraction = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1.") from exc
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1.")
    return fraction


def _scalar_sequence(values, name):
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(f"{name} must have shape [T] or [T, 1].")
    return values


def prepare_spci_data(data, config, *, model_selection_valid_ratio=None):
    """Prepare final train/test or inner-fit/validation datasets for SPCI.

    ``data.train_ratio`` defines the entire training prefix. A supplied
    model-selection ratio reserves its tail for tuning, and test observations
    are sliced away before normalization, residual calculation, or windowing.
    Normal evaluation ignores any tuning settings saved in the configuration
    and refits on the full training prefix. Normalization uses only the portion
    fitting the forest in either mode. Response targets and base predictions
    retain their raw units for reporting.
    """
    if "valid_ratio" in config.data:
        raise ValueError(
            "SPCI no longer accepts data.valid_ratio. Use only data.train_ratio "
            "(the remainder is test). To retain the former approximate final "
            "training fraction, add the old train_ratio and valid_ratio, then "
            "remove valid_ratio. Set tuning.model_selection_valid_ratio for tuning."
        )
    train_ratio = _fraction(config.data.train_ratio, "data.train_ratio")
    window = config.model.window_size
    if isinstance(window, bool) or not isinstance(window, Integral) or window <= 0:
        raise ValueError("model.window_size must be a positive integer.")
    horizon = config.model.prediction_step
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon != 1:
        raise ValueError("SPCI supports model.prediction_step=1 only.")
    tuning = model_selection_valid_ratio is not None
    if tuning:
        model_selection_valid_ratio = _fraction(
            model_selection_valid_ratio, "tuning.model_selection_valid_ratio"
        )
    if not data:
        raise ValueError("SPCI requires at least one sequence.")

    prepared = ConformalPredictionData({})
    for key, item in data.items():
        raw_y = _scalar_sequence(item["heldout_y"], "heldout_y")
        raw_predictions = _scalar_sequence(item["heldout_predictions"], "heldout_predictions")
        raw_x = np.asarray(item["heldout_x"])
        if raw_x.ndim not in (1, 2):
            raise ValueError("heldout_x must have shape [T] or [T, features].")
        size = len(raw_y)
        if len(raw_x) != size or len(raw_predictions) != size:
            raise ValueError(f"Held-out arrays must have matching lengths for {key!r}.")
        train_end = int(np.floor(size * train_ratio))
        fit_end = (
            int(np.floor(np.nextafter(train_end * (1 - model_selection_valid_ratio), np.inf)))
            if tuning else train_end
        )
        evaluation_end = train_end if tuning else size
        if min(fit_end - window, evaluation_end - fit_end, size - train_end) <= 0:
            raise ValueError(
                f"Insufficient data for SPCI on {key!r}: length={size}, "
                f"training={train_end}, fit={fit_end}, window_size={window}, "
                f"evaluation={evaluation_end - fit_end}, test={size - train_end}."
            )

        # Do not inspect reserved test values during hyperparameter selection.
        raw_x = raw_x[:evaluation_end].copy()
        raw_y = raw_y[:evaluation_end].copy()
        raw_predictions = raw_predictions[:evaluation_end].copy()
        for name, values in (("heldout_x", raw_x), ("heldout_y", raw_y),
                             ("heldout_predictions", raw_predictions)):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite in the available prefix for {key!r}.")
        metadata = {
            "heldout_x": raw_x, "heldout_y": raw_y,
            "heldout_predictions": raw_predictions,
            "nominal_train_size": train_end, "train_size": fit_end,
            "test_size": size - train_end,
        }
        x, y, predictions = raw_x, raw_y, raw_predictions
        if config.data.get("normalize", False):
            x_mu, x_std = compute_mean_std(raw_x[:fit_end])
            y_mu, y_std = compute_mean_std(raw_y[:fit_end])
            x = normalize_array_with_params(raw_x, x_mu, x_std)
            y = normalize_array_with_params(raw_y, y_mu, y_std)
            predictions = normalize_array_with_params(raw_predictions, y_mu, y_std)
            metadata.update({
                "train_x_mu": x_mu, "train_x_std": x_std,
                "train_y_mu": y_mu, "train_y_std": y_std,
                "train_residuals_mu": np.zeros_like(y_mu),
                "train_residuals_std": y_std + 1e-8,
            })
        residuals = y - predictions
        metadata["heldout_residuals"] = residuals
        metadata["raw_heldout_residuals"] = raw_y - raw_predictions
        strided_x, target_x = to_strided_feature(x, window, 1, return_target=True)
        strided_residual, target_residual = to_strided_residual(residuals, window, 1)
        arrays = (
            strided_x, strided_residual, to_strided_feature(y, window, 1),
            target_x, target_residual, raw_y[window:, None], raw_predictions[window:, None],
        )
        # Each row targets time window + row_index; retain preceding context
        # across the boundary without moving evaluation timestamps with window.
        split = fit_end - window
        evaluation_key = "model_selection_valid_dataset" if tuning else "test_dataset"
        prepared.dataset[key] = {
            "train_dataset": QuantileRegressionDataset(*(array[:split] for array in arrays)),
            evaluation_key: QuantileRegressionDataset(*(array[split:] for array in arrays)),
        }
        if tuning:
            metadata["model_selection_valid_size"] = train_end - fit_end
        prepared.data[key] = metadata
    return prepared
