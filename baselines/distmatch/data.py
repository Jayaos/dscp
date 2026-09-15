"""Causal residual preparation on the saved base-predictor held-out suffix."""

import numpy as np

from baselines.distmatch.config import split_boundaries


def _sequence_array(values, name):
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty scalar sequence with shape [T] or [T, 1].")
    return array


def scalar_sequence(values, name="residuals"):
    try:
        array = np.asarray(_sequence_array(values, name), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric scalar sequence.") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def prepare_sequence(item, config, split="test"):
    """Slice the permitted region before validating or transforming its values.

    In validation mode the reserved test suffix is used only for its length.
    No point-forecaster fitting data or contemporaneous covariates are consumed.
    """
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")
    if not isinstance(item, dict):
        raise ValueError("Each forecast series must be a mapping.")
    missing = {"heldout_y", "heldout_predictions"} - set(item)
    if missing:
        raise ValueError(f"Forecast series is missing fields: {sorted(missing)}")
    y_array = _sequence_array(item["heldout_y"], "heldout_y")
    prediction_array = _sequence_array(item["heldout_predictions"], "heldout_predictions")
    if len(y_array) != len(prediction_array):
        raise ValueError("heldout_y and heldout_predictions must have matching lengths.")
    data_config = config["data"]
    boundaries = split_boundaries(len(y_array), *(
        data_config[name] for name in ("train_ratio", "valid_ratio", "test_ratio")
    ))
    train_end, valid_end = boundaries["train_end"], boundaries["validation_end"]
    if split == "validation":
        if boundaries["validation_size"] == 0:
            raise ValueError("Validation evaluation requires a positive data.valid_ratio.")
        start, end = train_end, valid_end
    else:
        start, end = valid_end, len(y_array)
    y = scalar_sequence(y_array[:end], "heldout_y (available prefix)")
    predictions = scalar_sequence(prediction_array[:end], "heldout_predictions (available prefix)")
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = y - predictions
    if not np.isfinite(residuals).all():
        raise ValueError("Signed residuals must remain finite after subtraction.")
    return {
        "train_residuals": residuals[:train_end].copy(),
        "warmup_residuals": residuals[train_end:start].copy(),
        "residuals": residuals[start:end].copy(),
        "y": y[start:end].copy(),
        "predictions": predictions[start:end].copy(),
        "target_indices": np.arange(start, end, dtype=np.int64),
        "boundaries": boundaries,
    }
