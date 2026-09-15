"""Chronological raw-sequence preparation for the training-free ResCP baseline."""

import math
from numbers import Integral

import numpy as np


def _sequence_array(values, name):
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty scalar sequence with shape [T] or [T, 1].")
    return array


def scalar_sequence(values, name="residuals"):
    """Canonicalize each scalar array before subtraction to avoid broadcasting."""
    array = _sequence_array(values, name)
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric scalar values.") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _snap_integer(value):
    # Avoid ceil(16.000000000000004) changing a chronological boundary.
    nearest = round(value)
    return nearest if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-10) else value


def split_boundaries(length, calibration_ratio, validation_ratio, test_ratio):
    """Split the complete held-out suffix, matching DSCP's floor/ceil convention.

    Calibration and test must be nonempty. Validation may be disabled with zero.
    Ratios are fractions of the base-predictor held-out artifact, not raw data.
    """
    if isinstance(length, bool) or not isinstance(length, Integral) or length <= 0:
        raise ValueError("length must be a positive integer.")
    raw_ratios = (calibration_ratio, validation_ratio, test_ratio)
    if any(isinstance(value, (bool, str)) for value in raw_ratios):
        raise ValueError("Split ratios must be finite numeric values.")
    try:
        calibration_ratio, validation_ratio, test_ratio = map(float, raw_ratios)
    except (TypeError, ValueError) as exc:
        raise ValueError("Split ratios must be finite numeric values.") from exc
    ratios = (calibration_ratio, validation_ratio, test_ratio)
    if not all(math.isfinite(value) for value in ratios):
        raise ValueError("Split ratios must be finite numeric values.")
    if calibration_ratio <= 0 or validation_ratio < 0 or test_ratio <= 0:
        raise ValueError("calibration_ratio and test_ratio must be positive; validation_ratio may be zero.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("calibration_ratio + validation_ratio + test_ratio must sum to one.")
    calibration_size = math.floor(_snap_integer(length * calibration_ratio))
    validation_size = math.ceil(_snap_integer(length * validation_ratio))
    test_size = int(length) - calibration_size - validation_size
    if calibration_size < 1 or test_size < 1:
        raise ValueError(
            f"Empty calibration/test split for length={length}: calibration={calibration_size}, "
            f"validation={validation_size}, test={test_size}."
        )
    return {
        "calibration_end": calibration_size,
        "validation_end": calibration_size + validation_size,
        "test_start": calibration_size + validation_size,
        "calibration_size": calibration_size,
        "validation_size": validation_size,
        "test_size": test_size,
    }


def prepare_sequence(item, config, split="test"):
    """Return only permitted history and evaluation values for one split.

    Full-array shapes determine boundaries, but reserved test values are never
    validated, normalized, or subtracted during validation-only evaluation.
    """
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")
    y_array = _sequence_array(item["heldout_y"], "heldout_y")
    prediction_array = _sequence_array(item["heldout_predictions"], "heldout_predictions")
    if len(y_array) != len(prediction_array):
        raise ValueError("heldout_y and heldout_predictions must have matching lengths.")
    data_config = config["data"]
    try:
        ratios = [data_config[name] for name in ("calibration_ratio", "validation_ratio", "test_ratio")]
    except KeyError as exc:
        raise ValueError("ResCP requires data.calibration_ratio, data.validation_ratio, and data.test_ratio.") from exc
    boundaries = split_boundaries(len(y_array), *ratios)
    calibration_end = boundaries["calibration_end"]
    validation_end = boundaries["validation_end"]
    if split == "validation":
        if boundaries["validation_size"] == 0:
            raise ValueError("Validation evaluation requires a positive data.validation_ratio.")
        start, end = calibration_end, validation_end
    else:
        start, end = validation_end, len(y_array)
    y = scalar_sequence(y_array[:end], "heldout_y (available prefix)")
    predictions = scalar_sequence(prediction_array[:end], "heldout_predictions (available prefix)")
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = y - predictions
    if not np.isfinite(residuals).all():
        raise ValueError("Signed residuals must remain finite after subtraction.")
    return {
        "calibration_residuals": residuals[:calibration_end].copy(),
        "warmup_residuals": residuals[calibration_end:start].copy(),
        "residuals": residuals[start:end].copy(),
        "y": y[start:end].copy(),
        "predictions": predictions[start:end].copy(),
        "target_indices": np.arange(start, end, dtype=np.int64),
        "boundaries": boundaries,
    }
