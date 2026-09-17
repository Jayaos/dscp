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
    # Avoid floor(28.999999999999996) changing a chronological boundary.
    nearest = round(value)
    return nearest if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-10) else value


def validate_model_selection_valid_ratio(value):
    """Validate the fraction of nominal calibration reserved for model selection."""
    message = "tuning.model_selection_valid_ratio must be finite and strictly between 0 and 1."
    if isinstance(value, (bool, np.bool_, str)):
        raise ValueError(message)
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError(message)
    return ratio


def split_boundaries(
    length, calibration_ratio, test_ratio, *, model_selection_valid_ratio=None
):
    """Split held-out forecasts into calibration/test, optionally nesting tuning.

    Outer ratios are fractions of the saved base-predictor suffix. For tuning,
    the last ``model_selection_valid_ratio`` of nominal calibration supplies
    validation; the earlier prefix initializes ResCP. Its rounding matches the
    chronological model-selection split used by QR-CP and IQN-CP.
    """
    if isinstance(length, bool) or not isinstance(length, Integral) or length <= 0:
        raise ValueError("length must be a positive integer.")
    raw_ratios = (calibration_ratio, test_ratio)
    if any(isinstance(value, (bool, np.bool_, str)) for value in raw_ratios):
        raise ValueError("Split ratios must be finite numeric values.")
    try:
        calibration_ratio, test_ratio = map(float, raw_ratios)
    except (TypeError, ValueError) as exc:
        raise ValueError("Split ratios must be finite numeric values.") from exc
    ratios = (calibration_ratio, test_ratio)
    if not all(math.isfinite(value) for value in ratios):
        raise ValueError("Split ratios must be finite numeric values.")
    if calibration_ratio <= 0 or test_ratio <= 0:
        raise ValueError("calibration_ratio and test_ratio must be positive.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("calibration_ratio + test_ratio must sum to one.")
    nominal_calibration_size = math.floor(_snap_integer(length * calibration_ratio))
    test_size = int(length) - nominal_calibration_size
    if nominal_calibration_size < 1 or test_size < 1:
        raise ValueError(
            f"Empty calibration/test split for length={length}: "
            f"calibration={nominal_calibration_size}, test={test_size}."
        )
    calibration_size = nominal_calibration_size
    validation_size = 0
    if model_selection_valid_ratio is not None:
        ratio = validate_model_selection_valid_ratio(model_selection_valid_ratio)
        calibration_size = math.floor(
            np.nextafter(nominal_calibration_size * (1.0 - ratio), np.inf)
        )
        validation_size = nominal_calibration_size - calibration_size
        if calibration_size < 1 or validation_size < 1:
            raise ValueError(
                f"Empty nested model-selection split for length={length}: "
                f"calibration={calibration_size}, validation={validation_size}, test={test_size}."
            )
    return {
        "nominal_calibration_end": nominal_calibration_size,
        "nominal_calibration_size": nominal_calibration_size,
        "calibration_end": calibration_size,
        "validation_start": calibration_size,
        "validation_end": nominal_calibration_size,
        "test_start": nominal_calibration_size,
        "calibration_size": calibration_size,
        "validation_size": validation_size,
        "test_size": test_size,
    }


def prepare_sequence(item, config, split="test"):
    """Return only permitted history and evaluation values for one split.

    Test evaluation initializes on the complete nominal calibration prefix.
    Validation evaluation initializes on the earlier part of that prefix and
    evaluates its reserved tail, using tuning.model_selection_valid_ratio (0.2
    by default). Final test values are never validated or subtracted during
    validation-only evaluation.
    """
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")
    y_array = _sequence_array(item["heldout_y"], "heldout_y")
    prediction_array = _sequence_array(item["heldout_predictions"], "heldout_predictions")
    if len(y_array) != len(prediction_array):
        raise ValueError("heldout_y and heldout_predictions must have matching lengths.")
    data_config = config["data"]
    if "validation_ratio" in data_config:
        raise ValueError(
            "data.validation_ratio is no longer supported. Use calibration_ratio/test_ratio "
            "for the outer split and tuning.model_selection_valid_ratio for tuning."
        )
    try:
        ratios = [data_config[name] for name in ("calibration_ratio", "test_ratio")]
    except KeyError as exc:
        raise ValueError("ResCP requires data.calibration_ratio and data.test_ratio.") from exc
    selection_ratio = None
    if split == "validation":
        selection_ratio = validate_model_selection_valid_ratio(
            config.get("tuning", {}).get("model_selection_valid_ratio", 0.2)
        )
    boundaries = split_boundaries(
        len(y_array), *ratios, model_selection_valid_ratio=selection_ratio
    )
    calibration_end = boundaries["calibration_end"]
    validation_end = boundaries["validation_end"]
    if split == "validation":
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
