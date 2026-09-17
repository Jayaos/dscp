"""Chronological residual preparation for KOWCPI calibration and tuning."""

from collections.abc import Mapping
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


def _finite_sequence(values, name):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real numeric scalar values.")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric scalar values.") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def _finite_ratio(value, name):
    message = f"{name} must be a finite numeric value strictly between 0 and 1."
    if isinstance(value, (bool, np.bool_, str)) or np.ndim(value) != 0:
        raise ValueError(message)
    try:
        ratio = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError(message)
    return ratio


def _snap_integer(value):
    nearest = round(value)
    return nearest if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-10) else value


def split_boundaries(length, calibration_ratio, test_ratio=None, *, model_selection_valid_ratio=None):
    """Reserve final test, optionally holding out a tail of calibration for tuning.

    Calibration is a fraction of the saved point-predictor held-out sequence;
    all remaining observations form the test split. An optional legacy
    test_ratio is checked for consistency but does not determine the split.
    The tuning ratio is a fraction of the nominal calibration prefix only.
    """
    if isinstance(length, (bool, np.bool_)) or not isinstance(length, Integral) or length <= 0:
        raise ValueError("length must be a positive integer.")
    calibration_ratio = _finite_ratio(calibration_ratio, "data.calibration_ratio")
    if test_ratio is not None:
        test_ratio = _finite_ratio(test_ratio, "data.test_ratio")
        if not math.isclose(calibration_ratio + test_ratio, 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("data.calibration_ratio + data.test_ratio must sum to one.")
    nominal_calibration_size = math.floor(_snap_integer(length * calibration_ratio))
    test_size = int(length) - nominal_calibration_size
    if nominal_calibration_size < 1 or test_size < 1:
        raise ValueError("The calibration and test splits must both be nonempty.")

    calibration_size = nominal_calibration_size
    validation_size = 0
    if model_selection_valid_ratio is not None:
        ratio = _finite_ratio(model_selection_valid_ratio, "tuning.model_selection_valid_ratio")
        calibration_size = math.floor(
            np.nextafter(nominal_calibration_size * (1.0 - ratio), np.inf)
        )
        validation_size = nominal_calibration_size - calibration_size
        if calibration_size < 1 or validation_size < 1:
            raise ValueError("The nested calibration and validation splits must both be nonempty.")
    return {
        "nominal_calibration_size": nominal_calibration_size,
        "nominal_calibration_end": nominal_calibration_size,
        "calibration_size": calibration_size,
        "calibration_end": calibration_size,
        "validation_size": validation_size,
        "validation_start": calibration_size,
        "validation_end": nominal_calibration_size,
        "test_size": test_size,
        "test_start": nominal_calibration_size,
    }


def prepare_sequence(item, config, split="test"):
    """Return history followed by evaluation residuals, with no future test values.

    Normal runs initialize on all nominal calibration observations. Tuning uses
    its earlier prefix as history and evaluates the final fraction specified by
    ``tuning.model_selection_valid_ratio`` (default 0.15). Normalization uses
    only the initial history for the requested split.
    """
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")
    data_config = config.get("data")
    if not isinstance(data_config, Mapping):
        raise ValueError("KOWCPI requires a data configuration mapping.")
    if any(name in data_config for name in ("train_ratio", "valid_ratio", "validation_ratio")):
        raise ValueError(
            "Legacy data.train_ratio/data.valid_ratio/data.validation_ratio are not supported. "
            "Use data.calibration_ratio for the outer split (the remainder is test), and "
            "tuning.model_selection_valid_ratio for validation within calibration."
        )
    try:
        calibration_ratio = data_config["calibration_ratio"]
    except KeyError as exc:
        raise ValueError("KOWCPI requires data.calibration_ratio; the remainder is test.") from exc

    selection_ratio = None
    if split == "validation":
        tuning_config = config.get("tuning", {})
        if not isinstance(tuning_config, Mapping):
            raise ValueError("KOWCPI tuning configuration must be a mapping.")
        selection_ratio = _finite_ratio(
            tuning_config.get("model_selection_valid_ratio", 0.15),
            "tuning.model_selection_valid_ratio",
        )

    y_array = _sequence_array(item["heldout_y"], "heldout_y")
    prediction_array = _sequence_array(item["heldout_predictions"], "heldout_predictions")
    if len(y_array) != len(prediction_array):
        raise ValueError("heldout_y and heldout_predictions must have matching lengths.")
    boundaries = split_boundaries(
        len(y_array), calibration_ratio, data_config.get("test_ratio"),
        model_selection_valid_ratio=selection_ratio,
    )
    calibration_size = boundaries["calibration_size"]
    if split == "validation":
        start, end = boundaries["validation_start"], boundaries["validation_end"]
    else:
        start, end = boundaries["test_start"], len(y_array)

    # Slice before conversion or arithmetic: tuning must not inspect test values.
    raw_y = _finite_sequence(y_array[:end], "heldout_y (available prefix)")
    raw_predictions = _finite_sequence(prediction_array[:end], "heldout_predictions (available prefix)")
    residual_std = None
    normalization_params = None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if bool(data_config.get("normalize", False)):
            mean = float(np.mean(raw_y[:calibration_size]))
            residual_std = float(np.std(raw_y[:calibration_size]) + 1e-8)
            if not np.isfinite(mean) or not np.isfinite(residual_std) or residual_std <= 0.0:
                raise ValueError("Calibration normalization statistics must remain finite.")
            residuals = (raw_y - mean) / residual_std - (raw_predictions - mean) / residual_std
            normalization_params = (0.0, residual_std)
        else:
            residuals = raw_y - raw_predictions
    if not np.isfinite(residuals).all():
        raise ValueError("Signed residuals must remain finite after subtraction and normalization.")
    return {
        "residuals": residuals,
        "raw_y": raw_y,
        "raw_predictions": raw_predictions,
        "calibration_size": calibration_size,
        "evaluation_size": end - start,
        "evaluation_start": start,
        "evaluation_end": end,
        "residual_normalized_std": residual_std,
        "residual_normalization_params": normalization_params,
        "boundaries": boundaries,
    }
