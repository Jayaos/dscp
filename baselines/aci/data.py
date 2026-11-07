"""NexCP-compatible chronological calibration/test splits for ACI."""

from collections.abc import Mapping
import math

import numpy as np

from baselines.aci.model import finite_number, positive_integer


def sequence_array(values, name):
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or len(array) == 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty numeric scalar sequence with shape (T,) or (T, 1).")
    return array


def finite_sequence(values, name):
    array = np.asarray(sequence_array(values, name), dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values.")
    return array


def validate_split_settings(calibration_ratio=0.66, test_ratio=None, test_start=None):
    calibration_ratio = finite_number(calibration_ratio, "data.calibration_ratio")
    test_ratio = 1.0 - calibration_ratio if test_ratio is None else finite_number(test_ratio, "data.test_ratio")
    if not 0 < calibration_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("Calibration and test ratios must be strictly between zero and one.")
    if not math.isclose(calibration_ratio + test_ratio, 1.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError("Calibration and test ratios must sum to one.")
    if test_start is not None:
        starts = test_start.values() if isinstance(test_start, Mapping) else [test_start]
        if isinstance(test_start, Mapping) and not test_start:
            raise ValueError("data.test_start cannot be an empty mapping.")
        for start in starts:
            positive_integer(start, "data.test_start")
    return calibration_ratio, test_ratio


def split_boundaries(length, calibration_ratio=0.66, test_ratio=None, test_start=None):
    length = positive_integer(length, "length")
    calibration_ratio, _ = validate_split_settings(calibration_ratio, test_ratio, test_start)
    if isinstance(test_start, Mapping):
        raise ValueError("Resolve per-sequence test_start before constructing boundaries.")
    # Deliberately match NexCP's floating-point floor, including its rounding.
    boundary = int(np.floor(length * calibration_ratio)) if test_start is None else int(test_start)
    if not 0 < boundary < length:
        raise ValueError(f"Empty calibration/test split: length={length}, test_start={boundary}.")
    return {"calibration_start": 0, "calibration_end": boundary,
            "calibration_size": boundary, "test_start": boundary, "test_size": length - boundary}


def make_scores(y, predictions, score_type="absolute"):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        absolute = np.abs(y - predictions)
        if score_type == "absolute":
            scales = np.ones_like(predictions, dtype=np.float64)
        elif score_type == "relative":
            if np.any(predictions <= 0):
                raise ValueError("The paper's relative score requires strictly positive point forecasts.")
            scales = predictions.copy()
        else:
            raise ValueError("model.score_type must be 'absolute' or 'relative'.")
        scores = absolute / scales
    if not np.isfinite(scores).all():
        raise ValueError("Scores overflowed; predictions and residual scores must remain finite.")
    return scores, scales


def prepare_sequence(item, config, key=None, split="test"):
    """For validation, never inspect finite values or compute scores in test."""
    if split not in {"test", "validation"}:
        raise ValueError("split must be 'test' or 'validation'.")
    y_array = sequence_array(item["heldout_y"], "heldout_y")
    predictions_array = sequence_array(item["heldout_predictions"], "heldout_predictions")
    if len(y_array) != len(predictions_array):
        raise ValueError("heldout_y and heldout_predictions must have matching lengths.")
    settings = config["data"]
    start = settings.get("test_start")
    if isinstance(start, Mapping):
        if key not in start:
            raise ValueError(f"data.test_start has no boundary for sequence {key!r}.")
        start = start[key]
    boundaries = split_boundaries(len(y_array), settings.get("calibration_ratio", .66), settings.get("test_ratio"), start)
    outer_end = boundaries["calibration_end"]
    calibration_end, evaluation_end = outer_end, len(y_array)
    if split == "validation":
        ratio = finite_number(config.get("tuning", {}).get("validation_ratio", .2), "tuning.validation_ratio")
        if not 0 < ratio < 1:
            raise ValueError("tuning.validation_ratio must be strictly between zero and one.")
        calibration_end = math.floor(np.nextafter(outer_end * (1 - ratio), np.inf))
        evaluation_end = outer_end
        if not 0 < calibration_end < evaluation_end:
            raise ValueError("Calibration prefix is too short for initialization and tuning validation.")
    y = finite_sequence(y_array[:evaluation_end], "available heldout_y")
    predictions = finite_sequence(predictions_array[:evaluation_end], "available heldout_predictions")
    scores, scales = make_scores(y, predictions, config.get("model", {}).get("score_type", "absolute"))
    boundaries = {**boundaries, "initialization_end": calibration_end,
                  "evaluation_start": calibration_end, "evaluation_end": evaluation_end}
    return {"calibration_scores": scores[:calibration_end].copy(),
            "scores": scores[calibration_end:evaluation_end].copy(),
            "scales": scales[calibration_end:evaluation_end].copy(),
            "y": y[calibration_end:evaluation_end].copy(),
            "predictions": predictions[calibration_end:evaluation_end].copy(),
            "target_indices": np.arange(calibration_end, evaluation_end, dtype=np.int64),
            "boundaries": boundaries}
