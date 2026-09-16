"""Two chronological partitions of a saved base-predictor held-out sequence."""

from collections.abc import Mapping
from decimal import Decimal, ROUND_FLOOR
import math
from numbers import Integral, Real

import numpy as np

from baselines.split_cp.model import scalar_sequence


def validate_split_settings(calibration_ratio=0.66, test_ratio=0.34, test_start=None):
    """Validate ratio settings even when an exact index overrides the boundary."""
    ratios = (calibration_ratio, test_ratio)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
           or not math.isfinite(v) or not 0 < v < 1 for v in ratios):
        raise ValueError("calibration_ratio and test_ratio must be finite numbers between 0 and 1.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError("calibration_ratio + test_ratio must sum to one.")
    if test_start is not None:
        starts = test_start.values() if isinstance(test_start, Mapping) else [test_start]
        if isinstance(test_start, Mapping) and not test_start:
            raise ValueError("data.test_start mapping must not be empty.")
        for start in starts:
            if isinstance(start, (bool, np.bool_)) or not isinstance(start, Integral) or start < 1:
                raise ValueError("data.test_start must contain positive integer indices.")


def split_boundaries(length, calibration_ratio=0.66, test_ratio=0.34, test_start=None):
    """Return half-open calibration/test boundaries, with no discarded targets.

    test_start is an optional zero-based exact boundary; it takes precedence
    over the ratio-derived boundary. All earlier observations are calibration.
    """
    if isinstance(length, (bool, np.bool_)) or not isinstance(length, Integral) or length < 1:
        raise ValueError("length must be a positive integer.")
    validate_split_settings(calibration_ratio, test_ratio, test_start)
    if isinstance(test_start, Mapping):
        raise ValueError("Resolve data.test_start to a single series index first.")
    if test_start is None:
        size = Decimal(int(length)) * Decimal(str(float(calibration_ratio)))
        boundary = int(size.to_integral_value(rounding=ROUND_FLOOR))
    else:
        boundary = int(test_start)
    if not 0 < boundary < length:
        raise ValueError(f"Empty calibration/test split: length={length}, test_start={boundary}.")
    return {
        "calibration_start": 0,
        "calibration_end": boundary,
        "test_start": boundary,
        "calibration_size": boundary,
        "test_size": int(length) - boundary,
    }


def prepare_sequence(item, config, key=None):
    """Use only pre-test residuals for calibration; features are not required."""
    y = scalar_sequence(item["heldout_y"], "heldout_y")
    predictions = scalar_sequence(item["heldout_predictions"], "heldout_predictions")
    if len(y) != len(predictions):
        raise ValueError("heldout_y and heldout_predictions must have matching lengths.")
    settings = config["data"]
    start = settings.get("test_start")
    if isinstance(start, Mapping):
        if key not in start:
            raise ValueError(f"data.test_start has no boundary for series {key!r}.")
        start = start[key]
    boundaries = split_boundaries(
        len(y), settings.get("calibration_ratio", 0.66), settings.get("test_ratio", 0.34), start,
    )
    boundary = boundaries["test_start"]
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = y[:boundary] - predictions[:boundary]
    if not np.isfinite(residuals).all():
        raise ValueError("Calibration residuals overflowed the original response scale.")
    return {
        "calibration_residuals": residuals,
        "y": y[boundary:].copy(),
        "predictions": predictions[boundary:].copy(),
        "target_indices": np.arange(boundary, len(y), dtype=np.int64),
        "boundaries": boundaries,
    }
