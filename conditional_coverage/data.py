"""Causal observed-target contexts and shared chronological experiment splits.

Indices always refer to the saved held-out timeline, before prediction failures
or quantile-crossing exclusions. No base-predictor outputs enter the encoder.
"""

from collections.abc import Mapping
from decimal import Decimal, ROUND_FLOOR
import math
from numbers import Integral, Real

import numpy as np


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _ratio(value, name, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value >= 1
            or (value < 0 if allow_zero else value <= 0)):
        raise ValueError(f"{name} must be a finite fraction between zero and one.")
    return float(value)


def _real_array(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric observations.")
    return np.asarray(array, dtype=np.float64)


def observed_sequence(item):
    """Return ``(heldout_x, heldout_y)`` without reading saved predictions.

    Finiteness is checked later only where observations are actually needed.
    In particular, the final test outcome need not exist to embed its context.
    """
    if not isinstance(item, Mapping):
        raise ValueError("Each sequence must be a mapping of observed arrays.")
    missing = {"heldout_x", "heldout_y"} - item.keys()
    if missing:
        raise ValueError(f"Sequence is missing observed fields: {sorted(missing)}.")
    x = _real_array(item["heldout_x"], "heldout_x")
    y = _real_array(item["heldout_y"], "heldout_y")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("heldout_y must have nonempty shape (T,) or (T, 1).")
    if x.ndim != 2 or x.shape[1] == 0 or len(x) != len(y):
        raise ValueError("heldout_x must have shape (T, D), D >= 1, matching heldout_y.")
    return x, y


def _snap_integer(value):
    nearest = round(value)
    return nearest if math.isclose(value, nearest, rel_tol=0, abs_tol=1e-10) else value


def _method_boundary(length, method, config, sequence_key):
    if not isinstance(method, str) or not method:
        raise ValueError("Each comparison must identify a supported method.")
    if not isinstance(config, Mapping) or not isinstance(config.get("data"), Mapping):
        raise ValueError("Each comparison config must provide a data mapping.")
    settings = config["data"]
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        raise ValueError("Comparison model settings must be a mapping.")
    horizon = model.get("prediction_step", 1)
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon != 1:
        raise ValueError("Shared groups support one-step prediction only.")
    explicit = settings.get("test_start")
    if explicit is not None:
        if method not in {"split_cp", "aci", "cop", "lqt"}:
            raise ValueError(f"{method} does not support data.test_start overrides.")
        if isinstance(explicit, Mapping):
            if sequence_key not in explicit:
                raise ValueError(f"data.test_start is missing sequence {sequence_key!r}.")
            explicit = explicit[sequence_key]
        boundary = _integer(explicit, "data.test_start")
    elif method in {"qr_cp", "iqn_cp", "hopcpt", "distmatch"}:
        train = _ratio(settings.get("train_ratio"), "data.train_ratio")
        valid = _ratio(settings.get("valid_ratio"), "data.valid_ratio", allow_zero=True)
        if train + valid >= 1:
            raise ValueError("Training plus validation must leave a positive test suffix.")
        if "test_ratio" in settings:
            test = _ratio(settings["test_ratio"], "data.test_ratio")
            if not math.isclose(train + valid + test, 1, rel_tol=0, abs_tol=1e-8):
                raise ValueError("Training, validation and test ratios must sum to one.")
        snap = _snap_integer if method == "distmatch" else lambda value: value
        boundary = math.floor(snap(length * train)) + math.ceil(snap(length * valid))
    elif method == "local_cp":
        ratios = [_ratio(settings.get(name), f"data.{name}") for name in
                  ("train_ratio", "valid_ratio", "calibration_ratio", "test_ratio")]
        if not math.isclose(sum(ratios), 1, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Local-CP split ratios must sum to one.")
        boundary = math.floor(np.nextafter(length * np.cumsum(ratios[:3])[-1], np.inf))
    elif method == "spci":
        if "valid_ratio" in settings:
            raise ValueError("Legacy SPCI valid_ratio is ambiguous; use a current resolved config.")
        boundary = math.floor(length * _ratio(settings.get("train_ratio"), "data.train_ratio"))
    elif method in {"nexcp", "aci", "cop", "lqt", "split_cp", "rescp", "kowcpi"}:
        ratio = _ratio(settings.get("calibration_ratio"), "data.calibration_ratio")
        if "test_ratio" in settings:
            test = _ratio(settings["test_ratio"], "data.test_ratio")
            if not math.isclose(ratio + test, 1, rel_tol=0, abs_tol=1e-8):
                raise ValueError("Calibration and test ratios must sum to one.")
        if "validation_ratio" in settings:
            raise ValueError("Legacy calibration validation_ratio requires an explicit migration.")
        if method == "split_cp":
            boundary = int((Decimal(length) * Decimal(str(ratio))).to_integral_value(rounding=ROUND_FLOOR))
        elif method in {"rescp", "kowcpi"}:
            boundary = math.floor(_snap_integer(length * ratio))
        else:
            boundary = math.floor(length * ratio)
    else:
        raise ValueError(f"Unsupported comparison method: {method!r}.")
    if not 0 < boundary < length:
        raise ValueError(f"{method} has an empty pre-test or test split for length {length}.")
    return int(boundary)


def _indices(values, name, length, allow_empty=False):
    values = np.asarray(values)
    if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    values = values.astype(np.int64)
    if not allow_empty and not values.size:
        raise ValueError(f"{name} cannot be empty.")
    if values.size and (values[0] < 0 or values[-1] >= length or np.any(np.diff(values) <= 0)):
        raise ValueError(f"{name} must be strictly increasing, unique and within the source timeline.")
    return values


def _verify_logged_indices(record, expected, length, context):
    """Validate provenance while preserving observations excluded by a method."""
    if not isinstance(record, Mapping):
        raise ValueError(f"{context}: sequence log must be a mapping.")
    metadata = record.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{context}: metadata must be a mapping.")
    if metadata.get("split", "test") != "test":
        raise ValueError(f"{context}: comparison logs must describe final test evaluation.")
    for source in (metadata, metadata.get("boundaries", {})):
        if not isinstance(source, Mapping):
            raise ValueError(f"{context}: boundaries must be a mapping.")
        for field in ("test_start", "evaluation_start"):
            if field in source and source[field] != int(expected[0]):
                raise ValueError(f"{context}: logged {field} disagrees with configured test boundary.")
        for field in ("test_size", "total_points"):
            if field in source and source[field] != len(expected):
                raise ValueError(f"{context}: logged {field} disagrees with the original test timeline.")
    if "target_indices" in metadata:
        logged = _indices(metadata["target_indices"], f"{context} metadata target_indices", length)
        if not np.array_equal(logged, expected):
            raise ValueError(f"{context}: original logged test indices disagree with the configured timeline.")
    retained = expected
    if "valid_prediction_mask" in metadata:
        mask = np.asarray(metadata["valid_prediction_mask"])
        if mask.dtype != np.bool_ or mask.ndim != 1 or len(mask) != len(expected):
            raise ValueError(f"{context}: valid_prediction_mask must cover the original test timeline.")
        retained = expected[mask]
    results = record.get("evaluation_results", {})
    if not isinstance(results, Mapping):
        raise ValueError(f"{context}: evaluation_results must be a mapping.")
    for result in results.values():
        if not isinstance(result, Mapping):
            raise ValueError(f"{context}: each interval result must be a mapping.")
        if "target_indices" in result:
            ids = _indices(result["target_indices"], f"{context} result target_indices", length, allow_empty=True)
            if not np.array_equal(ids, retained):
                raise ValueError(f"{context}: retained target indices do not match the original timeline and mask.")
        for field in ("coverage", "target_y"):
            if field in result and len(result[field]) != len(retained):
                raise ValueError(f"{context}: {field} length disagrees with the original timeline and mask.")


def shared_indices(length, comparisons, sequence_key):
    """Intersect original pre-test prefixes and test suffixes across methods.

    ``comparisons`` contains name, method, config and optionally the complete
    sequence-keyed log mapping. Saved prediction exclusions never change splits.
    """
    length = _integer(length, "sequence length", minimum=2)
    if not isinstance(comparisons, (list, tuple)) or not comparisons:
        raise ValueError("At least one comparison is required.")
    boundaries = {}
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            raise ValueError("Each comparison must be a mapping.")
        name = comparison.get("name")
        if not isinstance(name, str) or not name or name in boundaries:
            raise ValueError("Comparison names must be nonempty and unique.")
        method = comparison.get("method")
        boundary = _method_boundary(length, method, comparison.get("config"), sequence_key)
        expected = np.arange(boundary, length, dtype=np.int64)
        log = comparison.get("log")
        if log is not None:
            if not isinstance(log, Mapping) or sequence_key not in log:
                raise ValueError(f"Comparison {name!r} has no log for sequence {sequence_key!r}.")
            _verify_logged_indices(log[sequence_key], expected, length, f"{name}/{sequence_key}")
        boundaries[name] = {"method": method, "calibration_end": boundary,
                            "test_start": boundary, "test_end": length,
                            "test_size": length - boundary}
    calibration_end = min(item["test_start"] for item in boundaries.values())
    test_start = max(item["test_start"] for item in boundaries.values())
    return {
        "calibration_indices": np.arange(calibration_end, dtype=np.int64),
        "test_indices": np.arange(test_start, length, dtype=np.int64),
        "member_boundaries": boundaries,
        "excluded_boundary_indices": np.arange(calibration_end, test_start, dtype=np.int64),
        "calibration_end": calibration_end,
        "test_start": test_start,
    }


def prepare_contexts(x, y, calibration_indices, test_indices, window_size, validation_ratio):
    """Normalize observed history with calibration-fit statistics only.

    Each target at index j uses ``features[j-window_size:j]``. Calibration
    targets start after within-prefix warmup. Validation is a chronological
    calibration tail. Test labels in ``y_normalized`` are deliberately missing;
    inference needs only contexts, which may include previously observed test y.
    """
    x, y = observed_sequence({"heldout_x": x, "heldout_y": y})
    window = _integer(window_size, "window_size")
    ratio = _ratio(validation_ratio, "validation_ratio")
    calibration = _indices(calibration_indices, "calibration_indices", len(y))
    test = _indices(test_indices, "test_indices", len(y))
    end = len(calibration)
    if not np.array_equal(calibration, np.arange(end)):
        raise ValueError("Calibration indices must be the full shared pre-test prefix starting at zero.")
    if test[0] < end:
        raise ValueError("Calibration and test observations must not overlap.")
    fit_end = math.floor(np.nextafter(end * (1 - ratio), np.inf))
    if fit_end <= window or fit_end >= end:
        raise ValueError("Insufficient calibration observations for context warmup, fitting and validation.")
    if test[0] < window:
        raise ValueError("Insufficient observed history for the first test context.")
    if not np.isfinite(y[:end]).all():
        raise ValueError("Calibration targets must be finite.")
    # Validate only feature rows used by fit statistics or an actual context.
    changes = np.zeros(len(y) + 1, dtype=np.int64)
    changes[0] += 1
    changes[max(end - 1, fit_end)] -= 1
    np.add.at(changes, test - window, 1)
    np.add.at(changes, test, -1)
    needed = np.cumsum(changes[:-1]) > 0
    raw = np.column_stack((x, y))
    if not np.isfinite(raw[needed]).all():
        raise ValueError("Observed values used in encoder contexts must be finite.")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = raw[:fit_end].mean(axis=0)
        std = raw[:fit_end].std(axis=0, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Calibration normalization statistics must be finite.")
    std = np.where(std > 0, std, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        features = ((raw - mean) / std).astype(np.float32)
        targets = np.full(len(y), np.nan, dtype=np.float32)
        targets[:end] = ((y[:end] - mean[-1]) / std[-1]).astype(np.float32)
    if not np.isfinite(features[needed]).all() or not np.isfinite(targets[:end]).all():
        raise ValueError("Normalized observations overflowed float32.")
    return {
        "features": features,
        "y_normalized": targets,
        "fit_indices": np.arange(window, fit_end, dtype=np.int64),
        "validation_indices": np.arange(fit_end, end, dtype=np.int64),
        "calibration_target_indices": np.arange(window, end, dtype=np.int64),
        "test_indices": test.copy(),
        "window_size": window,
        "scaler": {"feature_mean": mean, "feature_std": std,
                   "y_mean": float(mean[-1]), "y_std": float(std[-1]),
                   "fit_end": fit_end, "ddof": 0},
    }
