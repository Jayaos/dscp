"""Align saved one-step coverage results to the original held-out timeline."""

from collections.abc import Mapping
import math
from numbers import Integral, Real

import numpy as np


def _pair(values, name):
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != 2:
        raise ValueError(f"{name} must contain two quantile levels.")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in values):
        raise ValueError(f"{name} must contain real quantile levels.")
    lower, upper = sorted(float(v) for v in values)
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0 < lower < upper < 1:
        raise ValueError(f"{name} must satisfy 0 < lower < upper < 1.")
    return [lower, upper]


def _vector(values, name, *, scalar_column=False):
    array = np.asarray(values)
    if scalar_column and array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a numeric one-dimensional array.")
    return array


def _indices(values, name, length, *, chronological=False):
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    if array.size and (np.any(array < 0) or np.any(array >= length)):
        raise ValueError(f"{name} contains indices outside the source timeline.")
    array = array.astype(np.int64)
    if len(np.unique(array)) != len(array):
        raise ValueError(f"{name} contains duplicate target indices.")
    if chronological and np.any(np.diff(array) <= 0):
        raise ValueError(f"{name} must be in chronological order.")
    return array


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def _align_result(sequence_log, nominal_coverage, source_y, *, expected_test_indices,
                  allow_chronological_fallback, quantile_pair):
    if (isinstance(nominal_coverage, (bool, np.bool_))
            or not isinstance(nominal_coverage, Real)
            or not math.isfinite(nominal_coverage) or not 0 < nominal_coverage < 1):
        raise ValueError("nominal_coverage must be a finite fraction between zero and one.")
    if not isinstance(allow_chronological_fallback, bool):
        raise ValueError("allow_chronological_fallback must be a boolean.")
    if not isinstance(sequence_log, Mapping):
        raise ValueError("Sequence log must be a mapping.")
    metadata = sequence_log.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("Sequence metadata must be a mapping.")
    if metadata.get("split", "test") != "test":
        raise ValueError("Only final test results can be evaluated against shared test groups.")
    results = sequence_log.get("evaluation_results")
    if not isinstance(results, Mapping) or not results:
        raise ValueError("Sequence log must contain nonempty evaluation_results.")
    requested_pair = None if quantile_pair is None else _pair(quantile_pair, "quantile_pair")
    if requested_pair is not None and not math.isclose(
        requested_pair[1] - requested_pair[0], nominal_coverage, rel_tol=0, abs_tol=1e-8,
    ):
        raise ValueError("quantile_pair does not match nominal_coverage.")
    candidates = []
    for key, item in results.items():
        pair = _pair(key, "Saved quantile pair")
        if not math.isclose(pair[1] - pair[0], nominal_coverage, rel_tol=0, abs_tol=1e-8):
            continue
        if requested_pair is None or np.allclose(pair, requested_pair, rtol=0, atol=1e-8):
            candidates.append((pair, item))
    if not candidates:
        raise ValueError(f"No saved interval matches nominal coverage {nominal_coverage:g}.")
    if len(candidates) != 1:
        raise ValueError("Multiple saved intervals match nominal coverage; specify quantile_pair.")
    pair, result = candidates[0]
    if not isinstance(result, Mapping):
        raise ValueError("Saved interval result must be a mapping.")
    if "coverage" not in result or "target_y" not in result:
        raise ValueError("Saved interval result requires per-observation coverage and target_y.")
    coverage = _vector(result["coverage"], "coverage")
    if not np.isfinite(coverage).all() or not np.isin(coverage, [0, 1]).all():
        raise ValueError("coverage must contain only booleans or numeric zero/one values.")
    targets = _vector(result["target_y"], "target_y", scalar_column=True)
    if not np.isfinite(targets).all() or len(targets) != len(coverage):
        raise ValueError("target_y must be finite and align one-to-one with coverage.")
    source_y = _vector(source_y, "source_y", scalar_column=True)
    length, count = len(source_y), len(coverage)
    if not length:
        raise ValueError("source_y cannot be empty.")

    full = None
    if expected_test_indices is not None:
        full = _indices(expected_test_indices, "expected_test_indices", length, chronological=True)
    if "target_indices" in metadata:
        original = _indices(metadata["target_indices"], "metadata target_indices", length,
                            chronological=True)
        if full is not None and not np.array_equal(full, original):
            raise ValueError("Metadata target_indices disagree with the configured test timeline.")
        full = original
    mask = None
    if "valid_prediction_mask" in metadata:
        mask = np.asarray(metadata["valid_prediction_mask"])
        if mask.ndim != 1 or mask.dtype.kind != "b":
            raise ValueError("valid_prediction_mask must be a one-dimensional boolean array.")

    reported = {}
    known_exclusions = False
    for item in (metadata, result):
        for field in ("total_points", "evaluated_points", "excluded_points_count"):
            if field in item:
                value = _count(item[field], field)
                if field in reported and reported[field] != value:
                    raise ValueError(f"Conflicting saved {field} counts.")
                reported[field] = value
        known_exclusions |= (
            bool(item.get("excluded_points"))
            or item.get("evaluation_status") in {"completed_with_exclusions", "no_valid_predictions"}
        )
        if "exclusion_rate" in item:
            rate = item["exclusion_rate"]
            if (isinstance(rate, (bool, np.bool_)) or not isinstance(rate, Real)
                    or not math.isfinite(rate) or not 0 <= rate <= 1):
                raise ValueError("exclusion_rate must be a fraction in [0, 1].")
            known_exclusions |= rate > 0
    known_exclusions |= reported.get("excluded_points_count", 0) > 0
    known_exclusions |= reported.get("total_points", count) > count

    explicit = "target_indices" in result
    metadata_ids = "target_indices" in metadata
    if not explicit and not metadata_ids and not allow_chronological_fallback:
        raise ValueError(
            "Saved results have no target indices; enable allow_chronological_fallback "
            "to interpret their saved order as chronological."
        )
    if full is None:
        total = (len(mask) if mask is not None else reported.get("total_points"))
        if total is None and explicit and "excluded_points_count" in reported:
            total = count + reported["excluded_points_count"]
        if total is not None:
            if total > length:
                raise ValueError("Original test count exceeds the source timeline.")
            full = np.arange(length - total, length, dtype=np.int64)

    if mask is not None:
        if full is None or len(mask) != len(full):
            raise ValueError("valid_prediction_mask must cover the full original test timeline.")
        retained = full[mask]
    else:
        retained = full

    if explicit:
        indices = _indices(result["target_indices"], "result target_indices", length)
        alignment = "result_indices"
        if full is not None and not np.isin(indices, full).all():
            raise ValueError("Result target_indices fall outside the original test timeline.")
        if mask is not None and not np.array_equal(np.sort(indices), retained):
            raise ValueError("Result target_indices disagree with valid_prediction_mask.")
    elif metadata_ids:
        if mask is None and (known_exclusions or len(full) != count):
            raise ValueError("Excluded predictions require retained indices or valid_prediction_mask.")
        indices = retained
        alignment = "metadata_indices"
    else:
        if mask is None and (known_exclusions or (full is not None and len(full) != count)):
            raise ValueError("Excluded or missing predictions require retained indices or valid_prediction_mask.")
        if retained is None:
            if count > length:
                raise ValueError("Coverage count exceeds the source timeline.")
            indices = np.arange(length - count, length, dtype=np.int64)
            full = indices
        else:
            indices = retained
        alignment = ("chronological_config" if expected_test_indices is not None
                     else "chronological_suffix")

    if len(indices) != count:
        raise ValueError("Target indices must align one-to-one with coverage and target_y.")
    if not np.isfinite(source_y[indices]).all() or not np.allclose(
        targets, source_y[indices], rtol=1e-5, atol=1e-6,
    ):
        raise ValueError("Saved target_y disagrees with the source timeline at the aligned indices.")
    # Explicit retained identities can be sufficient for joining groups even
    # when an older log does not identify the complete original test timeline.
    total = len(full) if full is not None else None
    excluded = None if total is None else total - count
    actual_counts = {"total_points": total, "evaluated_points": count,
                     "excluded_points_count": excluded}
    for field, value in reported.items():
        if value != actual_counts[field]:
            raise ValueError(f"Saved {field} disagrees with the aligned prediction timeline.")
    if full is not None:
        boundaries = metadata.get("boundaries", {})
        if not isinstance(boundaries, Mapping):
            raise ValueError("Metadata boundaries must be a mapping.")
        for item in (metadata, boundaries):
            for field in ("test_start", "evaluation_start"):
                if field in item and (not len(full) or item[field] != int(full[0])):
                    raise ValueError(f"Logged {field} disagrees with the original test timeline.")
            if "test_size" in item and item["test_size"] != total:
                raise ValueError("Logged test_size disagrees with the original test timeline.")
    return {
        "target_indices": indices.copy(), "coverage": coverage.astype(bool),
        "target_y": targets.copy(), "quantile_pair": pair, "alignment": alignment,
        "full_test_count": total, "excluded_count": excluded,
    }


def align_result(sequence_log, nominal_coverage, source_y, *, expected_test_indices=None,
                 allow_chronological_fallback=False, quantile_pair=None, context=""):
    """Select a saved interval and attach validated original target identities.

    Explicit per-result indices take priority over metadata. The opt-in legacy
    fallback assumes the complete chronological test suffix, or the configured
    suffix when available. A saved validity mask preserves holes in that suffix.
    Known exclusions without identifiable retained rows are never guessed.
    No interval endpoints or models are needed: coverage comes from the log.
    Original test/exclusion counts are None when only retained identities are
    available; evaluated rows can still be joined to the saved shared groups.
    """
    try:
        return _align_result(
            sequence_log, nominal_coverage, source_y,
            expected_test_indices=expected_test_indices,
            allow_chronological_fallback=allow_chronological_fallback,
            quantile_pair=quantile_pair,
        )
    except ValueError as exc:
        if context:
            raise ValueError(f"{context}: {exc}") from exc
        raise
