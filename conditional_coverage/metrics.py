"""Coverage diagnostics for fixed groups and equal-weight sequence summaries."""

from numbers import Integral, Real

import numpy as np


CACE_DEFINITION = "mean_group_undercoverage"


def _indices(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    if np.any(values < 0) or len(np.unique(values)) != len(values):
        raise ValueError(f"{name} must contain unique, nonnegative indices.")
    return values.astype(np.int64)


def evaluate_sequence(group_record, aligned, k, nominal_coverage):
    """Join predictions by target ID, then score the unchanged shared groups.

    Coverage uses available predictions within each group. Missing predictions
    remain visible in the original group size and missing count. Groups with no
    evaluated predictions have undefined coverage and are omitted from CACE and
    WUC; ``evaluated_group_count`` records their effective denominator.
    CACE averages positive coverage shortfalls over every evaluated group;
    groups at or above nominal coverage contribute zero. WUC is their maximum.
    """
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer.")
    if (isinstance(nominal_coverage, (bool, np.bool_))
            or not isinstance(nominal_coverage, Real)
            or not np.isfinite(nominal_coverage) or not 0 < nominal_coverage < 1):
        raise ValueError("nominal_coverage must be a finite fraction between zero and one.")
    k = int(k)
    nominal_coverage = float(nominal_coverage)
    group_indices = _indices(group_record["test_indices"], "Group test indices")
    target_indices = _indices(aligned["target_indices"], "Prediction target indices")
    labels = np.asarray(group_record["groups"][k]["test_labels"])
    if (labels.shape != group_indices.shape or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels >= k)):
        raise ValueError("Group labels must align with shared test indices and lie in [0, k).")
    coverage = np.asarray(aligned["coverage"])
    if (coverage.shape != target_indices.shape or coverage.dtype.kind not in "biuf"
            or not np.isfinite(coverage).all()
            or not np.isin(coverage, [0, 1]).all()):
        raise ValueError("Coverage must contain one boolean or 0/1 value per prediction index.")

    shared, group_positions, prediction_positions = np.intersect1d(
        group_indices, target_indices, assume_unique=True, return_indices=True,
    )
    original_counts = np.bincount(labels.astype(np.int64), minlength=k)
    evaluated_labels = labels[group_positions].astype(np.int64)
    evaluated_counts = np.bincount(evaluated_labels, minlength=k)
    covered_counts = np.bincount(
        evaluated_labels, weights=coverage[prediction_positions], minlength=k,
    ).astype(np.int64)
    groups = []
    for group in range(k):
        size = int(original_counts[group])
        evaluated = int(evaluated_counts[group])
        covered = int(covered_counts[group])
        value = covered / evaluated if evaluated else None
        groups.append({
            "group": group,
            "group_size": size,
            "evaluated_count": evaluated,
            "missing_count": size - evaluated,
            "covered_count": covered,
            "coverage": value,
            "absolute_coverage_error": abs(value - nominal_coverage) if value is not None else None,
            "undercoverage": max(nominal_coverage - value, 0.0) if value is not None else None,
        })

    available = [group for group in groups if group["evaluated_count"]]
    evaluated = len(shared)
    missing = len(group_indices) - evaluated
    covered = int(covered_counts.sum())
    sequence = {
        "k": k,
        "group_count": k,
        "evaluated_group_count": len(available),
        "empty_group_count": k - len(available),
        "group_size": len(group_indices),
        "evaluated_count": evaluated,
        "missing_count": missing,
        "covered_count": covered,
        "outside_shared_count": len(target_indices) - evaluated,
        "coverage": covered / evaluated if evaluated else None,
        "cace_definition": CACE_DEFINITION,
        "cace": float(np.mean([group["undercoverage"] for group in available])) if available else None,
        "wuc": max(group["undercoverage"] for group in available) if available else None,
        "status": (
            "no_valid_predictions" if not evaluated else
            "partial" if missing or len(available) < k else "complete"
        ),
    }
    return {"groups": groups, "sequence": sequence}


def aggregate_sequences(sequence_rows):
    """Average each sequence's errors; cluster labels never cross sequences.

    CACE is mean group undercoverage, equally weighted across sequences.
    Unavailable sequences are counted but omitted from averages. ``wuc`` is
    mean sequence WUC; ``worst_sequence_wuc`` is the maximum sequence WUC.
    Pooled coverage is supplied separately from equal-weight mean coverage.
    """
    rows = list(sequence_rows)
    available = [row for row in rows if row["cace"] is not None]
    summary = {
        "sequence_count": len(rows),
        "evaluated_sequence_count": len(available),
        "unavailable_sequence_count": len(rows) - len(available),
        "cace_definition": CACE_DEFINITION,
        "cace": float(np.mean([row["cace"] for row in available])) if available else None,
        "wuc": float(np.mean([row["wuc"] for row in available])) if available else None,
        "worst_sequence_wuc": max(row["wuc"] for row in available) if available else None,
        "mean_sequence_coverage": float(np.mean([row["coverage"] for row in available])) if available else None,
    }
    for field in (
        "group_size", "evaluated_count", "missing_count", "covered_count",
        "evaluated_group_count", "empty_group_count", "outside_shared_count",
    ):
        summary[field] = sum(int(row[field]) for row in rows)
    summary["pooled_coverage"] = (
        summary["covered_count"] / summary["evaluated_count"]
        if summary["evaluated_count"] else None
    )
    return summary
