"""ACI reporting with complete coverage and conditional efficiency summaries.

``avg_interval_width`` and ``avg_winkler_score`` use finite, nonempty sets only.
Coverage retains every prediction. Full-test diagnostics retain infinite widths;
the extended full-test score assigns +infinity to full and empty sets. Assigning
infinite score to an empty set is an explicit reporting convention, not the
ordinary Winkler formula for an ordered interval.
"""

import math
from numbers import Integral, Real
from pathlib import Path
import re

import numpy as np


METRICS_SCOPE = {
    "coverage": "all_test_predictions",
    "avg_interval_width": "finite_nonempty_intervals_only",
    "avg_winkler_score": "finite_nonempty_intervals_only",
    "full_test_interval_width": "full=infinity; empty=0",
    "full_test_winkler_score": "extended convention: full=infinity; empty=infinity",
    "winkler_alpha": "fixed_nominal_miscoverage",
}


def _vector(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (T,) or (T, 1).")
    return array


def _mean_std(values):
    """Preserve undefined means and infinite means without NaN dispersion."""
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return None, None
    if any(math.isnan(value) for value in defined):
        raise ValueError("Metric values must not contain NaN.")
    if any(math.isinf(value) for value in defined):
        signs = {math.copysign(1, value) for value in defined if math.isinf(value)}
        if len(signs) != 1:
            raise ValueError("Metric values cannot mix positive and negative infinity.")
        return math.copysign(math.inf, next(iter(signs))), None
    mean = math.fsum(value / len(defined) for value in defined)
    deviations = [abs(value - mean) for value in defined]
    scale = max(deviations)
    std = (scale * math.sqrt(math.fsum((value / scale) ** 2 / len(defined)
                                     for value in deviations)) if scale else 0.0)
    return mean, std


def evaluate_intervals(targets, predictions, lower, upper, alpha, target_indices):
    """Evaluate response-scale endpoints, including full and empty sets.

    Full sets use (-inf, inf), empty sets use (inf, -inf). Other infinite or
    reversed finite endpoints are rejected. Singleton intervals are nonempty.
    """
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real)
            or not math.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("alpha must be finite and strictly between zero and one.")
    y = _vector(targets, "targets")
    point = _vector(predictions, "predictions")
    lo = _vector(lower, "lower")
    hi = _vector(upper, "upper")
    size = len(y)
    if size == 0 or any(len(array) != size for array in (point, lo, hi)):
        raise ValueError("Targets, predictions, and intervals must have the same positive length.")
    if not np.isfinite(y).all() or not np.isfinite(point).all():
        raise ValueError("Targets and point predictions must be finite.")
    indices = np.asarray(target_indices)
    if (indices.ndim != 1 or len(indices) != size
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices[1:] <= indices[:-1])):
        raise ValueError("target_indices must be matching, strictly increasing nonnegative integers.")
    regular = np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
    full = np.isneginf(lo) & np.isposinf(hi)
    empty = np.isposinf(lo) & np.isneginf(hi)
    if not np.all(regular | full | empty):
        raise ValueError("Intervals must be finite ordered intervals, full sets, or empty sets.")
    covered = (lo <= y) & (y <= hi)
    width = np.zeros(size, dtype=float)
    score = np.full(size, np.inf, dtype=float)
    width[full] = np.inf
    # Calculate finite scores separately to avoid inactive 0 * infinity terms.
    with np.errstate(over="ignore", invalid="ignore"):
        width[regular] = hi[regular] - lo[regular]
        score[regular] = (
            width[regular]
            + (2.0 / alpha) * np.maximum(lo[regular] - y[regular], 0.0)
            + (2.0 / alpha) * np.maximum(y[regular] - hi[regular], 0.0)
        )
        lower_residual = lo - point
        upper_residual = hi - point
    if (not np.isfinite(width[regular]).all() or not np.isfinite(score[regular]).all()
            or not np.isfinite(lower_residual[regular]).all()
            or not np.isfinite(upper_residual[regular]).all()):
        raise ValueError("Finite-interval metrics overflowed the response scale.")
    kinds = np.full(size, "regular", dtype=object)
    kinds[full] = "full"
    kinds[empty] = "empty"
    n_finite, n_full, n_empty = (int(np.count_nonzero(mask)) for mask in (regular, full, empty))
    avg_coverage = float(np.mean(covered))
    return {
        "lower_interval": lo.tolist(), "upper_interval": hi.tolist(),
        "lower_residual_quantile": lower_residual.tolist(),
        "upper_residual_quantile": upper_residual.tolist(),
        "target_y": y.tolist(), "target_predictions": point.tolist(),
        "target_indices": indices.tolist(), "coverage": covered.tolist(),
        "interval_kind": kinds.tolist(), "interval_width": width.tolist(),
        "winkler_score": score.tolist(), "finite_interval_mask": regular.tolist(),
        "finite_interval_width": width[regular].tolist(),
        "finite_winkler_score": score[regular].tolist(),
        "avg_coverage": avg_coverage,
        "avg_delta_coverage": avg_coverage - (1.0 - float(alpha)),
        "avg_interval_width": _mean_std(width[regular])[0],
        "avg_winkler_score": _mean_std(score[regular])[0],
        "full_test_avg_interval_width": _mean_std(width)[0],
        "full_test_avg_winkler_score": _mean_std(score)[0],
        "num_test_points": size, "num_finite_intervals": n_finite,
        "num_full_sets": n_full, "num_empty_sets": n_empty,
        "full_set_rate": n_full / size, "empty_set_rate": n_empty / size,
        "metrics_scope": METRICS_SCOPE.copy(),
    }


def summarize_results(log, target_quantiles):
    """Average metrics equally across series, with explicit missing counts.

    Efficiency means include only series with an ordinary interval. Full/empty
    rates without a suffix pool all timestamps; their _mean/_std counterparts
    summarize the per-series rates with equal weights.
    """
    summary = {}
    for pair in target_quantiles:
        key = tuple(pair)
        results = [item["evaluation_results"][key] for series, item in log.items()
                   if series != "summary_results"]
        if not results:
            raise ValueError("Cannot summarize ACI results without any series.")
        row = {"num_sequences": len(results), "metrics_scope": METRICS_SCOPE.copy()}
        for metric in ("avg_coverage", "avg_delta_coverage", "avg_interval_width",
                       "avg_winkler_score", "full_test_avg_interval_width",
                       "full_test_avg_winkler_score", "full_set_rate", "empty_set_rate"):
            mean, std = _mean_std(result[metric] for result in results)
            row[f"{metric}_mean"], row[f"{metric}_std"] = mean, std
        for count in ("num_test_points", "num_finite_intervals", "num_full_sets", "num_empty_sets"):
            row[count] = sum(result[count] for result in results)
        row["num_efficiency_sequences"] = sum(result["num_finite_intervals"] > 0 for result in results)
        row["num_undefined_efficiency_sequences"] = len(results) - row["num_efficiency_sequences"]
        row["full_set_rate"] = row["num_full_sets"] / row["num_test_points"]
        row["empty_set_rate"] = row["num_empty_sets"] / row["num_test_points"]
        row["aggregation"] = {
            "metric_means": "equal_weight_per_series",
            "efficiency_means": "equal_weight_per_series_with_finite_nonempty_intervals",
            "full_set_rate_and_empty_set_rate": "pooled_over_all_test_predictions",
        }
        summary[key] = row
    return summary


def plot_aci_results(log, pairs, plot_length, output_dir, method_name="ACI"):
    """Plot ordinary intervals and mark exceptional sets without hiding time."""
    if isinstance(plot_length, (bool, np.bool_)) or not isinstance(plot_length, Integral) or plot_length < 1:
        raise ValueError("plot_length must be a positive integer.")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for sequence_index, (sequence, item) in enumerate(log.items()):
        if sequence == "summary_results":
            continue
        for pair_index, pair in enumerate(pairs):
            result = item["evaluation_results"][tuple(pair)]
            take = slice(0, plot_length)
            indices = np.asarray(result["target_indices"])[take]
            targets = np.asarray(result["target_y"])[take]
            predictions = np.asarray(result["target_predictions"])[take]
            regular = np.asarray(result["finite_interval_mask"], dtype=bool)[take]
            kinds = np.asarray(result["interval_kind"])[take]
            lo = np.asarray(result["lower_interval"], dtype=float)[take]
            hi = np.asarray(result["upper_interval"], dtype=float)[take]
            # Display gaps only; original logged endpoints/indices remain intact.
            display_lo = np.where(regular, lo, np.nan)
            display_hi = np.where(regular, hi, np.nan)
            figure = Figure(figsize=(12, 4.5), constrained_layout=True)
            FigureCanvasAgg(figure)
            axes = figure.subplots()
            axes.plot(indices, targets, color="black", linewidth=1, label="Observed")
            axes.plot(indices, predictions, color="tab:blue", linewidth=1, label="Point forecast")
            axes.fill_between(indices, display_lo, display_hi, where=regular,
                              color="tab:blue", alpha=0.2, label="Finite prediction interval")
            for kind, marker, color, label in (("full", "^", "tab:purple", "Full set"),
                                               ("empty", "x", "tab:red", "Empty set")):
                mask = kinds == kind
                if mask.any():
                    axes.scatter(indices[mask], targets[mask], marker=marker, color=color,
                                 label=f"{label} (marked at observed value)", zorder=4)
            axes.set_xlabel("Target index in saved forecast sequence")
            axes.set_ylabel("Response")
            coverage = max(pair) - min(pair)
            axes.set_title(f"{method_name} | {sequence} | target coverage {coverage:.1%}")
            axes.legend(loc="best", fontsize="small")
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sequence)).strip(".") or "series"
            path = destination / f"{sequence_index:04d}_{safe_name}_{pair_index:02d}.png"
            figure.savefig(path, dpi=150)
            figure.clear()
            paths.append(path)
    return paths
