"""Log-only reproduction of COP paper Table 2 on cached AMZN/Prophet data.

This intentionally uses Hu et al.'s baseline functions and reporting wrapper,
not this repository's calibration/test benchmark variants.
"""

import csv
from datetime import date
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np

from baselines.cop_table2.upstream_methods import aci, LQT, COP

COMMIT = "a41c0dcadc10979d8521ab9b5b8f3d460807d883"
PAPER = "https://arxiv.org/html/2512.07770v2#S4.T2"
SOURCE = "https://github.com/creator-xi/Conformal-Optimistic-Prediction/tree/" + COMMIT
FORECAST_SHA256 = "5955e5750ef25833043e335d277fe0c59c26a0e29a9b4520c39b591682564efd"
TARGETS_CSV_SHA256 = "8f56f062dffe2947b87ec9819dbf377e2f107ded6f26d1910d4149c99c533ce7"
ALPHA, BURNIN = .1, 100
COVERAGE_START = BURNIN + 1
LEARNING_RATES = {
    "aci": (.1, .05, .01, .005),
    "lqt": (10., 5., 1., .5, .1, .05, .01),
    "cop": (1., .5, .1, .05),
}
TABLE2 = {
    "aci": {"coverage_percent": 90.2, "mean_width": math.inf, "median_width": 46.97},
    "lqt": {"coverage_percent": 89.3, "mean_width": 31.42, "median_width": 15.26},
    "cop": {"coverage_percent": 89.6, "mean_width": 39.86, "median_width": 27.91},
}


def load_amzn_data(data_path, targets_path):
    """Read existing inputs only; align the missing target as the authors do."""
    data_path, targets_path = Path(data_path), Path(targets_path)
    with np.load(data_path, allow_pickle=False) as saved:
        if "forecasts" not in saved.files:
            raise ValueError("The forecast archive must contain a 'forecasts' array.")
        point = saved["forecasts"]
        if point.ndim != 1 or point.size == 0 or point.dtype.kind not in "iuf":
            raise ValueError("forecasts must be a nonempty numeric vector.")
        point = np.asarray(point, dtype=np.float64)
    if not np.isfinite(point).all():
        raise ValueError("Forecasts must be finite, including the author's startup placeholders.")
    text = targets_path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != ["timestamp", "Open"]:
        raise ValueError("Target CSV must have the columns timestamp,Open in that order.")
    rows = list(reader)
    if len(rows) != len(point):
        raise ValueError(f"Target/forecast lengths differ: {len(rows)} versus {len(point)}. "
                         "Use the full union-date target file, including the missing AMZN date.")
    timestamps = [row["timestamp"] for row in rows]
    for timestamp in timestamps:
        date.fromisoformat(timestamp)
    if timestamps != sorted(set(timestamps)):
        raise ValueError("Target dates must be strictly increasing and unique.")
    y = np.asarray([float(row["Open"]) if row["Open"].strip() else np.nan for row in rows])
    missing = np.isnan(y)
    if not np.isfinite(y[~missing]).all() or missing.all() or missing[0]:
        raise ValueError("Targets must be finite except for interpolatable missing observations.")
    # Pandas' default linear interpolation uses row positions, not calendar-day gaps.
    indices = np.arange(len(y))
    y[missing] = np.interp(indices[missing], indices[~missing], y[~missing])
    forecast_hash = hashlib.sha256(point.astype("<f8", copy=False).tobytes()).hexdigest()
    target_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    leading_ones = 0
    while leading_ones < len(point) and point[leading_ones] == 1.:
        leading_ones += 1
    metadata = {
        "forecast_path": str(data_path), "targets_path": str(targets_path),
        "num_points": len(y), "first_date": timestamps[0], "last_date": timestamps[-1],
        "interpolated_target_dates": [timestamps[i] for i in np.flatnonzero(missing)],
        "leading_forecast_ones_retained": leading_ones,
        "forecast_values_sha256": forecast_hash,
        "matches_authors_prophet_forecasts": forecast_hash == FORECAST_SHA256,
        "target_csv_sha256": target_hash,
        "matches_pinned_target_csv": target_hash == TARGETS_CSV_SHA256,
        "source_commit": COMMIT,
    }
    return {"y": y, "forecasts": point, "timestamps": timestamps, "metadata": metadata}


def author_intervals(forecasts, lower_threshold, upper_threshold):
    """Exactly the base_test.py minsize=0 endpoint wrapper, including crossings."""
    raw_lower = forecasts - lower_threshold
    raw_upper = forecasts + upper_threshold
    if np.isnan(raw_lower).any() or np.isnan(raw_upper).any():
        raise ValueError("The source algorithm produced NaN endpoints.")
    lower = np.minimum(raw_lower, raw_upper)
    upper = np.maximum(raw_upper, raw_lower)
    return lower, upper, raw_lower > raw_upper


def evaluate_author_metrics(y, lower, upper, coverage_start=COVERAGE_START):
    """Coverage after burn-in; main widths over the whole source stream."""
    if not 0 <= coverage_start < len(y):
        raise ValueError("Need observations after the source evaluation warmup.")
    width = upper - lower
    if np.isnan(width).any() or np.any(width < 0):
        raise ValueError("Source intervals have undefined or negative widths.")
    covered = (lower <= y) & (y <= upper)
    return {
        "coverage_percent": float(np.mean(covered[coverage_start:]) * 100),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "post_burnin_mean_width": float(np.mean(width[coverage_start:])),
        "post_burnin_median_width": float(np.median(width[coverage_start:])),
        "num_width_points": len(y), "num_coverage_points": len(y)-coverage_start,
        "num_infinite_widths": int(np.isinf(width).sum()),
        "num_infinite_widths_post_burnin": int(np.isinf(width[coverage_start:]).sum()),
    }


def _rounded_equal(value, expected, decimals):
    if math.isinf(expected):
        return value == expected
    return math.isfinite(value) and f"{value:.{decimals}f}" == f"{expected:.{decimals}f}"


def matches_table2(metrics, reference):
    return all(_rounded_equal(metrics[name], reference[name], decimals)
               for name, decimals in (("coverage_percent", 1), ("mean_width", 2), ("median_width", 2)))


def _number(value, decimals=4):
    return f"{value:.{decimals}f}" if math.isfinite(value) else str(value)


def _delta(value, reference):
    if value == reference and math.isinf(value):
        return "both_inf"
    return _number(value-reference)


def run_reproduction(data_path, targets_path, methods=None, lqt_order=1):
    """Print every rate candidate; order 1 is the empirically recovered setup."""
    if lqt_order not in (1, 2):
        raise ValueError("lqt_order must be 1 (recovered Table 2 setup) or 2 (released YAML).")
    chosen = tuple(LEARNING_RATES) if methods is None else tuple(name.lower() for name in methods)
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen)-set(LEARNING_RATES):
        raise ValueError("methods must be a nonempty unique subset of aci, lqt, cop.")
    loaded = load_amzn_data(data_path, targets_path)
    y, point = loaded["y"], loaded["forecasts"]
    if len(y) <= COVERAGE_START:
        raise ValueError("The source protocol requires more than 101 observations.")
    print("COP paper Table 2: AMZN stock / cached Prophet predictor", flush=True)
    print("Paper: " + PAPER)
    print("Pinned author source: " + SOURCE)
    print("Inputs: " + json.dumps(loaded["metadata"], sort_keys=True))
    print("Protocol: uninterrupted stream; no calibration/test split; no forecaster retraining.")
    print("Scores: forecast-y and y-forecast; alpha=0.05 per tail; ahead=1; T_burnin=100.")
    print("ACI: author window_length=10000000; unclipped infinite thresholds are retained.")
    print(f"LQT: p_order={lqt_order}, bias=5; fixed rate; source first-100-score initialization.")
    print("LQT provenance: order 1 is empirically recovered from Table 2; order 2 is the released AMZN YAML. "
          "The paper does not explicitly identify its selected order.")
    print("COP: source ECDF, scale=0.5, proportional rate, stale trailing window of 100.")
    print("Intervals: author's minsize=0 endpoint reordering; no exceptional-set exclusions.")
    print(f"Coverage: indices 101:{len(y)} ({len(y)-101} points).")
    print(f"Primary mean/median widths: all {len(y)} points, as printed by base_test.py.")
    print("Supplemental post-burn-in widths are printed separately; infinities are not removed.")
    print("Rates: Appendix D grids; LQT adds 5, 0.5, 0.05 to the four rates in the released AMZN YAML.")
    print("Table 2 does not identify selected rates or a complete LQT grid point; every candidate is reported.")
    print("The released 3020-point cache spans 2006-2017; it is not truncated to the prose's 2006-2014.")
    if not loaded["metadata"]["matches_authors_prophet_forecasts"] or not loaded["metadata"]["matches_pinned_target_csv"]:
        print("INPUT CHECK: these inputs differ from the pinned reference fingerprints; numeric agreement is not assured.")
    print("\nPublished Table 2 references: method coverage_percent mean_width median_width")
    for method in chosen:
        ref = TABLE2[method]
        print(f"{method.upper():<6} {ref['coverage_percent']:.1f} {_number(ref['mean_width'],2)} {ref['median_width']:.2f}")
    print("\nSource-protocol results (widths include all timestamps):")
    print("method lr coverage_pct mean_width median_width delta_cov_pp delta_mean delta_median rounded_Table2_match")
    scores = (point-y, y-point)
    output = []
    for method in chosen:
        started = time.perf_counter()
        function = {"aci": aci, "lqt": LQT, "cop": COP}[method]
        kwargs = {"window_length": 10000000} if method == "aci" else (
            {"p_order": lqt_order, "bias": 5.} if method == "lqt" else {})
        for rate in LEARNING_RATES[method]:
            thresholds = [function(score, ALPHA/2, rate, T_burnin=BURNIN, ahead=1, **kwargs)["q"]
                          for score in scores]
            lower, upper, crossing = author_intervals(point, thresholds[0], thresholds[1])
            metrics = evaluate_author_metrics(y, lower, upper)
            ref = TABLE2[method]
            match = matches_table2(metrics, ref)
            row = {"method": method, "learning_rate": rate, **metrics,
                   "num_reordered_crossings": int(crossing.sum()), "matches_table2_rounding": match}
            output.append(row)
            print(f"{method.upper():<6} {rate:<7g} {_number(metrics['coverage_percent']):>12} "
                  f"{_number(metrics['mean_width']):>12} {_number(metrics['median_width']):>12} "
                  f"{_delta(metrics['coverage_percent'],ref['coverage_percent']):>12} "
                  f"{_delta(metrics['mean_width'],ref['mean_width']):>12} "
                  f"{_delta(metrics['median_width'],ref['median_width']):>12} "
                  f"{'YES' if match else 'no'}", flush=True)
        print(f"{method.upper()} elapsed_seconds={time.perf_counter()-started:.3f}", flush=True)
    print("\nSupplemental diagnostics: method lr mean_width_after101 median_width_after101 infinite_widths_all "
          "infinite_widths_after101 reordered_crossings_all")
    for row in output:
        print(f"{row['method'].upper():<6} {row['learning_rate']:<7g} "
              f"{_number(row['post_burnin_mean_width']):>12} {_number(row['post_burnin_median_width']):>12} "
              f"{row['num_infinite_widths']:>5} {row['num_infinite_widths_post_burnin']:>5} "
              f"{row['num_reordered_crossings']:>5}")
    print("\nAgreement at the paper's displayed precision (diagnostic comparison, not hyperparameter selection):")
    for method in chosen:
        matching = [row["learning_rate"] for row in output if row["method"] == method and row["matches_table2_rounding"]]
        if matching:
            print(f"{method.upper()}: matching learning rates {matching}.")
        else:
            print(f"{method.upper()}: no matching candidate in the released setup / paper-listed rate grid.")
            if method == "lqt":
                print(f"LQT p={lqt_order},bias=5 did not match Table 2 on these inputs. "
                      "The paper does not explicitly document its selected order.")
    return output
