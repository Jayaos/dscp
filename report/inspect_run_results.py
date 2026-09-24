"""Summarize one method run from its saved log.pkl and resolved_config.yaml.

Example:
    python report/inspect_run_results.py results/rescp/air/lstm/run --rolling-window-size 100

Each sequence receives equal weight. Standard deviations are population
standard deviations (ddof=0) across sequence means. Rolling coverage uses
complete, overlapping windows with stride one, averaged within each sequence
before aggregation across sequences. Delta means coverage minus the nominal
target (upper quantile minus lower quantile), including for rolling coverage.
Rolling undercoverage averages max(target - window coverage, 0) within each
sequence, then reports mean/std across those per-sequence averages.
Sequences shorter than the window contribute only to the non-rolling metrics.
"""

import argparse
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
import pickle
from pprint import pformat
import re


METRICS = (
    ("Coverage", "coverage"),
    ("Interval width", "interval_width"),
    ("Winkler score", "winkler_score"),
    ("Delta coverage", "delta_coverage"),
    ("Rolling coverage", "rolling_coverage"),
    ("Delta rolling coverage", "delta_rolling_coverage"),
    ("Rolling undercoverage", "rolling_undercoverage"),
)

METHOD_NAMES = {
    "split_cp": "SplitCP", "splitcp": "SplitCP", "rescp": "ResCP",
    "distmatch": "DistMatch", "hopcpt": "HopCPT", "nexcp": "NexCP",
    "kowcpi": "KOWCPI", "local_cp": "Local-CP", "lcp": "Local-CP",
    "iqn_cp": "IQN-CP", "iqn": "IQN-CP", "qr_cp": "QR-CP",
    "qr": "QR-CP", "spci": "SPCI",
}


def _validate_window(window):
    if isinstance(window, bool) or not isinstance(window, Integral) or window <= 0:
        raise ValueError("rolling_window_size must be a positive integer.")


def _metric_array(result, name, context, allow_empty=False):
    import numpy as np

    if name not in result:
        raise ValueError(f"{context}: missing per-timestep {name} values.")
    values = np.asarray(result[name], dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or (not values.size and not allow_empty):
        raise ValueError(f"{context}: {name} must be a nonempty 1-D array or column vector.")
    if np.isnan(values).any():
        raise ValueError(f"{context}: {name} contains NaN values.")
    if name == "coverage" and not np.isin(values, [0, 1]).all():
        raise ValueError(f"{context}: coverage must contain only booleans or 0/1 values.")
    return values


def _prediction_exclusion_counts(metadata, context):
    """Validate exclusions for supported methods without changing other logs."""
    import numpy as np

    if not isinstance(metadata, Mapping):
        return None
    if metadata.get("method") == "distmatch":
        method = "DistMatch"
    elif metadata.get("method") == "qr_cp" and metadata.get("head_type") == "independent":
        method = "QR-CP"
    else:
        return None
    counts = {}
    for name in ("total_points", "evaluated_points", "excluded_points_count"):
        value = metadata.get(name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{context}: {method} {name} must be a nonnegative integer.")
        counts[name] = int(value)
    if counts["evaluated_points"] + counts["excluded_points_count"] != counts["total_points"]:
        raise ValueError(f"{context}: {method} evaluated and excluded point counts do not sum to total_points.")
    mask = np.asarray(metadata.get("valid_prediction_mask"))
    if (mask.ndim != 1 or mask.dtype != np.bool_ or len(mask) != counts["total_points"]
            or int(mask.sum()) != counts["evaluated_points"]):
        raise ValueError(f"{context}: {method} valid_prediction_mask disagrees with point counts.")
    rate = metadata.get("exclusion_rate")
    expected_rate = counts["excluded_points_count"] / counts["total_points"] if counts["total_points"] else 0.0
    if (isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not np.isfinite(rate) or not np.isclose(rate, expected_rate, rtol=1e-12, atol=1e-12)):
        raise ValueError(f"{context}: {method} exclusion_rate disagrees with point counts.")
    no_valid = metadata.get("evaluation_status") == "no_valid_predictions"
    if no_valid != (counts["evaluated_points"] == 0):
        raise ValueError(f"{context}: {method} evaluation_status disagrees with evaluated_points.")
    return counts


def _mean_std(values):
    import numpy as np

    if not values:
        return None, None
    # Infinite intervals are valid for small SplitCP calibration sets.
    # Preserve their means; dispersion involving infinity is undefined.
    with np.errstate(invalid="ignore", over="ignore"):
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
    return (None if np.isnan(mean) else mean, None if np.isnan(std) else std)


def summarize_results(log, rolling_window_size):
    """Return per-quantile-pair mean/std metrics across the saved sequences.

    Read the stored per-timestep scores, retaining each method's original
    Winkler scoring and normalization conventions. Cached averages and the
    top-level ``summary_results`` entry are not used.
    """
    _validate_window(rolling_window_size)
    import numpy as np

    if not isinstance(log, Mapping) or not log:
        raise ValueError("Expected a nonempty sequence dictionary in log.pkl.")
    per_pair = {}
    point_counts = {}
    expected_pairs = None
    for key, item in log.items():
        if key == "summary_results":
            continue
        if not isinstance(item, Mapping) or not isinstance(item.get("evaluation_results"), Mapping):
            raise ValueError(f"Sequence {key!r}: missing evaluation_results dictionary.")
        evaluations = item["evaluation_results"]
        if not evaluations:
            raise ValueError(f"Sequence {key!r}: evaluation_results is empty.")
        if expected_pairs is None:
            expected_pairs = set(evaluations)
        elif set(evaluations) != expected_pairs:
            raise ValueError(f"Sequence {key!r}: quantile pairs differ between sequences.")
        exclusion_counts = _prediction_exclusion_counts(item.get("metadata"), f"Sequence {key!r}")
        no_valid = exclusion_counts is not None and exclusion_counts["evaluated_points"] == 0

        for pair, result in evaluations.items():
            context = f"Sequence {key!r}, quantile pair {pair!r}"
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"{context}: expected a tuple of two quantile levels.")
            lower, upper = sorted(float(level) for level in pair)
            if not (np.isfinite([lower, upper]).all() and 0 <= lower < upper <= 1):
                raise ValueError(f"{context}: invalid quantile levels.")
            if not isinstance(result, Mapping):
                raise ValueError(f"{context}: expected a metric dictionary.")
            target = upper - lower
            arrays = {name: _metric_array(result, name, context, allow_empty=no_valid)
                      for name in ("coverage", "interval_width", "winkler_score")}
            coverage = arrays["coverage"]
            if any(len(values) != len(coverage) for values in arrays.values()):
                raise ValueError(f"{context}: metric arrays have different lengths.")
            if exclusion_counts is not None and len(coverage) != exclusion_counts["evaluated_points"]:
                raise ValueError(f"{context}: metric length disagrees with evaluated_points.")
            values = per_pair.setdefault(pair, {name: [] for _, name in METRICS})
            counts = point_counts.setdefault(pair, {
                "has_exclusions": False, "num_total_sequences": 0, "num_no_valid_sequences": 0,
                "total_points": 0, "evaluated_points": 0, "excluded_points_count": 0,
            })
            counts["has_exclusions"] |= exclusion_counts is not None
            counts["num_total_sequences"] += 1
            counts["num_no_valid_sequences"] += int(no_valid)
            for name in ("total_points", "evaluated_points", "excluded_points_count"):
                counts[name] += (exclusion_counts[name] if exclusion_counts is not None
                                 else 0 if name == "excluded_points_count" else len(coverage))
            if no_valid:
                continue
            for name, array in arrays.items():
                values[name].append(float(np.mean(array)))
            values["delta_coverage"].append(values["coverage"][-1] - target)

            if len(coverage) >= rolling_window_size:
                cumulative = np.concatenate(([0.0], np.cumsum(coverage)))
                rolling = (cumulative[rolling_window_size:] - cumulative[:-rolling_window_size])
                rolling = rolling / rolling_window_size
                rolling_mean = float(np.mean(rolling))
                values["rolling_coverage"].append(rolling_mean)
                values["delta_rolling_coverage"].append(rolling_mean - target)
                # Apply the positive part per window, before either average.
                values["rolling_undercoverage"].append(float(np.mean(np.maximum(target - rolling, 0.0))))

    if not per_pair:
        raise ValueError("No sequence evaluation results found in log.pkl.")
    summary = {}
    for pair, values in per_pair.items():
        result = {
            "target_coverage": float(max(pair) - min(pair)),
            "num_sequences": len(values["coverage"]),
            "num_rolling_sequences": len(values["rolling_coverage"]),
        }
        for _, name in METRICS:
            result[f"avg_{name}_mean"], result[f"avg_{name}_std"] = _mean_std(values[name])
        counts = point_counts[pair]
        if counts["has_exclusions"]:
            result.update({name: value for name, value in counts.items() if name != "has_exclusions"})
            result["exclusion_rate"] = (counts["excluded_points_count"] / counts["total_points"]
                                        if counts["total_points"] else 0.0)
        summary[pair] = result
    return summary


def _path_label(paths, aliases):
    """Recognize whole path/name tokens, including underscore-separated names."""
    for path in paths:
        for token, label in aliases.items():
            if re.search(r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])", str(path).lower()):
                return label
    return None


def _method_identity(config, model, records, paths):
    explicit = config.get("method") or config.get("model_name")
    if not explicit:
        methods = {item.get("metadata", {}).get("method") for item in records.values()}
        methods.discard(None)
        if methods:
            explicit = ", ".join(sorted(map(str, methods)))
    if explicit:
        return str(explicit)
    signatures = (
        ({"reservoir_size"}, "ResCP"),
        ({"match_threshold", "past_window_len"}, "DistMatch"),
        ({"dim_hopfield_hidden"}, "HopCPT"),
        ({"rho", "max_past"}, "NexCP"),
        ({"kernel", "past_window"}, "KOWCPI"),
        ({"similarity_fn", "calibration_size"}, "Local-CP"),
        ({"prediction_head"}, "IQN-CP"),
        ({"cos_emb_dim"}, "IQN-CP"),
        ({"head_type", "dim_model"}, "QR-CP"),
        ({"n_estimators", "window_size"}, "SPCI"),
    )
    for keys, name in signatures:
        if keys.issubset(model):
            return f"{name} (inferred from model configuration)"
    name = _path_label(paths, METHOD_NAMES)
    return f"{name} (inferred from results path)" if name else "n/a (not saved)"


def _print_identity(results_dir, config, config_path, log):
    records = {key: item for key, item in log.items() if key != "summary_results"}
    saved_models = {key: item["model_config"] for key, item in records.items() if item.get("model_config")}
    model = config.get("model", {}) or next(iter(saved_models.values()), {})
    data = config.get("data", {})
    paths = [config.get("saving_dir", ""), results_dir]
    data_path = data.get("data_path")
    predictor = config.get("base_predictor") or data.get("base_predictor")
    dataset = config.get("dataset") or data.get("dataset")
    if data_path:
        # Recognize the standard <predictor>_<dataset>_data.pkl artifact on
        # either Windows or POSIX, without requiring the original data file.
        stem = Path(str(data_path).replace("\\", "/")).stem
        if stem.endswith("_data") and "_" in stem[:-5]:
            artifact_predictor, artifact_dataset = stem[:-5].split("_", 1)
            predictor = predictor or artifact_predictor
            dataset = dataset or artifact_dataset
    identity_paths = ([data_path] if data_path else []) + paths
    predictor = predictor or _path_label(identity_paths, {name: name for name in ("lr", "lstm", "chronos")})
    dataset = dataset or _path_label(identity_paths, {
        "air": "air", "solar": "solar", "nsdb-60m": "solar (nsdb-60m)", "sapflux": "sapflux",
    })

    print(f"Results directory: {results_dir}")
    print(f"Method: {_method_identity(config, model, records, paths)}")
    architecture = (f"RNN ({model['rnn_type']})" if "rnn_type" in model
                    else "Transformer" if "num_heads" in model else None)
    if architecture:
        print(f"Model architecture: {architecture}")
    print(f"Dataset: {dataset or 'n/a (not saved)'}")
    print(f"Base predictor: {predictor or 'n/a (not saved)'}")
    print(f"Data artifact: {data_path or 'n/a (not saved)'}")
    print(f"Config: {config_path if config_path else 'n/a (resolved_config.yaml not found)'}")
    if config:
        print("Hyperparameters (saved configuration):")
        for line in pformat(config, sort_dicts=True).splitlines():
            print(f"  {line}")
    elif saved_models:
        print("Hyperparameters (model configurations saved in log.pkl, by sequence):")
        for line in pformat(saved_models, sort_dicts=True).splitlines():
            print(f"  {line}")
    else:
        print("Hyperparameters: n/a (not saved)")


def format_value(value):
    return "n/a" if value is None else f"{value:.6g}"


def inspect_results(results_dir, rolling_window_size):
    """Print and return a summary for one saved run directory."""
    _validate_window(rolling_window_size)
    results_dir = Path(results_dir).expanduser().resolve()
    with (results_dir / "log.pkl").open("rb") as stream:
        log = pickle.load(stream)
    summary = summarize_results(log, rolling_window_size)
    config_path = results_dir / "resolved_config.yaml"
    config = {}
    if config_path.is_file():
        import yaml

        try:
            with config_path.open(encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
        if not isinstance(config, Mapping):
            raise ValueError(f"Expected a configuration dictionary in {config_path}.")
    else:
        config_path = None

    _print_identity(results_dir, config, config_path, log)
    print(f"\nRolling coverage window size: {rolling_window_size} (full windows, stride 1)")
    print("Mean/std across equally weighted sequence means; population std (ddof=0).")
    print("Rolling metrics first average windows within each sequence.")
    print("Delta = coverage - target coverage; n/a means unavailable or undefined.")
    print("Rolling undercoverage = max(target coverage - window coverage, 0), averaged per sequence.")
    for pair, result in summary.items():
        print(f"\nQuantile pair: {pair}; target coverage: {format_value(result['target_coverage'])}")
        count = result["num_sequences"]
        rolling_count = result["num_rolling_sequences"]
        print(f"Sequences: {count}; rolling sequences: {rolling_count}/{count}")
        if "num_no_valid_sequences" in result:
            print(f"  DistMatch sequences with no valid predictions: {result['num_no_valid_sequences']}"
                  f"/{result['num_total_sequences']} (excluded from metric averages).")
            print(f"  Points: evaluated={result['evaluated_points']}, excluded={result['excluded_points_count']},"
                  f" total={result['total_points']}; exclusion rate={result['exclusion_rate']:.6%}.")
            print("  Metrics use successful predictions; rolling windows skip excluded points.")
        if rolling_count < count:
            print(f"  {count - rolling_count} sequence(s) shorter than the window excluded from rolling metrics.")
        print(f"  {'Metric':<25}{'Mean':>14}{'Std':>14}")
        for label, name in METRICS:
            mean = format_value(result[f"avg_{name}_mean"])
            std = format_value(result[f"avg_{name}_std"])
            print(f"  {label:<25}{mean:>14}{std:>14}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, help="One run directory containing log.pkl and, if saved, resolved_config.yaml.")
    parser.add_argument("--rolling-window-size", type=int, required=True, help="Positive number of evaluated timesteps per rolling coverage window.")
    args = parser.parse_args(argv)
    try:
        return inspect_results(args.results_dir, args.rolling_window_size)
    except ImportError as exc:
        parser.exit(1, f"Error: {exc}. Run this script in the Python environment used for the method run.\n")
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, TypeError, ValueError) as exc:
        parser.exit(1, f"Error reading {args.results_dir}: {exc}\n")


if __name__ == "__main__":
    main()
