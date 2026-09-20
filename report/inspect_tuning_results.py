"""Inspect a tuning_results.pkl or an individual trial's result.pkl.

Examples:
    python view/inspect_tuning_results.py results/tuning/example/tuning_results.pkl
    python view/inspect_tuning_results.py results/tuning/example/tuning_results.pkl --all
    python view/inspect_tuning_results.py results/tuning/example/trial_0001/result.pkl
"""

import argparse
import pickle
from pathlib import Path
from pprint import pformat


def format_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_trial(trial, config_path, rank=None):
    result = trial["result"]
    prefix = f"Rank {rank}: " if rank is not None else ""
    print(f"\n{prefix}Trial {trial['trial_index']}")
    eligible = result.get("coverage_eligible", result.get("positive_delta_coverage"))
    status = "unknown" if eligible is None else ("passed" if eligible else "failed")
    print(f"  Coverage filter: {status}")
    print(f"  Mean Winkler score: {format_value(result.get('selection_score'))}")
    print(f"  Config: {config_path}")
    print("  Hyperparameters:")
    for line in pformat(trial.get("grid_values", {}), sort_dicts=True).splitlines():
        print(f"    {line}")

    metrics = result.get("pair_metrics", {})
    if metrics:
        columns = [
            ("Coverage", "avg_coverage"),
            ("Target", "target_coverage"),
            ("Gap", "avg_delta_coverage"),
            ("Width", "avg_interval_width"),
            ("Winkler", "avg_winkler_score"),
        ]
        pair_width = max(15, *(len(str(pair)) for pair in metrics))
        print("  Metrics (averaged across evaluated sequences):")
        print(f"    {'Quantile pair':<{pair_width}}" +
              "".join(f"{label:>13}" for label, _ in columns))
        for pair, values in metrics.items():
            print(f"    {str(pair):<{pair_width}}" +
                  "".join(f"{format_value(values.get(key)):>13}" for _, key in columns))


def inspect_results(path, show_all=False):
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Expected a tuning result dictionary.")

    is_summary = "top_trials" in payload and "all_trials" in payload
    if is_summary:
        trials = payload["all_trials"] if show_all else payload["top_trials"]
        if not isinstance(trials, list):
            raise ValueError("Expected a list of trial records.")
    elif "trial_index" in payload and "result" in payload:
        trials = [payload]
    else:
        raise ValueError("Expected tuning_results.pkl or a trial's result.pkl.")

    for trial in trials:
        if (not isinstance(trial, dict)
                or not isinstance(trial.get("trial_index"), int)
                or not isinstance(trial.get("result"), dict)):
            raise ValueError("Invalid trial record: expected trial_index and result.")

    print(f"File: {path}")
    for label, key in [
        ("Method", "method"),
        ("Evaluation split", "evaluation_split"),
        ("Final test evaluated", "final_test_evaluated"),
        ("Sequences", "sequence_keys"),
        ("Coverage gap threshold", "delta_threshold"),
    ]:
        if key in payload:
            value = payload[key]
            if key == "delta_threshold" and value is None:
                value = "disabled"
            print(f"{label}: {format_value(value)}")

    if is_summary:
        print(f"Total trials: {payload.get('num_trials', len(payload['all_trials']))}")
        eligible_count = payload.get(
            "num_eligible_trials", payload.get("num_positive_delta_coverage_trials")
        )
        print(f"Passed coverage filter: {format_value(eligible_count)}")
        if payload.get("selection_message"):
            print(payload["selection_message"])
        if show_all:
            trials = sorted(trials, key=lambda trial: trial["trial_index"])
            print(f"Showing all {len(trials)} trials in trial-index order.")
        else:
            print(f"Showing {len(trials)} saved top trials, best first (lower Winkler is better).")
            if not trials:
                print("No trials passed the coverage filter. Use --all to inspect every trial.")

    for position, trial in enumerate(trials, start=1):
        trial_dir = path.parent / f"trial_{trial['trial_index']:04d}" if is_summary else path.parent
        rank = position if is_summary and not show_all else None
        print_trial(trial, trial_dir / "resolved_config.yaml", rank=rank)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_file", type=Path, help="Path to tuning_results.pkl or a trial's result.pkl.")
    parser.add_argument("--all", action="store_true", help="Show all trials, including those failing coverage.")
    args = parser.parse_args()
    try:
        inspect_results(args.results_file.expanduser().resolve(), show_all=args.all)
    except ImportError as exc:
        parser.exit(1, f"Error: {exc}. Run this script in the Python environment used for tuning.\n")
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, TypeError, ValueError) as exc:
        parser.exit(1, f"Error reading {args.results_file}: {exc}\n")


if __name__ == "__main__":
    main()
