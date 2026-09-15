"""Tune ResCP on the chronological validation region, reserving final test."""

import argparse
import itertools
from numbers import Integral
from pathlib import Path
import pickle
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from omegaconf import OmegaConf

from sbatch.sbatch_run_tuning.common import (
    choose_sequence_keys,
    finalize_and_save_results,
    load_grid,
    plain_config,
    write_trial_artifacts,
)


ALLOWED_GRID_KEYS = frozenset({
    "model.reservoir_size", "model.spectral_radius", "model.leak_rate",
    "model.input_scaling", "model.connectivity", "model.temperature",
    "model.calibration_size", "model.sampling_num", "model.use_beta_search",
    "model.beta_bins", "model.decay", "model.decay_rate", "model.recurrence",
})


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be nonnegative.")
    return parsed


def _seed(value):
    parsed = int(value)
    if not 0 <= parsed < 2**32:
        raise argparse.ArgumentTypeError("Seed must be an integer in [0, 2**32).")
    return parsed


def load_data(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Tune ResCP using validation coverage and mean Winkler score. "
            "Export the best config for a separate final-test run."
        )
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--grid-config", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--sequence-key", default=None)
    parser.add_argument("--sequence-index", type=_nonnegative_int, default=0)
    parser.add_argument("--top-k", type=_positive_int, default=3)
    parser.add_argument("--seed", type=_seed, default=None,
                        help="Override the base seed; use the same seed for every grid candidate.")
    parser.add_argument("--num-cores", "--num_cores", type=_positive_int, default=1)
    return parser


def iter_trial_configs(base_config, grid):
    """Update candidates before resolving references such as ${model.temperature}."""
    unexpected = sorted(set(grid).difference(ALLOWED_GRID_KEYS))
    if unexpected:
        raise ValueError(
            "ResCP grid may only vary model settings {}. Unsupported keys: {}. "
            "Keep the artifact, splits, target quantiles and seed fixed in the base config.".format(
                sorted(ALLOWED_GRID_KEYS), unexpected
            )
        )
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=False))
        grid_values = dict(zip(keys, values))
        for dotted_key, value in grid_values.items():
            OmegaConf.update(config, dotted_key, value, merge=False)
        yield config, grid_values


def evaluate_sequences(data, config, split, num_cores=1):
    """Defer the model import so CLI inspection does not initialize the estimator."""
    from baselines.rescp.run_rescp import evaluate_sequences as evaluate

    return evaluate(data, config, split=split, num_cores=num_cores)


def aggregate_validation_results(log, target_quantiles, delta_threshold=-0.01):
    """Weight sequences and confidence pairs equally, independent of sequence length."""
    if not log:
        raise ValueError("No validation sequence results were supplied.")
    sequence_results = {}
    for key, item in log.items():
        pair_metrics = {}
        eligible = True
        scores = []
        for pair in target_quantiles:
            pair_key = tuple(float(value) for value in pair)
            result = item["evaluation_results"][pair_key]
            metrics = {}
            for name in ("coverage", "interval_width", "winkler_score"):
                values = np.asarray(result[name], dtype=float)
                if values.size == 0 or not np.isfinite(values).all():
                    raise ValueError("{} has empty or nonfinite validation {}.".format(key, name))
                metrics["avg_" + name] = float(values.mean())
            metrics["target_coverage"] = max(pair_key) - min(pair_key)
            metrics["avg_delta_coverage"] = metrics["avg_coverage"] - metrics["target_coverage"]
            if delta_threshold is not None:
                eligible = eligible and metrics["avg_delta_coverage"] > delta_threshold
            pair_metrics[str(pair_key)] = metrics
            scores.append(metrics["avg_winkler_score"])
        if not pair_metrics:
            raise ValueError("At least one target quantile pair is required.")
        sequence_results[key] = {
            "pair_metrics": pair_metrics,
            "selection_score": float(np.mean(scores)),
            "coverage_eligible": bool(eligible),
            "metadata": item.get("metadata", {}),
        }

    first_result = next(iter(sequence_results.values()))
    pair_metrics = {
        pair_key: {
            metric: float(np.mean([
                result["pair_metrics"][pair_key][metric]
                for result in sequence_results.values()
            ]))
            for metric in metrics
        }
        for pair_key, metrics in first_result["pair_metrics"].items()
    }
    eligible = all(result["coverage_eligible"] for result in sequence_results.values())
    return {
        "evaluation_split": "validation",
        "num_sequences_evaluated": len(sequence_results),
        "sequence_results": sequence_results,
        "pair_metrics": pair_metrics,
        "selection_score": float(np.mean([
            result["selection_score"] for result in sequence_results.values()
        ])),
        "coverage_eligible": eligible,
        "positive_delta_coverage": eligible,
    }


def run_tuning(base_config_path, grid_config_path, save_dir, *, sequence_key=None,
               sequence_index=0, top_k=3, seed=None, num_cores=1):
    if isinstance(num_cores, bool) or not isinstance(num_cores, Integral) or num_cores < 1:
        raise ValueError("num_cores must be a positive integer.")
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    base_config_path = Path(base_config_path).resolve()
    grid_config_path = Path(grid_config_path).resolve()
    save_dir = Path(save_dir).resolve()
    base_config = OmegaConf.load(base_config_path)
    seed = base_config.get("seed", 2026) if seed is None else seed
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    base_config.seed = int(seed)
    artifact = Path(str(base_config.data.data_path)).expanduser()
    base_config.data.data_path = str(
        artifact.resolve() if artifact.is_absolute() else (REPO_ROOT / artifact).resolve()
    )
    grid, tuning_config = load_grid(grid_config_path)
    # Validate grid keys before loading potentially large forecast artifacts.
    first_candidate = next(iter_trial_configs(base_config, grid))
    del first_candidate
    delta_threshold = tuning_config.get("delta_threshold", -0.01)
    if delta_threshold is not None:
        delta_threshold = float(delta_threshold)
        if not np.isfinite(delta_threshold):
            raise ValueError("tuning.delta_threshold must be finite or null.")
    base_config.tuning = tuning_config

    data = load_data(base_config.data.data_path)
    num_sequences = tuning_config.get("num_sequences", 1)
    if num_sequences is None:
        num_sequences = len(data) - sequence_index
    if int(num_sequences) != num_sequences or num_sequences < 1:
        raise ValueError("tuning.num_sequences must be a positive integer or null for all.")
    sequence_keys = choose_sequence_keys(data, sequence_key, sequence_index, int(num_sequences))
    selected_data = {key: data[key] for key in sequence_keys}
    target_quantiles = OmegaConf.to_container(base_config.model.target_quantiles, resolve=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    trials = []
    for trial_index, (config, grid_values) in enumerate(iter_trial_configs(base_config, grid), start=1):
        print("[rescp] validation trial {}: {}".format(trial_index, grid_values), flush=True)
        config.saving_dir = str(save_dir / "trial_{:04d}".format(trial_index))
        log = evaluate_sequences(selected_data, config, split="validation", num_cores=num_cores)
        result = aggregate_validation_results(log, target_quantiles, delta_threshold)
        record = {
            "trial_index": trial_index,
            "sequence_keys": sequence_keys,
            "grid_values": grid_values,
            "result": result,
            "resolved_config": plain_config(config),
        }
        trials.append(record)
        write_trial_artifacts(
            save_dir, trial_index, OmegaConf.create(record["resolved_config"]), record
        )

    eligible_trials = [trial for trial in trials if trial["result"]["coverage_eligible"]]
    ranked_trials = sorted(eligible_trials, key=lambda trial: trial["result"]["selection_score"])
    best_config_path = None
    if ranked_trials:
        best_config = OmegaConf.create(ranked_trials[0]["resolved_config"])
        best_config.saving_dir = str(save_dir / "final_test")
        best_config_path = save_dir / "best_config.yaml"
        OmegaConf.save(config=best_config, f=best_config_path, resolve=True)
    else:
        # A repeated search must not leave an earlier selection looking current.
        (save_dir / "best_config.yaml").unlink(missing_ok=True)
    payload = {
        "method": "rescp",
        "evaluation_split": "validation",
        "base_config_path": str(base_config_path),
        "grid_config_path": str(grid_config_path),
        "sequence_keys": sequence_keys,
        "num_sequences": len(sequence_keys),
        "num_cores": int(num_cores),
        "seed": int(base_config.seed),
        "delta_threshold": delta_threshold,
        "coverage_filter_enabled": delta_threshold is not None,
        "num_trials": len(trials),
        "num_eligible_trials": len(eligible_trials),
        "num_positive_delta_coverage_trials": len(eligible_trials),
        "top_k": int(top_k),
        "top_trials": ranked_trials[:int(top_k)],
        "all_trials": trials,
        "best_config_path": None if best_config_path is None else str(best_config_path),
        "selection_status": "selected" if ranked_trials else "no_eligible_trials",
        "selection_message": (
            "Best configuration exported for a separate final-test run."
            if ranked_trials else
            "No trial passed the coverage threshold for every sequence and confidence pair. "
            "Validation results are saved; no best configuration was selected."
        ),
    }
    finalize_and_save_results(save_dir, payload)
    print(payload["selection_message"], flush=True)
    return payload


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_tuning(
        args.base_config, args.grid_config, args.save_dir,
        sequence_key=args.sequence_key, sequence_index=args.sequence_index,
        top_k=args.top_k, seed=args.seed, num_cores=args.num_cores,
    )


if __name__ == "__main__":
    main()
