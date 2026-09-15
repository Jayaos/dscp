"""Select DistMatch settings on sequential validation, reserving final test."""

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

from baselines.distmatch.config import target_quantiles as configured_target_quantiles
from sbatch.sbatch_run_tuning.common import (
    choose_sequence_keys,
    finalize_and_save_results,
    load_grid,
    plain_config,
    write_trial_artifacts,
)


ALLOWED_GRID_KEYS = frozenset({
    "model.past_window_len", "model.match_threshold", "model.n_trees",
    "model.bagging_ratio", "model.beta_bins", "model.qrf_n_estimators",
    "model.qrf_max_depth", "model.min_samples_per_node", "model.use_beta_search",
})
PROTECTED_CONFIG_KEYS = (
    "data", "seed", "num_cores", "threads_per_worker",
    "model.target_quantiles", "model.prediction_step",
)


def _integer(value, name, minimum=1, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or value < minimum or (maximum is not None and value > maximum)):
        if maximum is not None:
            raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}].")
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer.")
    return int(value)


def _positive_int(value):
    try:
        return _integer(int(value), "Value")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _nonnegative_int(value):
    try:
        return _integer(int(value), "Value", minimum=0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _seed(value):
    try:
        return _integer(int(value), "seed", minimum=0, maximum=2**32 - 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Tune DistMatch using sequential validation coverage and mean Winkler score. "
            "Export the selected config for a separate final-test run."
        )
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--grid-config", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--sequence-key", default=None)
    parser.add_argument("--sequence-index", type=_nonnegative_int, default=0)
    parser.add_argument("--top-k", type=_positive_int, default=3)
    parser.add_argument("--seed", type=_seed, default=None,
                        help="Override the config seed; every candidate uses the same seed.")
    parser.add_argument(
        "--num-cores", "--num_cores", type=_positive_int, default=None,
        help="Override config num_cores for sequence workers; grid trials run sequentially.",
    )
    return parser


def _protected_settings(config):
    settings = {}
    for key in PROTECTED_CONFIG_KEYS:
        value = OmegaConf.select(config, key, default=None)
        settings[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return settings


def iter_trial_configs(base_config, grid):
    """Resolve model-dependent labels after updates and keep comparison settings fixed."""
    unexpected = sorted(set(grid).difference(ALLOWED_GRID_KEYS))
    if unexpected:
        raise ValueError(
            f"DistMatch grid may only vary model settings {sorted(ALLOWED_GRID_KEYS)}. "
            f"Unsupported keys: {unexpected}. Keep the artifact, splits, target quantiles, "
            "seed and worker settings fixed in the base config."
        )
    protected = _protected_settings(base_config)
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=False))
        grid_values = dict(zip(keys, values))
        for key, value in grid_values.items():
            OmegaConf.update(config, key, value, merge=False)
        if _protected_settings(config) != protected:
            raise ValueError("Grid interpolation must not change the artifact, splits, target quantiles, seed or workers.")
        yield config, grid_values


def load_data(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def evaluate_sequences(data, config, split, num_cores=None):
    """Keep CLI/grid inspection independent of the DistMatch estimator import."""
    from baselines.distmatch.run_distmatch import evaluate_sequences as evaluate

    return evaluate(data, config, split=split, num_cores=num_cores)


def aggregate_validation_results(log, target_quantiles, delta_threshold=-0.01):
    """Weight each sequence and interval equally; filter coverage before ranking."""
    if not log or not target_quantiles:
        raise ValueError("At least one validation sequence and target quantile pair are required.")
    sequence_results = {}
    for key, item in log.items():
        pair_metrics, scores, eligible = {}, [], True
        for pair in target_quantiles:
            pair_key = tuple(float(value) for value in pair)
            result = item["evaluation_results"][pair_key]
            metrics = {}
            for name in ("coverage", "interval_width", "winkler_score"):
                values = np.asarray(result[name], dtype=float)
                if values.size == 0 or not np.isfinite(values).all():
                    raise ValueError(f"{key!r} has empty or nonfinite validation {name}.")
                metrics["avg_" + name] = float(values.mean())
            metrics["target_coverage"] = max(pair_key) - min(pair_key)
            metrics["avg_delta_coverage"] = metrics["avg_coverage"] - metrics["target_coverage"]
            if delta_threshold is not None:
                eligible = eligible and metrics["avg_delta_coverage"] > delta_threshold
            pair_metrics[str(pair_key)] = metrics
            scores.append(metrics["avg_winkler_score"])
        sequence_results[key] = {
            "pair_metrics": pair_metrics,
            "selection_score": float(np.mean(scores)),
            "coverage_eligible": bool(eligible),
            "metadata": item.get("metadata", {}),
        }
    first = next(iter(sequence_results.values()))
    pair_metrics = {
        pair: {
            metric: float(np.mean([entry["pair_metrics"][pair][metric] for entry in sequence_results.values()]))
            for metric in metrics
        }
        for pair, metrics in first["pair_metrics"].items()
    }
    eligible = all(entry["coverage_eligible"] for entry in sequence_results.values())
    return {
        "evaluation_split": "validation",
        "final_test_evaluated": False,
        "num_sequences_evaluated": len(sequence_results),
        "sequence_results": sequence_results,
        "pair_metrics": pair_metrics,
        "selection_score": float(np.mean([entry["selection_score"] for entry in sequence_results.values()])),
        "coverage_eligible": eligible,
        "positive_delta_coverage": eligible,
    }


def run_tuning(base_config_path, grid_config_path, save_dir, *, sequence_key=None,
               sequence_index=0, top_k=3, seed=None, num_cores=None):
    top_k = _integer(top_k, "top_k")
    sequence_index = _integer(sequence_index, "sequence_index", minimum=0)
    base_config_path, grid_config_path = Path(base_config_path).resolve(), Path(grid_config_path).resolve()
    save_dir = Path(save_dir).resolve()
    base_config = OmegaConf.load(base_config_path)
    base_config.seed = _integer(
        base_config.get("seed", 2026) if seed is None else seed,
        "seed", minimum=0, maximum=2**32 - 1,
    )
    num_cores = _integer(base_config.get("num_cores", 1) if num_cores is None else num_cores, "num_cores")
    base_config.num_cores = num_cores
    base_config.threads_per_worker = _integer(base_config.get("threads_per_worker", 1), "threads_per_worker")
    artifact = Path(str(base_config.data.data_path)).expanduser()
    base_config.data.data_path = str(artifact.resolve() if artifact.is_absolute() else (REPO_ROOT / artifact).resolve())
    grid, tuning_config = load_grid(grid_config_path)
    # Reject unsupported settings before opening the forecast artifact.
    next(iter_trial_configs(base_config, grid))
    delta_threshold = tuning_config.get("delta_threshold", -0.01)
    if delta_threshold is not None:
        delta_threshold = float(delta_threshold)
        if not np.isfinite(delta_threshold):
            raise ValueError("tuning.delta_threshold must be finite or null.")
    base_config.tuning = tuning_config
    data = load_data(base_config.data.data_path)
    num_sequences = tuning_config.get("num_sequences", 1)
    if num_sequences is None:
        num_sequences = 1 if sequence_key is not None else len(data) - sequence_index
    num_sequences = _integer(num_sequences, "tuning.num_sequences")
    sequence_keys = choose_sequence_keys(data, sequence_key, sequence_index, num_sequences)
    selected_data = {key: data[key] for key in sequence_keys}
    target_quantiles = configured_target_quantiles(plain_config(base_config))
    save_dir.mkdir(parents=True, exist_ok=True)

    trials = []
    for trial_index, (config, grid_values) in enumerate(iter_trial_configs(base_config, grid), start=1):
        print(f"[distmatch] validation trial {trial_index}: {grid_values}", flush=True)
        config.saving_dir = str(save_dir / f"trial_{trial_index:04d}")
        log = evaluate_sequences(selected_data, config, split="validation", num_cores=num_cores)
        record = {
            "trial_index": trial_index,
            "sequence_keys": sequence_keys,
            "grid_values": grid_values,
            "evaluation_split": "validation",
            "final_test_evaluated": False,
            "result": aggregate_validation_results(log, target_quantiles, delta_threshold),
            "resolved_config": plain_config(config),
        }
        trials.append(record)
        write_trial_artifacts(save_dir, trial_index, OmegaConf.create(record["resolved_config"]), record)

    eligible_trials = [trial for trial in trials if trial["result"]["coverage_eligible"]]
    ranked = sorted(eligible_trials, key=lambda trial: trial["result"]["selection_score"])
    best_config_path = save_dir / "best_config.yaml"
    if ranked:
        best_config = OmegaConf.create(ranked[0]["resolved_config"])
        best_config.saving_dir = str(save_dir / "final_test")
        OmegaConf.save(config=best_config, f=best_config_path, resolve=True)
    else:
        best_config_path.unlink(missing_ok=True)
    payload = {
        "method": "distmatch",
        "evaluation_split": "validation",
        "final_test_evaluated": False,
        "final_test_protocol": "fresh_initial_train_fit_then_validation_replay",
        "base_config_path": str(base_config_path),
        "grid_config_path": str(grid_config_path),
        "sequence_keys": sequence_keys,
        "num_sequences": len(sequence_keys),
        "num_cores": num_cores,
        "threads_per_worker": int(base_config.threads_per_worker),
        "seed": int(base_config.seed),
        "delta_threshold": delta_threshold,
        "coverage_filter_enabled": delta_threshold is not None,
        "num_trials": len(trials),
        "num_eligible_trials": len(eligible_trials),
        "num_positive_delta_coverage_trials": len(eligible_trials),
        "top_k": top_k,
        "top_trials": ranked[:top_k],
        "all_trials": trials,
        "best_config_path": str(best_config_path) if ranked else None,
        "selection_status": "selected" if ranked else "no_eligible_trials",
        "selection_message": (
            "Best configuration exported for a separate final-test run."
            if ranked else
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
