"""Run DistMatch on saved point forecasts, with parallel independent sequences."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
import multiprocessing
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from omegaconf import OmegaConf

from utils.experiment_config import load_experiment_config
from threadpoolctl import threadpool_limits
import torch
from tqdm import tqdm

from baselines.distmatch.config import (
    MODEL_DEFAULTS,
    positive_integer,
    target_quantiles,
    validate_config,
)
from baselines.distmatch.data import prepare_sequence
from baselines.distmatch.exclusions import ExclusionJournal, write_excluded_points
from baselines.distmatch.model import (
    DistMatchCrossedBoundsError, DistMatchResidualIntervalEstimator, UPSTREAM_COMMIT,
)
from baselines.distmatch.progress import SequenceProgress
from utils.reporting import (
    compute_coverage,
    compute_interval_width,
    compute_winkler_score,
    construct_interval_endpoints,
    summarize_evaluation_results,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_OPTIONS = set(MODEL_DEFAULTS) - {"prediction_step", "target_quantiles"}
_PROGRESS_POSITION = 0


def sequence_seed(seed, key):
    """Stable across process counts, sequence order, and Python hash seeds."""
    identity = f"{seed}\0{type(key).__name__}\0{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "little")


def _resolve_paths(config):
    """Repository-relative paths match the checked-in experiment presets."""
    config = validate_config(config)
    for section, name in (("data", "data_path"), ("matching", "cache_dir")):
        raw = config[section].get(name)
        if raw is not None:
            path = Path(raw).expanduser()
            config[section][name] = str((path if path.is_absolute() else REPO_ROOT / path).resolve())
    raw = config.get("saving_dir")
    if raw is not None:
        path = Path(raw).expanduser()
        config["saving_dir"] = str((path if path.is_absolute() else REPO_ROOT / path).resolve())
    return config


def _evaluate_sequence(key, item, config, split, progress):
    prepared = prepare_sequence(item, config, split=split)
    normalization = prepared["normalization"]
    residual_scale = normalization["target_std"]
    pairs = target_quantiles(config)
    seed = sequence_seed(config["seed"], key)
    estimator = DistMatchResidualIntervalEstimator(
        seed=seed,
        **{name: config["model"][name] for name in MODEL_OPTIONS},
        **config["matching"],
    )
    started = time.perf_counter()
    estimator.fit(
        prepared["train_residuals"],
        progress=progress.update if config["show_progress"] else None,
    )
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    replay_size = len(prepared["warmup_residuals"])
    if replay_size:
        progress.update("replay", 0, replay_size)
    for index, residual in enumerate(prepared["warmup_residuals"], 1):
        estimator.observe(float(residual))
        progress.update("replay", index, replay_size)
    replay_seconds = time.perf_counter() - started
    initial_memory_size = estimator.memory_size
    results = {pair: {
        "lower_residual_quantile": [], "upper_residual_quantile": [],
        "selected_beta_per_tree": [],
    } for pair in pairs}
    started = time.perf_counter()
    evaluation_size = len(prepared["residuals"])
    valid_mask = np.zeros(evaluation_size, dtype=bool)
    excluded_points = []
    progress.update(split, 0, evaluation_size)
    with ExclusionJournal(config, key, split) as journal:
        for offset, residual in enumerate(prepared["residuals"]):
            # The prediction call is atomic across coverage levels. A known
            # numerical failure excludes this timestamp from every level.
            try:
                intervals = estimator.predict_intervals([tuple(sorted(pair)) for pair in pairs])
            except DistMatchCrossedBoundsError as exc:
                # Tuning must not improve a candidate's score by discarding
                # failed validation predictions.
                if split != "test":
                    raise
                event = _excluded_point(key, config, prepared, pairs, seed, offset, exc)
                excluded_points.append(event)
                journal.record(event)
            else:
                for pair in pairs:
                    lower, upper, betas = intervals[tuple(sorted(pair))]
                    # Restore artifact units for saved endpoints and metrics.
                    results[pair]["lower_residual_quantile"].append(float(lower) * residual_scale)
                    results[pair]["upper_residual_quantile"].append(float(upper) * residual_scale)
                    results[pair]["selected_beta_per_tree"].append([float(beta) for beta in betas])
                valid_mask[offset] = True
            # Failed predictions still consume their observation, exactly once.
            # This preserves the history, memory, and observation-based seeds.
            estimator.observe(float(residual))
            progress.update(split, offset + 1, evaluation_size)
    evaluation_seconds = time.perf_counter() - started

    evaluated_points = int(valid_mask.sum())
    counts = {
        "total_points": evaluation_size,
        "evaluated_points": evaluated_points,
        "excluded_points_count": len(excluded_points),
        "exclusion_rate": len(excluded_points) / evaluation_size if evaluation_size else 0.0,
    }
    status = ("no_valid_predictions" if not evaluated_points else
              "completed_with_exclusions" if excluded_points else "complete")
    targets = torch.as_tensor(prepared["y"][valid_mask], dtype=torch.float64)
    predictions = torch.as_tensor(prepared["predictions"][valid_mask], dtype=torch.float64)
    for pair, result in results.items():
        lo = torch.tensor(result["lower_residual_quantile"], dtype=torch.float64)
        hi = torch.tensor(result["upper_residual_quantile"], dtype=torch.float64)
        upper, lower = construct_interval_endpoints(hi, lo, predictions)
        if not (torch.isfinite(upper).all() and torch.isfinite(lower).all()):
            raise ValueError(f"Nonfinite prediction interval for sequence {key!r}.")
        if torch.any(lower > upper):
            raise ValueError(f"Crossed prediction interval for sequence {key!r}.")
        coverage = max(pair) - min(pair)
        alpha = 1.0 - coverage
        result.update({
            "lower_interval": lower.tolist(),
            "upper_interval": upper.tolist(),
            "target_y": targets.tolist(),
            "target_predictions": predictions.tolist(),
            "target_indices": prepared["target_indices"][valid_mask].tolist(),
            "coverage": compute_coverage(upper, lower, targets),
            "interval_width": compute_interval_width(upper, lower),
            # Beta-selected bounds do not retain the configured tail levels.
            # Use the standard 2/alpha Winkler penalty for that case.
            "winkler_score": compute_winkler_score(
                hi, lo, targets, predictions,
                alpha if config["model"]["use_beta_search"] else pair,
            ),
            "target_coverage": coverage,
            "evaluation_status": status,
            **counts,
        })
        for metric in ("coverage", "interval_width", "winkler_score"):
            result[f"avg_{metric}"] = float(np.mean(result[metric])) if evaluated_points else None
        result["avg_delta_coverage"] = result["avg_coverage"] - coverage if evaluated_points else None
    return {
        "evaluation_results": results,
        "metadata": {
            "method": "distmatch",
            "split": split,
            "boundaries": prepared["boundaries"],
            "target_indices": prepared["target_indices"].tolist(),
            "valid_prediction_mask": valid_mask.tolist(),
            "excluded_points": excluded_points,
            "evaluation_status": status,
            "exclusion_policy": "skip_crossed_bounds_timestamp_all_coverage_levels",
            "reporting_timeline": "successful_predictions_only",
            **counts,
            "seed": config["seed"],
            "sequence_seed": seed,
            "upstream_commit": UPSTREAM_COMMIT,
            "normalize_residual": config["data"]["normalize_residual"],
            "normalization": normalization,
            "interval_scale": "original_response",
            "residual_quantile_scale": "original_residual",
            "tree_structure": "fixed_after_training",
            "validation_replay_size": len(prepared["warmup_residuals"]),
            "initial_memory_size": int(initial_memory_size),
            "final_memory_size": int(estimator.memory_size),
            "training_seconds": training_seconds,
            "validation_replay_seconds": replay_seconds,
            "evaluation_seconds": evaluation_seconds,
            "diagnostics": estimator.diagnostics(),
        },
    }


def _excluded_point(key, config, prepared, pairs, seed, offset, exc):
    """Keep original data/index values and label the QRF's residual units."""
    raw_path = config["data"].get("data_path")
    artifact = Path(raw_path) if raw_path else None
    return {
        **exc.diagnostics,
        "dataset": artifact.parent.parent.name if artifact else None,
        "predictor": artifact.parent.name if artifact else None,
        "sequence_key": str(key), "seed": config["seed"], "sequence_seed": seed,
        "split": "test", "test_offset": offset,
        "target_index": int(prepared["target_indices"][offset]),
        "target_y": float(prepared["y"][offset]),
        "target_prediction": float(prepared["predictions"][offset]),
        "target_residual": float(prepared["y"][offset] - prepared["predictions"][offset]),
        "excluded_quantile_pairs": [list(pair) for pair in pairs],
        "bound_scale": "normalized_residual" if prepared["normalization"]["enabled"] else "original_residual",
        "residual_scale": float(prepared["normalization"]["target_std"]),
        "reason": str(exc), "action": "excluded_from_metrics",
    }


def evaluate_sequence(key, item, config, split="test"):
    """Fit one series and evaluate only the requested chronological region."""
    config = _resolve_paths(config)
    threads = config["threads_per_worker"]
    # Load sklearn/SciPy native pools before taking the threadpoolctl snapshot.
    # Pools imported inside the limits context would otherwise escape its cap.
    DistMatchResidualIntervalEstimator._load_qrf()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        with threadpool_limits(limits=threads), SequenceProgress(
            key, enabled=config["show_progress"], position=_PROGRESS_POSITION,
        ) as progress:
            return _evaluate_sequence(key, item, config, split, progress)
    finally:
        torch.set_num_threads(previous_threads)


def _initialize_worker_progress(lock, positions):
    """Share the output lock and reserve one stable terminal row per worker."""
    global _PROGRESS_POSITION
    tqdm.set_lock(lock)
    with positions.get_lock():
        _PROGRESS_POSITION = positions.value
        positions.value += 1


def _worker(key, item, config, split):
    try:
        return evaluate_sequence(key, item, config, split)
    except Exception as exc:
        raise RuntimeError(f"DistMatch failed for sequence {key!r}: {exc}") from exc


def evaluate_sequences(data, config, split="test", num_cores=None):
    """Use configured process count, with optional explicit caller override.

    Only sequences are parallelized. Each sequence advances through timestamps
    serially, and all native numerical thread pools are capped inside workers.
    """
    config = _resolve_paths(validate_config(config, num_cores=num_cores))
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")
    if not isinstance(data, dict) or not data:
        raise ValueError("The prediction artifact must be a nonempty dictionary of series.")
    workers = min(config["num_cores"], len(data))
    if workers == 1:
        return {key: evaluate_sequence(key, item, config, split) for key, item in data.items()}
    completed = {}
    # Explicit spawn is portable to Windows and avoids inheriting parent RNGs
    # or initialized native numerical thread pools on Linux.
    context = multiprocessing.get_context("spawn")
    lock, positions = context.RLock(), context.Value("i", 0)
    previous_lock = tqdm.get_lock()
    tqdm.set_lock(lock)
    try:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context,
            initializer=_initialize_worker_progress, initargs=(lock, positions),
        ) as executor:
            futures = {executor.submit(_worker, key, item, config, split): key
                       for key, item in data.items()}
            for future in as_completed(futures):
                key = futures[future]
                completed[key] = future.result()
                # Parent tqdm cannot clear bars owned by child processes.
                # Newlines during interactive rendering would shift their rows.
                if not sys.stdout.isatty():
                    tqdm.write(
                        f"DistMatch: completed sequence {key!r} ({len(completed)}/{len(data)})",
                        file=sys.stdout,
                    )
                    sys.stdout.flush()
        if sys.stdout.isatty():
            print(f"DistMatch: completed {len(completed)}/{len(data)} sequences", flush=True)
    finally:
        tqdm.set_lock(previous_lock)
    return {key: completed[key] for key in data}


def _save_pickle(path, value):
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _run_metadata(config, num_sequences, elapsed):
    return {
        "elapsed_seconds": elapsed,
        "num_sequences": num_sequences,
        "configured_num_cores": config["num_cores"],
        "effective_num_cores": min(config["num_cores"], num_sequences),
        "threads_per_worker": config["threads_per_worker"],
        "upstream_commit": UPSTREAM_COMMIT,
        "package_versions": {name: version(name) for name in (
            "numpy", "scipy", "scikit-learn", "sklearn-quantile", "torch", "omegaconf"
        )},
    }


def _summarize_distmatch_results(log, pairs):
    """Preserve equal sequence weighting, omitting only unavailable scores."""
    summaries = {}
    for pair in pairs:
        eligible = {key: item for key, item in log.items()
                    if item["evaluation_results"][pair]["coverage"]}
        if eligible:
            result = summarize_evaluation_results(eligible, [pair])[pair]
        else:
            result = {f"avg_{metric}_{stat}": None
                      for metric in ("coverage", "delta_coverage", "interval_width", "winkler_score")
                      for stat in ("mean", "std")}
        evaluated = sum(len(item["evaluation_results"][pair]["coverage"]) for item in log.values())
        total = sum(item.get("metadata", {}).get(
            "total_points", len(item["evaluation_results"][pair]["coverage"]),
        ) for item in log.values())
        result.update({
            "num_sequences": len(eligible), "num_total_sequences": len(log),
            "num_no_valid_sequences": len(log) - len(eligible),
            "total_points": total, "evaluated_points": evaluated,
            "excluded_points_count": total - evaluated,
            "exclusion_rate": (total - evaluated) / total if total else 0.0,
            "evaluation_status": "no_valid_predictions" if not eligible else
                                 "completed_with_exclusions" if evaluated < total else "complete",
        })
        summaries[pair] = result
    return summaries


def _write_run_results(config, log, metadata):
    """Write the same result format for ordinary runs and completed shard sets."""
    output_dir = Path(config["saving_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(config), f=output_dir / "resolved_config.yaml", resolve=True)
    pairs = target_quantiles(config)
    summary = _summarize_distmatch_results(log, pairs)
    first = summary[pairs[0]]
    metadata = {**metadata, "method": "distmatch", **{
        name: first[name] for name in (
            "total_points", "evaluated_points", "excluded_points_count", "exclusion_rate",
            "num_no_valid_sequences", "evaluation_status",
        )
    }, "exclusion_policy": "skip_crossed_bounds_timestamp_all_coverage_levels",
        "reporting_timeline": "successful_predictions_only",
        "excluded_points_file": "excluded_points.csv"}
    write_excluded_points(log, output_dir)
    _save_pickle(output_dir / "log.pkl", log)
    _save_pickle(output_dir / "summary_results.pkl", summary)
    OmegaConf.save(config=OmegaConf.create(metadata), f=output_dir / "run_metadata.yaml")
    for pair, result in summary.items():
        if result["evaluation_status"] == "no_valid_predictions":
            print(f"{pair}: no valid predictions ({result['excluded_points_count']} excluded)", flush=True)
            continue
        print(
            f"{pair}: coverage={result['avg_coverage_mean']:.4f}, "
            f"width={result['avg_interval_width_mean']:.6g}, "
            f"Winkler={result['avg_winkler_score_mean']:.6g}, "
            f"evaluated={result['evaluated_points']}/{result['total_points']}, "
            f"excluded={result['excluded_points_count']}", flush=True,
        )
    if config.get("plotting", {}).get("plotting", False):
        import matplotlib
        matplotlib.use("Agg")
        from utils.plotting import plot_cp_prediction_intervals

        length = positive_integer(config["plotting"].get("plotting_seq_len", 200), "plotting.plotting_seq_len")
        # Arrays already contain only successful timestamps. Empty sequences
        # remain in log.pkl, but have no points to plot.
        plotted = {key: item for key, item in log.items()
                   if item["evaluation_results"][pairs[0]]["coverage"]}
        plot_cp_prediction_intervals(plotted, pairs, length, str(output_dir / "plots"))
    return log


def run_distmatch(config_path, num_cores=None):
    """Evaluate final test and save DSCP-compatible results and resolved config."""
    config = _resolve_paths(validate_config(load_experiment_config(config_path), num_cores=num_cores))
    if not config["data"].get("data_path"):
        raise ValueError("data.data_path is required for an experiment run.")
    if not config.get("saving_dir"):
        raise ValueError("saving_dir is required for an experiment run.")
    artifact = Path(config["data"]["data_path"])
    with artifact.open("rb") as stream:
        data = pickle.load(stream)
    output_dir = Path(config["saving_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(config), f=output_dir / "resolved_config.yaml", resolve=True)
    print(f"DistMatch: {len(data)} sequences, {config['num_cores']} configured sequence workers", flush=True)
    started = time.perf_counter()
    log = evaluate_sequences(data, config, split="test")
    metadata = _run_metadata(config, len(data), time.perf_counter() - started)
    return _write_run_results(config, log, metadata)
