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
from baselines.distmatch.model import DistMatchResidualIntervalEstimator, UPSTREAM_COMMIT
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
    pairs = target_quantiles(config)
    seed = sequence_seed(config["seed"], key)
    estimator = DistMatchResidualIntervalEstimator(
        seed=seed,
        **{name: config["model"][name] for name in MODEL_OPTIONS},
        **config["matching"],
    )
    started = time.perf_counter()
    estimator.fit(
        prepared["train_residuals"], normalize=config["data"]["normalize"],
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
    progress.update(split, 0, evaluation_size)
    for index, residual in enumerate(prepared["residuals"], 1):
        # All coverage levels share the same history. The current target is
        # supplied exactly once, only after every interval has been issued.
        intervals = estimator.predict_intervals([tuple(sorted(pair)) for pair in pairs])
        for pair in pairs:
            lower, upper, betas = intervals[tuple(sorted(pair))]
            results[pair]["lower_residual_quantile"].append(float(lower))
            results[pair]["upper_residual_quantile"].append(float(upper))
            results[pair]["selected_beta_per_tree"].append([float(beta) for beta in betas])
        estimator.observe(float(residual))
        progress.update(split, index, evaluation_size)
    evaluation_seconds = time.perf_counter() - started

    targets = torch.as_tensor(prepared["y"], dtype=torch.float64)
    predictions = torch.as_tensor(prepared["predictions"], dtype=torch.float64)
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
            "target_indices": prepared["target_indices"].tolist(),
            "coverage": compute_coverage(upper, lower, targets),
            "interval_width": compute_interval_width(upper, lower),
            # Beta-selected bounds do not retain the configured tail levels.
            # Use the standard 2/alpha Winkler penalty for that case.
            "winkler_score": compute_winkler_score(
                hi, lo, targets, predictions,
                alpha if config["model"]["use_beta_search"] else pair,
            ),
            "target_coverage": coverage,
        })
        result["avg_coverage"] = float(np.mean(result["coverage"]))
        result["avg_delta_coverage"] = result["avg_coverage"] - coverage
        result["avg_interval_width"] = float(np.mean(result["interval_width"]))
        result["avg_winkler_score"] = float(np.mean(result["winkler_score"]))
    return {
        "evaluation_results": results,
        "metadata": {
            "split": split,
            "boundaries": prepared["boundaries"],
            "target_indices": prepared["target_indices"].tolist(),
            "seed": config["seed"],
            "sequence_seed": seed,
            "upstream_commit": UPSTREAM_COMMIT,
            "input_mean": float(estimator.input_mean),
            "input_std": float(estimator.input_std),
            "normalize": config["data"]["normalize"],
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


def run_distmatch(config_path, num_cores=None):
    """Evaluate final test and save DSCP-compatible results and resolved config."""
    config = _resolve_paths(validate_config(OmegaConf.load(config_path), num_cores=num_cores))
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
    elapsed = time.perf_counter() - started
    pairs = target_quantiles(config)
    summary = summarize_evaluation_results(log, pairs)
    _save_pickle(output_dir / "log.pkl", log)
    _save_pickle(output_dir / "summary_results.pkl", summary)
    metadata = {
        "elapsed_seconds": elapsed,
        "num_sequences": len(data),
        "configured_num_cores": config["num_cores"],
        "effective_num_cores": min(config["num_cores"], len(data)),
        "threads_per_worker": config["threads_per_worker"],
        "upstream_commit": UPSTREAM_COMMIT,
        "package_versions": {name: version(name) for name in (
            "numpy", "scipy", "scikit-learn", "sklearn-quantile", "torch", "omegaconf"
        )},
    }
    OmegaConf.save(config=OmegaConf.create(metadata), f=output_dir / "run_metadata.yaml")
    for pair, result in summary.items():
        print(
            f"{pair}: coverage={result['avg_coverage_mean']:.4f}, "
            f"width={result['avg_interval_width_mean']:.6g}, "
            f"Winkler={result['avg_winkler_score_mean']:.6g}", flush=True,
        )
    if config.get("plotting", {}).get("plotting", False):
        import matplotlib
        matplotlib.use("Agg")
        from utils.plotting import plot_cp_prediction_intervals

        length = positive_integer(config["plotting"].get("plotting_seq_len", 200), "plotting.plotting_seq_len")
        plot_cp_prediction_intervals(log, pairs, length, str(output_dir / "plots"))
    return log
