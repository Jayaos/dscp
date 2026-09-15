"""Run ResCP on saved point forecasts using DSCP's evaluation/logging contracts."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import math
from numbers import Integral
from pathlib import Path
import pickle
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from baselines.rescp.data import prepare_sequence
from baselines.rescp.model import ResCPResidualIntervalEstimator
from utils.reporting import (
    compute_coverage,
    compute_interval_width,
    compute_winkler_score,
    construct_interval_endpoints,
    summarize_evaluation_results,
)
from utils.utils import save_data


UPSTREAM_COMMIT = "1d8e560b77890ee1fc7acad591d33b7b3e4b694f"
MODEL_OPTIONS = {
    "reservoir_size", "spectral_radius", "leak_rate", "input_scaling",
    "connectivity", "temperature", "calibration_size", "sampling_num",
    "use_beta_search", "beta_bins", "decay", "decay_rate", "recurrence",
}


def _config_dict(config):
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return config


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def target_quantiles(config):
    model = _config_dict(config).get("model", {})
    pairs = model.get("target_quantiles", [[0.05, 0.95]])
    if not isinstance(pairs, (list, tuple)) or len(pairs) == 0:
        raise ValueError("model.target_quantiles must be a nonempty list of quantile pairs.")
    result, seen = [], set()
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each target_quantiles entry must contain two quantile levels.")
        pair = tuple(float(level) for level in pair)
        lower, upper = sorted(pair)
        if not all(math.isfinite(level) for level in pair) or not 0 < lower < upper < 1:
            raise ValueError("Quantile levels must satisfy 0 < lower < upper < 1.")
        if (lower, upper) in seen:
            raise ValueError("model.target_quantiles must not contain duplicate intervals.")
        seen.add((lower, upper))
        result.append(pair)
    return result


def _sampling_seed(seed, key):
    # Python's hash is process-randomized. A stable key keeps worker counts and
    # the number/order of other series from changing this series' samples.
    identity = f"{seed}\0{type(key).__name__}\0{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "little")


def _evaluate_sequence(key, item, config, split):
    model_config = config.get("model", {})
    step = model_config.get("prediction_step", 1)
    if isinstance(step, bool) or not isinstance(step, Integral) or step != 1:
        raise ValueError("ResCP currently supports model.prediction_step=1 only.")
    unknown = set(model_config) - MODEL_OPTIONS - {"target_quantiles", "prediction_step"}
    if unknown:
        raise ValueError(f"Unknown ResCP model settings: {sorted(unknown)}")
    seed = config.get("seed", 2026)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    normalize = config["data"].get("normalize", True)
    if not isinstance(normalize, bool):
        raise ValueError("data.normalize must be a boolean.")
    pairs = target_quantiles(config)
    prepared = prepare_sequence(item, config, split=split)
    options = {name: value for name, value in model_config.items() if name in MODEL_OPTIONS}
    sampling_seed = _sampling_seed(seed, key)
    estimator = ResCPResidualIntervalEstimator(seed=seed, sampling_seed=sampling_seed, **options)
    start_time = time.perf_counter()
    estimator.fit(prepared["calibration_residuals"], normalize=normalize)
    # Validation is available history at the first test timestamp. Replaying it
    # does not refit the scaler or consume any sampling RNG draws.
    for residual in prepared["warmup_residuals"]:
        estimator.observe(float(residual))
    initial_memory_size = estimator.memory_size
    initialization_seconds = time.perf_counter() - start_time

    results = {
        pair: {"lower_residual_quantile": [], "upper_residual_quantile": [], "selected_beta": []}
        for pair in pairs
    }
    start_time = time.perf_counter()
    for residual in prepared["residuals"]:
        for pair in pairs:
            lower, upper, beta = estimator.predict_interval(tuple(sorted(pair)))
            result = results[pair]
            result["lower_residual_quantile"].append(float(lower))
            result["upper_residual_quantile"].append(float(upper))
            result["selected_beta"].append(float(beta))
        estimator.observe(float(residual))
    evaluation_seconds = time.perf_counter() - start_time

    targets = torch.as_tensor(prepared["y"], dtype=torch.float64)
    predictions = torch.as_tensor(prepared["predictions"], dtype=torch.float64)
    for pair, result in results.items():
        lo = torch.tensor(result["lower_residual_quantile"], dtype=torch.float64)
        hi = torch.tensor(result["upper_residual_quantile"], dtype=torch.float64)
        upper, lower = construct_interval_endpoints(hi, lo, predictions)
        if not torch.isfinite(upper).all() or not torch.isfinite(lower).all():
            raise ValueError(f"Nonfinite prediction interval for sequence {key!r}.")
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
            # Standard Winkler at the nominal coverage under width-minimizing
            # beta search. Fixed-tail mode retains DSCP's actual-tail scoring.
            "winkler_score": compute_winkler_score(
                hi, lo, targets, predictions,
                alpha if model_config.get("use_beta_search", True) else pair,
            ),
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
            "input_mean": float(estimator.input_mean),
            "input_std": float(estimator.input_std),
            "normalize_encoder_inputs": normalize,
            "interval_scale": "original_response",
            "residual_quantile_scale": "original_residual",
            "reservoir_seed": int(seed),
            "sampling_seed": sampling_seed,
            "initial_memory_size": int(initial_memory_size),
            "final_memory_size": int(estimator.memory_size),
            "sampling_num": int(estimator.effective_sampling_num),
            "recurrence": model_config.get("recurrence", "upstream"),
            "decay": model_config.get("decay", "linear"),
            "upstream_commit": UPSTREAM_COMMIT,
            "initialization_seconds": initialization_seconds,
            "evaluation_seconds": evaluation_seconds,
        },
    }


def evaluate_sequence(key, item, config, split="test"):
    """Evaluate one series, with split='validation' excluding reserved test values."""
    config = _config_dict(config)
    threads = _positive_integer(config.get("threads_per_worker", 1), "threads_per_worker")
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        return _evaluate_sequence(key, item, config, split)
    finally:
        torch.set_num_threads(old_threads)


def evaluate_sequences(data, config, split="test", num_cores=1):
    """Evaluate independent series; output ordering and RNGs are worker-invariant."""
    num_cores = _positive_integer(num_cores, "num_cores")
    if not data:
        raise ValueError("The prediction artifact must contain at least one series.")
    config = _config_dict(config)
    if num_cores == 1 or len(data) == 1:
        return {key: evaluate_sequence(key, item, config, split) for key, item in data.items()}
    completed = {}
    with ProcessPoolExecutor(max_workers=min(num_cores, len(data))) as executor:
        futures = {
            executor.submit(evaluate_sequence, key, item, config, split): key
            for key, item in data.items()
        }
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
    return {key: completed[key] for key in data}


def run_rescp(config_path, num_cores=1):
    """Run the fixed configuration on test and write standard DSCP artifacts."""
    config = OmegaConf.load(config_path)
    output_dir = Path(config.saving_dir)
    with open(config.data.data_path, "rb") as stream:
        data = pickle.load(stream)
    pairs = target_quantiles(config)
    print(f"ResCP: evaluating {len(data)} series on the test split", flush=True)
    log = evaluate_sequences(data, config, split="test", num_cores=num_cores)
    summary = summarize_evaluation_results(log, pairs)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=output_dir / "resolved_config.yaml", resolve=True)
    save_data(output_dir / "log.pkl", log)
    save_data(output_dir / "summary_results.pkl", summary)
    for pair, result in summary.items():
        print(
            f"{pair}: coverage={result['avg_coverage_mean']:.4f}, "
            f"width={result['avg_interval_width_mean']:.6g}, "
            f"Winkler={result['avg_winkler_score_mean']:.6g}", flush=True,
        )
    if OmegaConf.select(config, "plotting.plotting", default=False):
        from utils.plotting import plot_cp_prediction_intervals

        plot_length = _positive_integer(
            OmegaConf.select(config, "plotting.plotting_seq_len", default=200),
            "plotting.plotting_seq_len",
        )
        plot_cp_prediction_intervals(log, pairs, plot_length, str(output_dir / "plots"))
    return log
