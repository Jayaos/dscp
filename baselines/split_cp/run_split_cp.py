"""Evaluate fixed SplitCP intervals through the existing DSCP experiment schema."""

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
import math
from numbers import Integral, Real
from pathlib import Path
import pickle
import time

import numpy as np
from omegaconf import OmegaConf

from utils.experiment_config import load_experiment_config
import torch

from baselines.split_cp.data import prepare_sequence, validate_split_settings
from baselines.split_cp.model import SplitCPResidualIntervalEstimator, quantile_rank
from utils.reporting import (
    compute_coverage,
    compute_interval_width,
    compute_winkler_score,
    summarize_evaluation_results,
)
from utils.utils import save_data


UPSTREAM_COMMIT = "15c51c66e3e5cab578a5c4ebd7494685efa83788"
UPSTREAM_SOURCE = (
    f"https://github.com/ryantibs/conformal/blob/{UPSTREAM_COMMIT}/conformalInference/R/split.R"
)


def _plain_config(config):
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping):
        raise ValueError("SplitCP configuration must be a mapping.")
    return OmegaConf.to_container(OmegaConf.create(config), resolve=True)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def target_quantiles(config):
    """Keep DSCP tuple keys; symmetric pairs encode nominal total coverage."""
    pairs = _plain_config(config).get("model", {}).get("target_quantiles", [[0.05, 0.95]])
    if not isinstance(pairs, (list, tuple)) or not pairs:
        raise ValueError("model.target_quantiles must be a nonempty list of pairs.")
    result, seen = [], set()
    for pair in pairs:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(isinstance(v, bool) or not isinstance(v, Real) for v in pair)):
            raise ValueError("Each target_quantiles pair must contain two numeric levels.")
        pair = tuple(float(v) for v in pair)
        lower, upper = sorted(pair)
        if not all(math.isfinite(v) for v in pair) or not 0 < lower < upper < 1:
            raise ValueError("Quantile levels must satisfy 0 < lower < upper < 1.")
        if not math.isclose(lower + upper, 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("SplitCP requires symmetric target_quantiles, e.g. [0.05, 0.95].")
        if (lower, upper) in seen:
            raise ValueError("model.target_quantiles must not contain duplicate intervals.")
        seen.add((lower, upper))
        result.append(pair)
    return result


def validate_config(config):
    """Validate experiment settings without loading an artifact or writing files."""
    config = _plain_config(config)
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("SplitCP requires a data configuration.")
    unknown = set(data) - {"data_path", "calibration_ratio", "test_ratio", "test_start"}
    if unknown:
        raise ValueError(f"Unknown SplitCP data settings: {sorted(unknown)}. Only calibration/test are used.")
    data.setdefault("calibration_ratio", 0.66)
    if "test_ratio" not in data:
        ratio = data["calibration_ratio"]
        if isinstance(ratio, bool) or not isinstance(ratio, Real) or not math.isfinite(ratio):
            raise ValueError("calibration_ratio must be a finite number between 0 and 1.")
        data["test_ratio"] = float(Decimal(1) - Decimal(str(ratio)))
    validate_split_settings(data["calibration_ratio"], data["test_ratio"], data.get("test_start"))
    model = config.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("model must be a mapping.")
    unknown = set(model) - {"target_quantiles", "prediction_step"}
    if unknown:
        raise ValueError(f"Unknown SplitCP model settings: {sorted(unknown)}.")
    step = model.setdefault("prediction_step", 1)
    if isinstance(step, bool) or not isinstance(step, Integral) or step != 1:
        raise ValueError("SplitCP supports model.prediction_step=1 only.")
    model["target_quantiles"] = [list(pair) for pair in target_quantiles(config)]
    config["num_cores"] = _positive_integer(config.get("num_cores", 1), "num_cores")
    plotting = config.setdefault("plotting", {})
    if not isinstance(plotting, dict):
        raise ValueError("plotting must be a mapping.")
    if not isinstance(plotting.setdefault("plotting", False), bool):
        raise ValueError("plotting.plotting must be a boolean.")
    plotting["plotting_seq_len"] = _positive_integer(
        plotting.get("plotting_seq_len", 200), "plotting.plotting_seq_len",
    )
    return config


def _alpha(pair):
    lower, upper = sorted(pair)
    return float(Decimal(1) - Decimal(str(upper)) + Decimal(str(lower)))


def evaluate_sequence(key, item, config):
    """Calibrate on the prefix once, then evaluate every target in the test suffix."""
    config = validate_config(config)
    prepared = prepare_sequence(item, config, key=key)
    started = time.perf_counter()
    estimator = SplitCPResidualIntervalEstimator().fit(prepared["calibration_residuals"])
    calibration_seconds = time.perf_counter() - started
    started = time.perf_counter()
    targets = torch.as_tensor(prepared["y"], dtype=torch.float64)
    predictions = torch.as_tensor(prepared["predictions"], dtype=torch.float64)
    results, radii, ranks = {}, {}, {}
    for pair in target_quantiles(config):
        alpha = _alpha(pair)
        radius = estimator.quantile(alpha)
        lower, upper = estimator.predict_interval(prepared["predictions"], alpha)
        lo = torch.full_like(predictions, -radius)
        hi = torch.full_like(predictions, radius)
        # Infinite endpoints are the prescribed small-calibration result. The
        # shared scorer multiplies zero by infinity, so handle this case directly.
        if math.isinf(radius):
            winkler = [float("inf")] * len(targets)
        else:
            winkler = compute_winkler_score(hi, lo, targets, predictions, alpha)
        result = {
            "lower_interval": lower.tolist(),
            "upper_interval": upper.tolist(),
            "lower_residual_quantile": lo.tolist(),
            "upper_residual_quantile": hi.tolist(),
            "target_y": targets.tolist(),
            "target_predictions": predictions.tolist(),
            "target_indices": prepared["target_indices"].tolist(),
            "coverage": compute_coverage(upper, lower, targets),
            "interval_width": compute_interval_width(upper, lower),
            "winkler_score": winkler,
        }
        result["avg_coverage"] = float(np.mean(result["coverage"]))
        result["avg_delta_coverage"] = result["avg_coverage"] - (1 - alpha)
        result["avg_interval_width"] = float(np.mean(result["interval_width"]))
        result["avg_winkler_score"] = float(np.mean(winkler))
        results[pair] = result
        radii[pair] = radius
        ranks[pair] = quantile_rank(estimator.calibration_size, alpha)
    return {
        "evaluation_results": results,
        "metadata": {
            "method": "SplitCP",
            "split": "test",
            "boundaries": prepared["boundaries"],
            "target_indices": prepared["target_indices"].tolist(),
            "calibration_size": estimator.calibration_size,
            "calibration_update": "fixed",
            "quantile_ranks": ranks,
            "quantile_radii": radii,
            "interval_scale": "original_response",
            "residual_quantile_scale": "original_residual",
            "upstream_source": UPSTREAM_SOURCE,
            "upstream_commit": UPSTREAM_COMMIT,
            "calibration_seconds": calibration_seconds,
            "evaluation_seconds": time.perf_counter() - started,
        },
    }


def _worker(args):
    return evaluate_sequence(*args)


def evaluate_sequences(data, config, num_cores=None):
    """Calibrate each series independently, preserving artifact key order."""
    config = validate_config(config)
    workers = _positive_integer(config["num_cores"] if num_cores is None else num_cores, "num_cores")
    if not isinstance(data, Mapping) or not data:
        raise ValueError("The prediction artifact must contain at least one series.")
    starts = config["data"].get("test_start")
    if isinstance(starts, Mapping) and set(starts) != set(data):
        raise ValueError("data.test_start mapping must specify exactly the artifact's series keys.")
    if workers == 1 or len(data) == 1:
        return {key: evaluate_sequence(key, item, config) for key, item in data.items()}
    jobs = ((key, item, config) for key, item in data.items())
    with ProcessPoolExecutor(max_workers=min(workers, len(data))) as executor:
        return dict(zip(data, executor.map(_worker, jobs)))


def run_split_cp(config_path, num_cores=None):
    """Run a YAML path or configuration mapping and save standard DSCP artifacts."""
    config = validate_config(
        load_experiment_config(config_path) if isinstance(config_path, (str, Path)) else config_path,
    )
    if num_cores is not None:
        config["num_cores"] = _positive_integer(num_cores, "num_cores")
    if not config["data"].get("data_path") or not config.get("saving_dir"):
        raise ValueError("Running SplitCP requires data.data_path and saving_dir.")
    with open(config["data"]["data_path"], "rb") as stream:
        data = pickle.load(stream)
    print(f"SplitCP: evaluating {len(data)} series with fixed calibration", flush=True)
    log = evaluate_sequences(data, config)
    pairs = target_quantiles(config)
    with np.errstate(invalid="ignore"):
        summary = summarize_evaluation_results(log, pairs)
    # Dispersion is undefined if any per-series mean is infinite; preserve the
    # infinite mean and record null dispersion rather than serializing NaN.
    for result in summary.values():
        for name, value in result.items():
            if name.endswith("_std") and math.isnan(value):
                result[name] = None
    output = Path(config["saving_dir"])
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(config), f=output / "resolved_config.yaml", resolve=True)
    save_data(output / "log.pkl", log)
    save_data(output / "summary_results.pkl", summary)
    for pair, result in summary.items():
        print(f"Summary for confidence pair {pair}", flush=True)
        for metric in (
            "avg_coverage", "avg_delta_coverage", "avg_interval_width", "avg_winkler_score",
        ):
            print(
                f"{metric} mean: {result[f'{metric}_mean']}, std: {result[f'{metric}_std']}",
                flush=True,
            )
    if config["plotting"]["plotting"]:
        from utils.plotting import plot_cp_prediction_intervals

        plot_cp_prediction_intervals(log, pairs, config["plotting"]["plotting_seq_len"], str(output / "plots"))
    return log
