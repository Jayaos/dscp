"""Lightweight configuration validation, shared by the CLI and experiment runner."""

import math
from numbers import Integral, Real

from omegaconf import OmegaConf


MODEL_DEFAULTS = {
    "prediction_step": 1,
    "target_quantiles": [[0.05, 0.95]],
    "past_window_len": 100,
    "match_threshold": 0.1,
    "n_trees": 10,
    "bagging_ratio": 0.9,
    "beta_bins": 10,
    "qrf_n_estimators": 10,
    "qrf_max_depth": 2,
    "min_samples_per_node": 0,
    "use_beta_search": True,
}
MATCHING_DEFAULTS = {
    "cache_dir": None,
    "ks_block_size": 256,
    "max_cache_memory_mb": 256,
}


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def validate_ratios(train_ratio, valid_ratio, test_ratio):
    ratios = tuple(_finite_number(value, name) for value, name in zip(
        (train_ratio, valid_ratio, test_ratio),
        ("data.train_ratio", "data.valid_ratio", "data.test_ratio"),
    ))
    if ratios[0] <= 0 or ratios[1] < 0 or ratios[2] <= 0:
        raise ValueError("train_ratio and test_ratio must be positive; valid_ratio may be zero.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError("data.train_ratio + data.valid_ratio + data.test_ratio must sum to one.")
    return ratios


def _snap_integer(value):
    nearest = round(value)
    return nearest if math.isclose(value, nearest, rel_tol=0, abs_tol=1e-10) else value


def split_boundaries(length, train_ratio, valid_ratio, test_ratio):
    """Use DSCP's floor(train)/ceil(valid) boundaries on the complete suffix."""
    length = positive_integer(length, "sequence length")
    train_ratio, valid_ratio, test_ratio = validate_ratios(train_ratio, valid_ratio, test_ratio)
    train_size = math.floor(_snap_integer(length * train_ratio))
    valid_size = math.ceil(_snap_integer(length * valid_ratio))
    test_size = length - train_size - valid_size
    if train_size < 1 or test_size < 1:
        raise ValueError(
            f"Empty training/test partition for length={length}: "
            f"train={train_size}, validation={valid_size}, test={test_size}."
        )
    return {
        "train_end": train_size,
        "validation_end": train_size + valid_size,
        "test_start": train_size + valid_size,
        "train_size": train_size,
        "validation_size": valid_size,
        "test_size": test_size,
    }


def target_quantiles(config):
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    pairs = config.get("model", {}).get("target_quantiles", [[0.05, 0.95]])
    if not isinstance(pairs, (list, tuple)) or not pairs:
        raise ValueError("model.target_quantiles must be a nonempty list of pairs.")
    result, seen = [], set()
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each target_quantiles entry must contain two levels.")
        pair = tuple(_finite_number(value, "quantile level") for value in pair)
        lower, upper = sorted(pair)
        if not 0 < lower < upper < 1:
            raise ValueError("Quantile levels must satisfy 0 < lower < upper < 1.")
        if (lower, upper) in seen:
            raise ValueError("model.target_quantiles must not contain duplicate intervals.")
        seen.add((lower, upper))
        result.append(pair)
    return result


def validate_config(config, num_cores=None):
    """Return a resolved plain config; no array/model imports or data access."""
    config = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("DistMatch configuration must be a mapping.")
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("DistMatch configuration requires a data mapping.")
    names = ("train_ratio", "valid_ratio", "test_ratio")
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"Missing DistMatch data settings: {missing}")
    validate_ratios(*(data[name] for name in names))
    data.setdefault("normalize", False)
    if not isinstance(data["normalize"], bool):
        raise ValueError("data.normalize must be a boolean.")
    config["seed"] = positive_integer(config.get("seed", 2026), "seed", minimum=0)
    if config["seed"] >= 2**32:
        raise ValueError("seed must be smaller than 2**32.")
    config["num_cores"] = positive_integer(
        config.get("num_cores", 1) if num_cores is None else num_cores, "num_cores"
    )
    config["threads_per_worker"] = positive_integer(
        config.get("threads_per_worker", 1), "threads_per_worker"
    )
    config.setdefault("show_progress", True)
    if not isinstance(config["show_progress"], bool):
        raise ValueError("show_progress must be a boolean.")
    for section, defaults in (("model", MODEL_DEFAULTS), ("matching", MATCHING_DEFAULTS)):
        supplied = config.get(section, {})
        if not isinstance(supplied, dict):
            raise ValueError(f"{section} must be a mapping.")
        unknown = set(supplied) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown DistMatch {section} settings: {sorted(unknown)}")
        config[section] = {**defaults, **supplied}
    model = config["model"]
    if positive_integer(model["prediction_step"], "model.prediction_step") != 1:
        raise ValueError("DistMatch currently supports model.prediction_step=1 only.")
    for name in ("past_window_len", "n_trees"):
        positive_integer(model[name], f"model.{name}")
    # sklearn-quantile 0.1.1 can produce NaN endpoint quantiles with one tree.
    positive_integer(model["qrf_n_estimators"], "model.qrf_n_estimators", minimum=2)
    positive_integer(model["beta_bins"], "model.beta_bins", minimum=2)
    positive_integer(model["min_samples_per_node"], "model.min_samples_per_node", minimum=0)
    if model["qrf_max_depth"] is not None:
        positive_integer(model["qrf_max_depth"], "model.qrf_max_depth")
    for name in ("match_threshold", "bagging_ratio"):
        value = _finite_number(model[name], f"model.{name}")
        if not 0 < value <= 1:
            raise ValueError(f"model.{name} must be in (0, 1].")
    if not isinstance(model["use_beta_search"], bool):
        raise ValueError("model.use_beta_search must be a boolean.")
    target_quantiles(config)
    matching = config["matching"]
    positive_integer(matching["ks_block_size"], "matching.ks_block_size")
    if _finite_number(matching["max_cache_memory_mb"], "matching.max_cache_memory_mb") <= 0:
        raise ValueError("matching.max_cache_memory_mb must be positive.")
    if matching["cache_dir"] is not None and not isinstance(matching["cache_dir"], str):
        raise ValueError("matching.cache_dir must be a path string or null.")
    return config
