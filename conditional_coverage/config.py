"""Configuration and source validation for the shared grouping experiment."""

from collections.abc import Mapping
from pathlib import Path
import copy
import hashlib
import math
import pickle

import yaml
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DEFAULTS = {
    "rnn_type": "lstm", "dim_model": 64, "num_layers": 1,
    "dropout": 0.0, "window_size": 100,
}
TRAINING_DEFAULTS = {
    "batch_size": 64, "learning_rate": 0.001, "epochs": 50,
    "early_stop": 5, "validation_ratio": 0.2,
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value, base):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("Paths must be nonempty strings.")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(base) / path).resolve()


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def validate_config(config, base_dir=None):
    if not isinstance(config, Mapping):
        raise ValueError("Grouping configuration must be a mapping.")
    config = copy.deepcopy(dict(config))
    base = resolve_path(config.get("path_base", "."), base_dir or Path.cwd())
    for name in ("data_path", "output_dir"):
        if name not in config:
            raise ValueError(f"Grouping configuration requires {name}.")
        config[name] = str(resolve_path(config[name], base))
    config["path_base"] = str(base)
    config.setdefault("seed", 2026)
    if (isinstance(config["seed"], bool) or not isinstance(config["seed"], int)
            or not 0 <= config["seed"] < 2**32):
        raise ValueError("seed must be an integer in [0, 2**32).")
    config.setdefault("device", "cpu")
    config.setdefault("num_threads", 1)
    positive_integer(config["num_threads"], "num_threads")
    config.setdefault("cache_embeddings", True)
    if not isinstance(config["cache_embeddings"], bool):
        raise ValueError("cache_embeddings must be a boolean.")
    for name, defaults in (("model", MODEL_DEFAULTS), ("training", TRAINING_DEFAULTS)):
        if not isinstance(config.get(name, {}), Mapping):
            raise ValueError(f"{name} must be a mapping.")
        supplied = config.get(name, {})
        unknown = set(supplied) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown {name} options: {sorted(unknown)}.")
        config[name] = {**defaults, **supplied}
    for name in ("dim_model", "num_layers", "window_size"):
        positive_integer(config["model"][name], f"model.{name}")
    if config["model"]["rnn_type"] not in ("lstm", "gru", "rnn"):
        raise ValueError("model.rnn_type must be lstm, gru, or rnn.")
    for name in ("batch_size", "epochs", "early_stop"):
        positive_integer(config["training"][name], f"training.{name}")
    for section, name, lower, upper in (
        ("model", "dropout", 0.0, 1.0),
        ("training", "validation_ratio", 0.0, 1.0),
        ("training", "learning_rate", 0.0, float("inf")),
    ):
        value = config[section][name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value >= upper
                or (value < lower if name == "dropout" else value <= lower)):
            raise ValueError(f"Invalid {section}.{name}: {value!r}.")
    clustering = config.get("clustering", {})
    if not isinstance(clustering, Mapping):
        raise ValueError("clustering must be a mapping.")
    if "ks" in clustering:
        raise ValueError("clustering.ks is no longer supported; set clustering.k to one positive integer.")
    if set(clustering) - {"k", "n_init"}:
        raise ValueError("clustering accepts only k and n_init.")
    config["clustering"] = {"k": 5, "n_init": 10, **clustering}
    positive_integer(config["clustering"]["k"], "clustering.k")
    positive_integer(config["clustering"]["n_init"], "clustering.n_init")
    comparisons = config.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("comparisons must list at least one method configuration or run.")
    names = set()
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ValueError("Each comparison must be a mapping.")
        name, method = comparison.get("name"), comparison.get("method")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Comparisons require unique nonempty names.")
        if not isinstance(method, str) or not method:
            raise ValueError(f"Comparison {name!r} requires method.")
        names.add(name)
        if ("config_path" in comparison) == ("run_dir" in comparison):
            raise ValueError(f"Comparison {name!r} needs exactly one of config_path or run_dir.")
        for field in ("config_path", "run_dir", "source_root", "source_data_path"):
            if field in comparison:
                comparison[field] = str(resolve_path(comparison[field], base))
    if "sequences" in config:
        keys = config["sequences"]
        if (not isinstance(keys, list) or not keys
                or any(isinstance(k, bool) or not isinstance(k, (str, int)) for k in keys)
                or len(set(keys)) != len(keys)):
            raise ValueError("sequences must contain distinct string or integer sequence keys.")
    return config


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        return validate_config(yaml.safe_load(stream), path.parent)


def load_comparisons(config):
    """Read split settings and optional logs, never method model checkpoints."""
    comparisons = []
    source = Path(config["data_path"])
    source_hash = None
    for entry in config["comparisons"]:
        config_path = (Path(entry["run_dir"]) / "resolved_config.yaml"
                       if "run_dir" in entry else Path(entry["config_path"]))
        loaded = OmegaConf.load(config_path)
        if "data" not in loaded or "data_path" not in loaded.data:
            raise ValueError(f"{config_path} must identify data.data_path.")
        # Resolve only data: unused saving_dir timestamps need no resolver.
        data = OmegaConf.to_container(loaded.data, resolve=True)
        method_source = resolve_path(
            entry.get("source_data_path", data["data_path"]),
            entry.get("source_root", REPO_ROOT),
        )
        if method_source != source:
            if not method_source.is_file() or not source.is_file():
                raise ValueError(
                    f"{entry['name']}: different or unavailable source timeline {method_source}. "
                    "Use the same forecast artifact; source_data_path can map a relocated run."
                )
            source_hash = source_hash or file_sha256(source)
            if file_sha256(method_source) != source_hash:
                raise ValueError(f"{entry['name']}: source artifact differs; cannot mix timelines.")
        horizon = loaded.get("model", {}).get("prediction_step", 1)
        if isinstance(horizon, bool) or horizon != 1:
            raise ValueError("Shared groups currently support one-step experiments only.")
        comparison = {
            "name": entry["name"], "method": entry["method"],
            "config": {"data": data, "model": {"prediction_step": 1}},
            "config_path": str(config_path), "config_sha256": file_sha256(config_path),
            "source_data_path": str(method_source),
        }
        if "run_dir" in entry:
            log_path = Path(entry["run_dir"]) / "log.pkl"
            with log_path.open("rb") as stream:
                log = pickle.load(stream)
            if not isinstance(log, Mapping):
                raise ValueError(f"{log_path}: expected a mapping of sequences.")
            comparison.update(log=log, log_path=str(log_path), log_sha256=file_sha256(log_path))
        comparisons.append(comparison)
    return comparisons
