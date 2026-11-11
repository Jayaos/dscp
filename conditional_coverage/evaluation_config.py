"""Configuration for evaluating saved method results against shared groups."""

from collections.abc import Mapping
import copy
import math
from pathlib import Path

import yaml

from .config import resolve_path


SUPPORTED_METHODS = {"qr_cp", "iqn_cp", "local_cp"}


def validate_evaluation_config(config, base_dir=None):
    if not isinstance(config, Mapping):
        raise ValueError("Evaluation configuration must be a mapping.")
    config = copy.deepcopy(dict(config))
    unknown = set(config) - {
        "path_base", "groups_dir", "output_dir", "data_path", "nominal_coverage",
        "allow_chronological_fallback", "runs", "sequences",
    }
    if unknown:
        raise ValueError(f"Unknown evaluation options: {sorted(unknown)}.")
    base = resolve_path(config.get("path_base", "."), base_dir or Path.cwd())
    config["path_base"] = str(base)
    for field in ("groups_dir", "output_dir"):
        if field not in config:
            raise ValueError(f"Evaluation configuration requires {field}.")
        config[field] = str(resolve_path(config[field], base))
    if "data_path" in config:
        config["data_path"] = str(resolve_path(config["data_path"], base))
    nominal = config.setdefault("nominal_coverage", 0.9)
    if (isinstance(nominal, bool) or not isinstance(nominal, (int, float))
            or not math.isfinite(nominal) or not 0 < nominal < 1):
        raise ValueError("nominal_coverage must be a finite fraction between zero and one.")
    config.setdefault("allow_chronological_fallback", False)
    if not isinstance(config["allow_chronological_fallback"], bool):
        raise ValueError("allow_chronological_fallback must be a boolean.")
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must list at least one saved QR-CP, IQN-CP, or Local-CP run.")
    names = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Each run must be a mapping.")
        unknown = set(run) - {
            "name", "method", "run_dir", "config_path", "quantile_pair",
            "allow_chronological_fallback",
        }
        if unknown:
            raise ValueError(f"Unknown run options: {sorted(unknown)}.")
        name = run.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Runs require unique nonempty names.")
        names.add(name)
        if run.get("method") not in SUPPORTED_METHODS:
            raise ValueError(f"{name}: method must be qr_cp, iqn_cp, or local_cp.")
        if "run_dir" not in run:
            raise ValueError(f"{name}: run_dir is required.")
        run["run_dir"] = str(resolve_path(run["run_dir"], base))
        if "config_path" in run:
            run["config_path"] = str(resolve_path(run["config_path"], base))
        run.setdefault("allow_chronological_fallback", config["allow_chronological_fallback"])
        if not isinstance(run["allow_chronological_fallback"], bool):
            raise ValueError(f"{name}: allow_chronological_fallback must be a boolean.")
        if "quantile_pair" in run:
            pair = run["quantile_pair"]
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in pair)):
                raise ValueError(f"{name}: quantile_pair must contain two finite quantile levels.")
            lower, upper = sorted(pair)
            if not 0 < lower < upper < 1 or not math.isclose(upper - lower, nominal, abs_tol=1e-8):
                raise ValueError(f"{name}: quantile_pair must match nominal_coverage.")
            run["quantile_pair"] = [float(lower), float(upper)]
    if "sequences" in config:
        keys = config["sequences"]
        if (not isinstance(keys, list) or not keys
                or any(isinstance(key, bool) or not isinstance(key, (str, int)) for key in keys)
                or len(set(keys)) != len(keys)):
            raise ValueError("sequences must contain distinct string or integer sequence keys.")
    return config


def load_evaluation_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        return validate_evaluation_config(yaml.safe_load(stream), path.parent)
