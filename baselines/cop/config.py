"""COP settings and calibration-selection protocol."""

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
from itertools import product
from numbers import Integral

from omegaconf import OmegaConf

from baselines.aci.data import validate_split_settings
from baselines.aci.model import finite_number, positive_integer
from baselines.aci.run_aci import target_quantiles, _alpha
from baselines.cop.model import MODEL_DEFAULTS, validate_model

RUNNER_DEFAULTS = {"score_type": "signed", "initialization": "calibration_tail",
                   "prediction_step": 1, "target_quantiles": [[.05, .95]]}


def candidates(config):
    tuning = config["tuning"]
    for rate, window, scale in product(tuning["learning_rate_grid"],
                                       tuning["window_size_grid"], tuning["scale_grid"]):
        yield {"learning_rate": rate, "window_size": window, "scale": scale}


def validate_config(config):
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping):
        raise ValueError("COP configuration must be a mapping.")
    config = deepcopy(dict(config))
    for section in ("data", "model", "tuning", "plotting"):
        if not isinstance(config.setdefault(section, {}), dict):
            raise ValueError(f"{section} must be a mapping.")
    data, model, tuning, plotting = (config[name] for name in ("data", "model", "tuning", "plotting"))
    unknown = set(data) - {"data_path", "calibration_ratio", "test_ratio", "test_start"}
    if unknown:
        raise ValueError("Unsupported data settings: " + ", ".join(sorted(unknown)))
    ratio = finite_number(data.setdefault("calibration_ratio", .66), "data.calibration_ratio")
    data.setdefault("test_ratio", float(Decimal(1) - Decimal(str(ratio))))
    validate_split_settings(ratio, data["test_ratio"], data.get("test_start"))
    unknown = set(model) - set(MODEL_DEFAULTS) - set(RUNNER_DEFAULTS)
    if unknown:
        raise ValueError("Unknown COP model settings: " + ", ".join(sorted(unknown)))
    for name, value in RUNNER_DEFAULTS.items():
        model.setdefault(name, deepcopy(value))
    model = validate_model(model)
    config["model"] = model
    if model["score_type"] not in {"signed", "absolute"}:
        raise ValueError("model.score_type must be 'signed' or 'absolute'.")
    if model["initialization"] not in {"calibration_tail", "calibration_replay", "cold_start"}:
        raise ValueError("model.initialization must be 'calibration_tail', 'calibration_replay', or 'cold_start'.")
    if positive_integer(model["prediction_step"], "model.prediction_step") != 1:
        raise ValueError("Only sequential prediction_step=1 is supported.")
    try:
        model["target_quantiles"] = [list(pair) for pair in target_quantiles(config)]
    except ValueError as exc:
        message = str(exc).replace("Absolute/relative-error ACI", "This COP runner")
        raise ValueError(message) from exc
    config["num_cores"] = positive_integer(config.get("num_cores", 1), "num_cores")
    if not isinstance(plotting.setdefault("plotting", False), bool):
        raise ValueError("plotting.plotting must be a boolean.")
    plotting["plotting_seq_len"] = positive_integer(plotting.get("plotting_seq_len", 200), "plotting_seq_len")
    allowed = {"enabled", "learning_rate_grid", "window_size_grid", "scale_grid",
               "coverage_tolerance", "warmup_steps"}
    if set(tuning) - allowed:
        raise ValueError("Unknown tuning settings: " + ", ".join(sorted(set(tuning) - allowed)))
    if not isinstance(tuning.setdefault("enabled", True), bool):
        raise ValueError("tuning.enabled must be a boolean.")
    tolerance = finite_number(tuning.setdefault("coverage_tolerance", .01), "tuning.coverage_tolerance")
    if not 0 <= tolerance < 1:
        raise ValueError("tuning.coverage_tolerance must be in [0, 1).")
    warmup = tuning.setdefault("warmup_steps", None)
    if warmup is not None and (isinstance(warmup, bool) or not isinstance(warmup, Integral) or warmup < 0):
        raise ValueError("tuning.warmup_steps must be null or a nonnegative integer.")
    tuning.setdefault("learning_rate_grid", [1., .5, .1, .05])
    tuning.setdefault("window_size_grid", [model["window_size"]])
    tuning.setdefault("scale_grid", [model["scale"]])
    for name in ("learning_rate_grid", "window_size_grid", "scale_grid"):
        if not isinstance(tuning[name], list) or not tuning[name]:
            raise ValueError(f"tuning.{name} must be a nonempty list.")
    for params in candidates(config):
        validate_model({**model, **params})
    selected = config.get("selected_hyperparameters_path")
    if selected is not None and (not isinstance(selected, str) or not selected):
        raise ValueError("selected_hyperparameters_path must be a nonempty path or null.")
    if selected and tuning["enabled"]:
        raise ValueError("Disable tuning when replaying selected_hyperparameters_path.")
    return config


def stream_alphas(pair, score_type):
    if score_type == "absolute":
        return {"absolute": _alpha(pair)}
    lower, upper = sorted(pair)
    return {"lower": lower, "upper": float(Decimal(1) - Decimal(str(upper)))}
