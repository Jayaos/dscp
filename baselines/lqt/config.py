"""Validated configuration and the published experiment search grids."""

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
from itertools import product
from omegaconf import OmegaConf
from baselines.aci.data import validate_split_settings
from baselines.aci.model import finite_number, positive_integer
from baselines.aci.run_aci import target_quantiles, _alpha
from baselines.lqt.model import MODEL_DEFAULTS, validate_model

LEARNING_RATE_GRID = [1e-5, 1e-4, 1e-3, 1e-2, .1, 1., 10., 100., 1000., 10000., 100000.]
BIAS_GRID = [.1, 1., 5., 10., 200., 100., 1000.]
RUNNER_MODEL_DEFAULTS = {"score_type": "absolute", "prediction_step": 1,
                         "target_quantiles": [[.05, .95]]}


def validate_config(config):
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping):
        raise ValueError("LQT configuration must be a mapping.")
    config = deepcopy(dict(config))
    for section in ("data", "model", "tuning", "plotting"):
        if not isinstance(config.setdefault(section, {}), dict):
            raise ValueError(f"{section} must be a mapping.")
    data, model, tuning, plotting = (config[key] for key in ("data", "model", "tuning", "plotting"))
    unknown = set(data) - {"data_path", "calibration_ratio", "test_ratio", "test_start"}
    if unknown:
        raise ValueError("Unsupported data settings: " + ", ".join(sorted(unknown)))
    ratio = finite_number(data.setdefault("calibration_ratio", .66), "data.calibration_ratio")
    data.setdefault("test_ratio", float(Decimal(1) - Decimal(str(ratio))))
    validate_split_settings(ratio, data["test_ratio"], data.get("test_start"))
    unknown = set(model) - set(MODEL_DEFAULTS) - set(RUNNER_MODEL_DEFAULTS)
    if unknown:
        raise ValueError("Unknown LQT model settings: " + ", ".join(sorted(unknown)))
    for key, value in RUNNER_MODEL_DEFAULTS.items():
        model.setdefault(key, deepcopy(value))
    model = validate_model(model)
    config["model"] = model
    if model["score_type"] != "absolute":
        raise ValueError("LQT uses model.score_type=absolute.")
    if positive_integer(model["prediction_step"], "model.prediction_step") != 1:
        raise ValueError("Only sequential prediction_step=1 is supported.")
    model["target_quantiles"] = [list(pair) for pair in target_quantiles(config)]
    config["num_cores"] = positive_integer(config.get("num_cores", 1), "num_cores")
    if not isinstance(plotting.setdefault("plotting", False), bool):
        raise ValueError("plotting.plotting must be a boolean.")
    plotting["plotting_seq_len"] = positive_integer(plotting.get("plotting_seq_len", 200), "plotting_seq_len")
    allowed = {"enabled", "learning_rate_grid", "p_order_grid", "bias_grid", "coverage_tolerance"}
    if set(tuning) - allowed:
        raise ValueError("Unknown tuning settings: " + ", ".join(sorted(set(tuning) - allowed)))
    if not isinstance(tuning.setdefault("enabled", True), bool):
        raise ValueError("tuning.enabled must be a boolean.")
    tolerance = finite_number(tuning.setdefault("coverage_tolerance", .01), "tuning.coverage_tolerance")
    if not 0 <= tolerance < 1:
        raise ValueError("tuning.coverage_tolerance must be in [0, 1).")
    tuning.setdefault("learning_rate_grid", LEARNING_RATE_GRID.copy())
    tuning.setdefault("p_order_grid", [0, 1, 2])
    tuning.setdefault("bias_grid", BIAS_GRID.copy())
    for name in ("learning_rate_grid", "p_order_grid", "bias_grid"):
        if not isinstance(tuning[name], list) or not tuning[name]:
            raise ValueError(f"tuning.{name} must be a nonempty list.")
    for params in candidates(config):
        validate_model({**model, **params})
    selection = config.get("selected_hyperparameters_path")
    if selection is not None and (not isinstance(selection, str) or not selection):
        raise ValueError("selected_hyperparameters_path must be a nonempty path or null.")
    if selection and tuning["enabled"]:
        raise ValueError("Disable tuning when replaying selected_hyperparameters_path.")
    return config


def candidates(config):
    tuning = config["tuning"]
    for rate, order, bias in product(tuning["learning_rate_grid"], tuning["p_order_grid"], tuning["bias_grid"]):
        yield {"learning_rate": rate, "p_order": order, "bias": bias}
