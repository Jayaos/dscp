"""Original LQT experiment from Areces et al. (ICML 2025).

This is an independent implementation of the published LQT experiment.
See UPSTREAM.md for the pinned source. No threshold or online update is clipped.
"""

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Integral

import numpy as np

from baselines.aci.model import finite_number


UPSTREAM_COMMIT = "940dd5a1e1c2db9ad5402af95b11160fd79bf3d8"
UPSTREAM_SOURCE = (
    "https://github.com/chrismohrii/conformalopt-experiments/blob/"
    + UPSTREAM_COMMIT + "/methods.py"
)
PAPER = "https://proceedings.mlr.press/v267/areces25a.html"

MODEL_DEFAULTS = {
    "method": "lqt",
    "learning_rate": 0.1,
    "learning_rate_schedule": "fixed",
    "decay_exponent": 0.6,
    "p_order": 1,
    "bias": 1.0,
    "initial_quantile": 1.0,
}


def validate_model(model):
    """Validate LQT algorithm settings.

    Extra runner settings (such as score_type and target_quantiles) are
    preserved; the experiment runner validates its own configuration schema.
    The fixed lr/p/bias defaults are fallback values, not paper-selected
    universal hyperparameters: the publication tunes these on each series.
    """
    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping.")
    result = {**deepcopy(MODEL_DEFAULTS), **deepcopy(dict(model))}
    if result["method"] != "lqt":
        raise ValueError("model.method must be 'lqt'.")
    result["learning_rate"] = finite_number(result["learning_rate"], "model.learning_rate")
    if result["learning_rate"] < 0:
        raise ValueError("model.learning_rate must be nonnegative.")
    if result["learning_rate_schedule"] not in {"fixed", "decaying"}:
        raise ValueError("model.learning_rate_schedule must be 'fixed' or 'decaying'.")
    result["decay_exponent"] = finite_number(result["decay_exponent"], "model.decay_exponent")
    if result["decay_exponent"] <= 0:
        raise ValueError("model.decay_exponent must be strictly positive.")
    p = result["p_order"]
    if isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral) or p < 0:
        raise ValueError("model.p_order must be a nonnegative integer.")
    result["p_order"] = int(p)
    result["bias"] = finite_number(result["bias"], "model.bias")
    if result["bias"] <= 0:
        raise ValueError("model.bias must be strictly positive.")
    result["initial_quantile"] = finite_number(result["initial_quantile"], "model.initial_quantile")
    return result


def _inputs(scores, alpha):
    raw = np.asarray(scores)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf":
        raise ValueError("scores must be a nonempty one-dimensional numeric array.")
    values = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Absolute-error scores must be finite and nonnegative.")
    alpha = finite_number(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one.")
    return values, alpha


def _rates(length, model):
    rates = np.full(length, model["learning_rate"], dtype=np.float64)
    if model["learning_rate_schedule"] == "decaying":
        rates *= np.arange(1, length + 1, dtype=np.float64) ** (-model["decay_exponent"])
    return rates


def _feature(scores, time, order, bias):
    return np.append(scores[time - order:time], bias)


def predict_thresholds(scores, alpha, model):
    """Return causal thresholds from a freshly initialized online predictor.

    Initial warmup thresholds are zero, not negative infinity: a zero
    absolute-error score is covered. A finite negative threshold denotes an
    empty prediction set and must not be clipped by downstream reporting.
    """
    values, alpha = _inputs(scores, alpha)
    model = validate_model(model)
    n, p = len(values), model["p_order"]
    rates = _rates(n, model)
    thresholds = np.zeros(n, dtype=np.float64)
    result = {"learning_rates": rates}
    theta = np.full(p + 1, 1.0 / p if p else model["initial_quantile"])
    for t in range(p, n):
        phi = _feature(values, t, p, model["bias"])
        with np.errstate(over="ignore", invalid="ignore"):
            thresholds[t] = theta @ phi
        if not math.isfinite(thresholds[t]):
            raise RuntimeError(f"LQT threshold overflowed at score index {t}.")
        if t + 1 < n:
            error = float(values[t] > thresholds[t])
            with np.errstate(over="ignore", invalid="ignore"):
                theta = theta + rates[t] * (error - alpha) * phi
            if not np.isfinite(theta).all():
                raise RuntimeError(f"LQT coefficients overflowed after score index {t}.")
    result["final_theta"] = theta
    result["thresholds"] = thresholds
    result["errors"] = (values > thresholds).astype(np.int64)
    return result
