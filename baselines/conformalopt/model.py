"""Original LQT and SQT+AR experiments from Areces et al. (ICML 2025).

This is an independent implementation of the published experiment algorithms.
See UPSTREAM.md for the pinned source and the causal correction to the
zero-order AR initialization. No threshold or online update is clipped.
"""

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Integral

import numpy as np

from baselines.aci.model import finite_number, positive_integer


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
    "ar_window_size": 200,
    "solver": "CLARABEL",
    "solver_options": {},
}


def validate_model(model):
    """Validate algorithm settings without importing an optional AR solver.

    Extra runner settings (such as score_type and target_quantiles) are
    preserved; the experiment runner validates its own configuration schema.
    The fixed lr/p/bias defaults are fallback values, not paper-selected
    universal hyperparameters: the publication tunes these on each series.
    """
    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping.")
    result = {**deepcopy(MODEL_DEFAULTS), **deepcopy(dict(model))}
    if result["method"] not in {"lqt", "sqt_ar"}:
        raise ValueError("model.method must be 'lqt' or 'sqt_ar'.")
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
    result["ar_window_size"] = positive_integer(result["ar_window_size"], "model.ar_window_size")
    if not isinstance(result["solver"], str) or not result["solver"].strip():
        raise ValueError("model.solver must be a nonempty CVXPY solver name.")
    result["solver"] = result["solver"].upper()
    if not isinstance(result["solver_options"], Mapping):
        raise ValueError("model.solver_options must be a mapping.")
    result["solver_options"] = deepcopy(dict(result["solver_options"]))
    if "solver" in result["solver_options"]:
        raise ValueError("Set the solver using model.solver, not solver_options.")
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


def compute_ar_scorecasts(scores, alpha, model):
    """Fit the authors' unregularized rolling quantile AR scorecaster.

    The prediction at index t uses outcomes only through t-1. Regression
    features are oldest-to-newest lagged scores followed by an intercept.
    The original trimming condition retains window_size + 1 rows.

    The first fit contains an all-zero feature row. For p=0, upstream pairs
    that row with scores[-1], which changes only an additive objective
    constant. We use zero instead, avoiding any future-outcome access.

    Scorecasting does not depend on the online learning rate. The returned
    dictionary can be reused across rate candidates on the SAME score
    sequence, alpha and AR configuration.
    """
    values, alpha = _inputs(scores, alpha)
    model = validate_model(model)
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "SQT+AR requires CVXPY; update the dscp environment from envs/env-dscp.yml."
        ) from exc
    if model["solver"] not in cp.installed_solvers():
        raise ValueError("Configured CVXPY solver is not installed: " + model["solver"])

    n, p, window = len(values), model["p_order"], model["ar_window_size"]
    predictions = np.zeros(n, dtype=np.float64)
    row_counts = np.zeros(n, dtype=np.int64)
    statuses = ["warmup"] * n
    rows, outcomes = [], []
    for t in range(p, n):
        if t == p:
            # The experiment initializes all features before p to zero.
            rows.append(np.zeros(p + 1, dtype=np.float64))
            outcomes.append(0.0 if p == 0 else float(values[p - 1]))
        else:
            rows.append(_feature(values, t - 1, p, 1.0))
            outcomes.append(float(values[t - 1]))
        if len(rows) > window + 1:
            del rows[0]
            del outcomes[0]
        design = np.asarray(rows)
        observed = np.asarray(outcomes)
        coefficients = cp.Variable(p + 1)
        residual = observed - design @ coefficients
        quantile = 1.0 - alpha
        objective = cp.sum(cp.maximum(quantile * residual, (quantile - 1.0) * residual))
        problem = cp.Problem(cp.Minimize(objective))
        try:
            problem.solve(solver=model["solver"], **model["solver_options"])
        except cp.error.SolverError as exc:
            raise RuntimeError(f"AR quantile regression solver failed at score index {t}: {exc}") from exc
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or coefficients.value is None:
            raise RuntimeError(f"AR quantile regression failed at score index {t}: {problem.status}.")
        beta = np.asarray(coefficients.value, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(beta @ _feature(values, t, p, 1.0))
        if not np.isfinite(beta).all() or not math.isfinite(value):
            raise RuntimeError(f"AR quantile regression produced a nonfinite prediction at score index {t}.")
        predictions[t] = value
        row_counts[t] = len(rows)
        statuses[t] = problem.status

    return {
        "scorecasts": predictions,
        "ar_training_rows": row_counts,
        "solver_status": statuses,
        "solver": model["solver"],
        "cvxpy_version": cp.__version__,
        # Identify the cache so a different sequence/config cannot silently use it.
        "scores": values.copy(),
        "alpha": alpha,
        "p_order": p,
        "ar_window_size": window,
        "solver_options": deepcopy(model["solver_options"]),
    }


def _cached_scorecasts(cached, scores, alpha, model):
    if not isinstance(cached, Mapping):
        raise ValueError("scorecasts must be a dictionary from compute_ar_scorecasts.")
    for name, expected in (
        ("alpha", alpha), ("p_order", model["p_order"]),
        ("ar_window_size", model["ar_window_size"]), ("solver", model["solver"]),
        ("solver_options", model["solver_options"]),
    ):
        if cached.get(name) != expected:
            raise ValueError(f"Cached AR scorecasts have mismatched {name}.")
    if not np.array_equal(cached.get("scores"), scores):
        raise ValueError("Cached AR scorecasts belong to a different score sequence.")
    values = np.asarray(cached.get("scorecasts"), dtype=np.float64)
    if values.shape != scores.shape or not np.isfinite(values).all():
        raise ValueError("Cached AR scorecasts must be finite and match the score sequence shape.")
    if np.any(values[:model["p_order"]] != 0):
        raise ValueError("Cached AR scorecasts must preserve the zero-threshold warmup.")
    return values


def predict_thresholds(scores, alpha, model, scorecasts=None):
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
    if model["method"] == "lqt":
        if scorecasts is not None:
            raise ValueError("LQT does not use AR scorecasts.")
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
    else:
        cached = compute_ar_scorecasts(values, alpha, model) if scorecasts is None else scorecasts
        forecast = _cached_scorecasts(cached, values, alpha, model)
        corrections = np.zeros(n, dtype=np.float64)
        for t in range(p, n):
            with np.errstate(over="ignore", invalid="ignore"):
                thresholds[t] = forecast[t] + corrections[t]
            if not math.isfinite(thresholds[t]):
                raise RuntimeError(f"SQT+AR threshold overflowed at score index {t}.")
            if t + 1 < n:
                error = float(values[t] > thresholds[t])
                with np.errstate(over="ignore", invalid="ignore"):
                    corrections[t + 1] = corrections[t] + rates[t] * (error - alpha)
                if not math.isfinite(corrections[t + 1]):
                    raise RuntimeError(f"SQT+AR correction overflowed after score index {t}.")
        result.update({
            "scorecasts": forecast.copy(), "corrections": corrections,
            "ar_training_rows": np.asarray(cached["ar_training_rows"]).copy(),
            "solver_status": list(cached["solver_status"]),
            "solver": cached["solver"], "cvxpy_version": cached["cvxpy_version"],
        })
    result["thresholds"] = thresholds
    result["errors"] = (values > thresholds).astype(np.int64)
    return result
