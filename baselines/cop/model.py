"""Conformal Optimistic Prediction (Hu et al., ICLR 2026).

The deterministic ECDF path follows the authors' COP implementation.
See UPSTREAM.md and UPSTREAM_LICENSE for provenance. Signed score thresholds
are never clipped; the interval wrapper decides whether two tails intersect.
"""

from collections import deque
from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np

from baselines.aci.model import finite_number, positive_integer


UPSTREAM_COMMIT = "a41c0dcadc10979d8521ab9b5b8f3d460807d883"
UPSTREAM_SOURCE = (
    "https://github.com/creator-xi/Conformal-Optimistic-Prediction/blob/"
    + UPSTREAM_COMMIT + "/core/methods.py"
)
PAPER = "https://arxiv.org/abs/2512.07770"

MODEL_DEFAULTS = {
    "learning_rate": 0.1,
    "window_size": 100,
    "scale": 0.5,
    "proportional_lr": True,
    "initial_quantile": 0.0,
    "cdf": "ecdf",
}


def validate_model(model):
    """Validate numerical settings, retaining separately validated runner keys."""
    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping.")
    result = {**deepcopy(MODEL_DEFAULTS), **deepcopy(dict(model))}
    for name in ("learning_rate", "scale", "initial_quantile"):
        result[name] = finite_number(result[name], "model." + name)
    if result["learning_rate"] < 0:
        raise ValueError("model.learning_rate must be nonnegative.")
    if result["scale"] < 0:
        raise ValueError("model.scale must be nonnegative.")
    result["window_size"] = positive_integer(result["window_size"], "model.window_size")
    if not isinstance(result["proportional_lr"], bool):
        raise ValueError("model.proportional_lr must be a boolean.")
    if result["cdf"] != "ecdf":
        raise ValueError("model.cdf must be 'ecdf', the published executable COP path.")
    return result


def _scores(values):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf":
        raise ValueError("scores must be a nonempty one-dimensional numeric array.")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("scores must contain only finite values.")
    return result


class COP:
    """One independent COP score stream.

    Before observing score t, predict() returns the issued threshold. observe()
    computes the next primary threshold and its optimistic refinement from
    the PREVIOUS window, then appends the newly observed score. The outcome
    changes the binary gradient but cannot enter that step's CDF or score range.

    Replay calibration by alternating predict()/observe(), then continue with
    test observations on the same instance. Each instance owns all its state.
    """

    def __init__(self, alpha=0.1, learning_rate=0.1, window_size=100,
                 scale=0.5, proportional_lr=True, initial_quantile=0.0, cdf="ecdf"):
        self.alpha = finite_number(alpha, "alpha")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be strictly between zero and one.")
        self.model = validate_model({
            "learning_rate": learning_rate, "window_size": window_size,
            "scale": scale, "proportional_lr": proportional_lr,
            "initial_quantile": initial_quantile, "cdf": cdf,
        })
        self.primary_threshold = self.model["initial_quantile"]
        self.threshold = self.model["initial_quantile"]
        self.seen = 0
        self._history = deque(maxlen=self.model["window_size"])

    @property
    def history_size(self):
        return len(self._history)

    def predict(self):
        """Return the currently issued threshold without reading an outcome."""
        return self.threshold

    def observe(self, score):
        """Consume one outcome and return diagnostics for the completed step."""
        score = finite_number(score, "score")
        past = np.fromiter(self._history, dtype=np.float64)
        rate = self.model["learning_rate"]
        if self.model["proportional_lr"] and self.seen > 0:
            with np.errstate(over="ignore", invalid="ignore"):
                rate = float(rate * (past.max() - past.min()))
        if not math.isfinite(rate):
            raise RuntimeError(f"COP learning rate overflowed at score index {self.seen}.")

        issued, primary = self.threshold, self.primary_threshold
        error = int(score > issued)
        gradient = self.alpha if not error else -(1.0 - self.alpha)
        primary_next = primary - rate * gradient
        if not math.isfinite(primary_next):
            raise RuntimeError(f"COP primary threshold overflowed at score index {self.seen}.")
        active = self.seen >= self.model["window_size"]
        optimistic_gradient = float(np.mean(past <= primary_next) - (1.0 - self.alpha)) if active else 0.0
        # Keep upstream multiplication order: integrator=-scale*grad_i,
        # refinement=lr_t*integrator, then add it to the primary threshold.
        refinement = rate * (-self.model["scale"] * optimistic_gradient)
        threshold_next = primary_next + refinement
        if not math.isfinite(refinement) or not math.isfinite(threshold_next):
            raise RuntimeError(f"COP optimistic threshold overflowed at score index {self.seen}.")

        diagnostic = {
            "threshold": issued, "primary_threshold": primary,
            "learning_rate": rate, "gradient": gradient, "error": error,
            "optimistic_gradient": optimistic_gradient, "refinement": refinement,
            "cdf_active": active, "history_size_used": self.history_size,
            "primary_next": primary_next, "threshold_next": threshold_next,
        }
        # Commit only after all calculations succeed, keeping failures explicit.
        self.primary_threshold = primary_next
        self.threshold = threshold_next
        self._history.append(score)
        self.seen += 1
        return diagnostic

    def state_dict(self):
        """Return a detached diagnostic snapshot, not a resumable checkpoint."""
        return {
            "primary_threshold": self.primary_threshold, "threshold": self.threshold,
            "seen": self.seen, "history_size": self.history_size,
            "history": list(self._history),
        }


def predict_thresholds(scores, alpha, model):
    """Run a fresh COP stream and expose issued thresholds and update traces.

    Unlike the authors' array routine, the last outcome is also consumed so
    final_state describes a valid continuation point. This does not alter any
    issued threshold. The finite update checks also apply to that final step.
    """
    values = _scores(scores)
    model = validate_model(model)
    estimator = COP(alpha=alpha, **{name: model[name] for name in MODEL_DEFAULTS})
    rows = [estimator.observe(float(score)) for score in values]
    fields = {
        "thresholds": "threshold",
        "primary_thresholds": "primary_threshold",
        "learning_rates": "learning_rate",
        "gradients": "gradient",
        "optimistic_gradients": "optimistic_gradient",
        "refinements": "refinement",
        "errors": "error",
        "cdf_active": "cdf_active",
        "history_sizes_used": "history_size_used",
    }
    result = {}
    for output, field in fields.items():
        dtype = bool if field == "cdf_active" else np.int64 if field in {"error", "history_size_used"} else float
        result[output] = np.asarray([row[field] for row in rows], dtype=dtype)
    result["final_state"] = estimator.state_dict()
    return result
