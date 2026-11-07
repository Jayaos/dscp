"""Original simple ACI update with an unclipped adaptive miscoverage level.

Independent implementation of Gibbs and Candes (2021), equation (2).
The default quantile matches R's type 7, used in the authors' ACCode.R.
"""

from bisect import bisect_left, insort
from collections import deque
import math
from numbers import Integral, Real

import numpy as np


UPSTREAM_COMMIT = "b7fff0cbd2d3a1e954bc9c426327e355a402d2cd"
UPSTREAM_SOURCE = (
    "https://github.com/isgibbs/AdaptiveConformal/blob/"
    + UPSTREAM_COMMIT + "/ACCode.R"
)
PAPER = "https://arxiv.org/pdf/2106.00170"


def finite_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


class ACI:
    """One independent adaptive level and rolling score pool.

    Call fit once, then alternate predict and observe. Calibration initializes
    the score pool only: alpha starts at initial_alpha (nominal alpha by
    default), with no adaptive updates on the calibration observations.
    A negative-infinite threshold denotes the empty set, not a negative width.
    """

    def __init__(self, alpha=0.1, gamma=0.005, window_size=1250,
                 initial_alpha=None, quantile_method="linear"):
        self.alpha = finite_number(alpha, "alpha")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be strictly between zero and one.")
        self.gamma = finite_number(gamma, "gamma")
        if self.gamma < 0:
            raise ValueError("gamma must be nonnegative (zero is a fixed-level control).")
        self.window_size = None if window_size is None else positive_integer(window_size, "window_size")
        self.initial_alpha = self.alpha if initial_alpha is None else finite_number(initial_alpha, "initial_alpha")
        if not 0 <= self.initial_alpha <= 1:
            raise ValueError("initial_alpha must be between zero and one inclusive.")
        if quantile_method not in {"linear", "inverted_cdf"}:
            raise ValueError("quantile_method must be 'linear' or 'inverted_cdf'.")
        self.quantile_method = quantile_method
        self.alpha_t = self.initial_alpha
        self._history = deque()
        self._sorted = []
        self._pending = None

    @property
    def history_size(self):
        return len(self._history)

    def _append(self, score):
        if self.window_size is not None and len(self._history) == self.window_size:
            oldest = self._history.popleft()
            del self._sorted[bisect_left(self._sorted, oldest)]
        self._history.append(score)
        insort(self._sorted, score)

    def fit(self, scores):
        values = np.asarray(scores)
        if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iuf":
            raise ValueError("Calibration scores must be a nonempty one-dimensional numeric array.")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Absolute-error calibration scores must be finite and nonnegative.")
        self.alpha_t = self.initial_alpha
        self._pending = None
        if self.window_size is not None:
            values = values[-self.window_size:]
        self._history = deque(float(value) for value in values)
        self._sorted = sorted(self._history)
        return self

    def predict(self):
        """Issue a score threshold using only previously observed scores."""
        if not self._history:
            raise RuntimeError("Initialize ACI with calibration scores before prediction.")
        if self._pending is not None:
            raise RuntimeError("Observe the pending prediction before issuing another one.")
        if self.alpha_t <= 0:
            threshold = math.inf
        elif self.alpha_t >= 1:
            threshold = -math.inf
        else:
            probability = 1.0 - self.alpha_t
            n = len(self._sorted)
            if self.quantile_method == "inverted_cdf":
                threshold = self._sorted[max(0, math.ceil(n * probability) - 1)]
            else:
                position = (n - 1) * probability
                low, high = math.floor(position), math.ceil(position)
                fraction = position - low
                threshold = self._sorted[low] + fraction * (self._sorted[high] - self._sorted[low])
        self._pending = (float(threshold), self.alpha_t)
        return float(threshold)

    def observe(self, score):
        """Score the issued set, update alpha without clipping, then append."""
        if self._pending is None:
            raise RuntimeError("Issue a prediction before observing its score.")
        score = finite_number(score, "score")
        if score < 0:
            raise ValueError("An absolute-error score must be nonnegative.")
        threshold, alpha_used = self._pending
        error = int(score > threshold)  # Equality is covered.
        self.alpha_t = alpha_used + self.gamma * (self.alpha - error)
        if not math.isfinite(self.alpha_t):
            raise ValueError("The configured step size overflowed the adaptive level.")
        self._append(score)
        self._pending = None
        return {"alpha_used": alpha_used, "alpha_next": self.alpha_t,
                "score_threshold": threshold, "score": score, "error": error}
