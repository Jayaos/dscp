"""Unweighted absolute-residual split conformal prediction.

Algorithm reference: ryantibs/conformal, conformalInference/R/split.R.
This independent NumPy implementation specializes the mathematical procedure
to saved forecasts, equal weights, and no local scaling. See UPSTREAM.md.
"""

from decimal import Decimal, ROUND_CEILING
from numbers import Integral, Real

import numpy as np


def scalar_sequence(values, name="values"):
    """Validate scalar observations, canonicalizing before any subtraction."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain numeric scalar values.") from exc
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty scalar sequence with shape [T] or [T, 1].")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def quantile_rank(n, alpha):
    """One-based rank in n calibration scores augmented with positive infinity.

    Decimal arithmetic respects the nominal decimal probability, avoiding an
    accidental extra rank from binary rounding at integer boundaries.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 1:
        raise ValueError("The calibration size must be a positive integer.")
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real)
            or not np.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("alpha must be finite and strictly between 0 and 1.")
    rank = Decimal(int(n) + 1) * (Decimal(1) - Decimal(str(float(alpha))))
    return int(rank.to_integral_value(rounding=ROUND_CEILING))


def conformal_quantile(scores, alpha):
    """Return the finite-sample conformal cutoff, including infinity if needed."""
    scores = scalar_sequence(scores, "scores")
    if np.any(scores < 0):
        raise ValueError("Absolute-residual scores must be nonnegative.")
    rank = quantile_rank(len(scores), alpha)
    if rank > len(scores):
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


class SplitCPResidualIntervalEstimator:
    """Calibrate once on residuals; subsequent predictions never update scores."""

    def __init__(self):
        self._scores = None

    def fit(self, calibration_residuals):
        residuals = scalar_sequence(calibration_residuals, "calibration_residuals")
        self._scores = np.sort(np.abs(residuals))
        self._scores.flags.writeable = False
        return self

    @property
    def calibration_size(self):
        if self._scores is None:
            raise RuntimeError("Fit the SplitCP calibrator before prediction.")
        return len(self._scores)

    def quantile(self, alpha):
        rank = quantile_rank(self.calibration_size, alpha)
        return float(self._scores[rank - 1]) if rank <= self.calibration_size else float("inf")

    def predict_interval(self, predictions, alpha):
        """Return (lower, upper) endpoints in the original response units."""
        radius = self.quantile(alpha)
        predictions = scalar_sequence(predictions, "predictions")
        with np.errstate(over="ignore"):
            lower, upper = predictions - radius, predictions + radius
        if np.isfinite(radius) and (not np.isfinite(lower).all() or not np.isfinite(upper).all()):
            raise ValueError("Finite SplitCP endpoints overflowed the response scale.")
        return lower, upper
