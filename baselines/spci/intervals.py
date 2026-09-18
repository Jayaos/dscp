"""Residual interval construction shared by SPCI evaluation and tuning."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class IntervalCandidates:
    alpha: float
    lower_indices: np.ndarray
    upper_indices: np.ndarray


@dataclass(frozen=True)
class IntervalPlan:
    quantiles: np.ndarray
    pairs: dict
    optimize_beta: bool
    beta_bins: int


@dataclass(frozen=True)
class SelectedInterval:
    lower: np.ndarray
    upper: np.ndarray
    beta: np.ndarray
    upper_quantile: np.ndarray
    alpha: float
    confidence_pair: tuple
    optimize_beta: bool

    @property
    def score_quantiles(self):
        # The original pair labels nominal coverage when beta varies over time.
        # Scalar alpha requests the standard Winkler penalty on either tail,
        # including candidates at beta=0 or beta=alpha.
        return self.alpha if self.optimize_beta else self.confidence_pair

    def metadata(self):
        return {
            "optimize_beta": self.optimize_beta,
            "nominal_alpha": self.alpha,
            "selected_beta": self.beta.tolist(),
            "lower_quantile_levels": self.beta.tolist(),
            "upper_quantile_levels": self.upper_quantile.tolist(),
        }


def build_interval_plan(model_config):
    """Collect forest quantiles and candidate endpoints for every coverage level.

    Missing ``optimize_beta`` preserves fixed configured endpoints. When enabled,
    each configured pair specifies its nominal coverage (upper minus lower);
    ``beta_bins`` equally spaced candidates include both 0 and alpha. Candidate
    quantiles may therefore reach 0 and 1 even though configured pairs are interior.
    """
    optimize = model_config.get("optimize_beta", False)
    if not isinstance(optimize, (bool, np.bool_)):
        raise ValueError("model.optimize_beta must be a boolean.")
    bins = model_config.get("beta_bins", 5)
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 2:
        raise ValueError("model.beta_bins must be an integer of at least 2.")
    try:
        pairs = np.asarray(model_config.get("target_quantiles"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("model.target_quantiles must contain pairs of quantile levels.") from exc
    if pairs.ndim != 2 or pairs.shape[0] == 0 or pairs.shape[1] != 2:
        raise ValueError("model.target_quantiles must contain at least one pair of quantile levels.")
    ordered = np.sort(pairs, axis=1)
    if (not np.isfinite(pairs).all() or np.any(ordered[:, 0] <= 0)
            or np.any(ordered[:, 1] >= 1) or np.any(ordered[:, 0] >= ordered[:, 1])):
        raise ValueError("model.target_quantiles must satisfy 0 < lower < upper < 1.")

    levels = {}
    for pair, (lower, upper) in zip(pairs, ordered):
        coverage = float(upper - lower)
        alpha = 1.0 - coverage
        if optimize:
            lower_levels = np.linspace(0.0, alpha, int(bins))
            upper_levels = np.clip(lower_levels + coverage, 0.0, 1.0)
        else:
            lower_levels, upper_levels = np.array([lower]), np.array([upper])
        levels[tuple(float(value) for value in pair)] = (alpha, lower_levels, upper_levels)
    quantiles = np.unique(np.concatenate([
        endpoints for _, low, high in levels.values() for endpoints in (low, high)
    ]))
    candidates = {
        pair: IntervalCandidates(alpha, np.searchsorted(quantiles, low), np.searchsorted(quantiles, high))
        for pair, (alpha, low, high) in levels.items()
    }
    return IntervalPlan(quantiles, candidates, bool(optimize), int(bins))


def select_intervals(predicted_quantiles, plan):
    """Choose the narrowest predicted interval per observation and coverage level.

    This function takes no observed outcomes. The forest stays fixed: only its
    predicted endpoint pair changes. Ties select the first (smallest-beta)
    candidate, matching the original SPCI grid search.
    """
    predictions = np.asarray(predicted_quantiles, dtype=float)
    if (predictions.ndim != 2 or predictions.shape[0] != len(plan.quantiles)
            or predictions.shape[1] == 0):
        raise ValueError("SPCI quantile predictions must have shape (num_quantiles, num_observations).")
    if not np.isfinite(predictions).all():
        raise ValueError("The quantile forest returned nonfinite predictions.")
    if np.any(np.diff(predictions, axis=0) < 0):
        raise ValueError("The quantile forest returned crossing quantile predictions.")

    columns = np.arange(predictions.shape[1])
    intervals = {}
    for pair, candidates in plan.pairs.items():
        lower = predictions[candidates.lower_indices]
        upper = predictions[candidates.upper_indices]
        choice = np.argmin(upper - lower, axis=0)
        intervals[pair] = SelectedInterval(
            lower=lower[choice, columns],
            upper=upper[choice, columns],
            beta=plan.quantiles[candidates.lower_indices[choice]],
            upper_quantile=plan.quantiles[candidates.upper_indices[choice]],
            alpha=candidates.alpha,
            confidence_pair=pair,
            optimize_beta=plan.optimize_beta,
        )
    return intervals
