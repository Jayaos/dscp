"""Quantile-forest construction shared by SPCI evaluation and tuning."""

from numbers import Integral
import warnings

import numpy as np
from sklearn.utils import check_random_state
from sklearn_quantile import (
    RandomForestQuantileRegressor,
    SampleRandomForestQuantileRegressor,
)


def _repair_sampled_leaves(estimator):
    """Finish upstream leaf draws that float32 weight accumulation left unset."""
    values = estimator.tree_.value[:, 0, 0]
    bad_leaves = np.flatnonzero(
        (estimator.tree_.children_left == -1) & ~np.isfinite(values)
    )
    if not bad_leaves.size:
        return 0

    # Recreate sklearn_quantile's per-node draws, including its float32 cast.
    # Only the CDF calculation changes; finite sampled leaves stay untouched.
    draws = check_random_state(estimator.random_state).random_sample(
        estimator.tree_.node_count
    ).astype(np.float32)
    for leaf in bad_leaves:
        in_leaf = estimator.y_train_leaves_ == leaf
        weights = np.asarray(estimator.y_weights_[in_leaf], dtype=np.float64)
        residuals = estimator.y_train_[in_leaf, 0]
        positive = weights > 0
        if (
            not positive.any()
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.isfinite(residuals[positive]).all()
        ):
            raise ValueError(
                f"Cannot repair sampled quantile-forest leaf {leaf}: "
                "expected finite training residuals and positive sampling weights."
            )
        # Normalizing in float64 and fixing the endpoint ensures every uniform
        # draw selects an observed, positive-weight residual from this leaf.
        cdf = np.cumsum(weights[positive], dtype=np.float64)
        cdf /= cdf[-1]
        cdf[-1] = 1.0
        index = np.searchsorted(cdf, draws[leaf], side="left")
        values[leaf] = residuals[positive][index]
    return int(bad_leaves.size)


class RobustSampleRandomForestQuantileRegressor(SampleRandomForestQuantileRegressor):
    """Sampled forest with a numerical repair for unresolved training-leaf draws.

    sklearn_quantile 0.1.1 subtracts float32 weights from each uniform draw.
    Rounding can leave a terminal value as NaN even with finite training data;
    np.quantile then propagates that NaN to every requested quantile. Repair
    those leaves using their original draws and normalized float64 weights.
    """

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        self.n_repaired_leaves_ = sum(
            _repair_sampled_leaves(estimator) for estimator in self.estimators_
        )
        if self.n_repaired_leaves_:
            warnings.warn(
                f"Repaired {self.n_repaired_leaves_} nonfinite sampled quantile-forest "
                "leaf value(s) using their training residuals and normalized weights.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def build_quantile_forest(config, n_train_samples, quantiles):
    """Build the same exact/sampled forest for tuning and final evaluation."""
    n_train_samples = _positive_integer(n_train_samples, "n_train_samples")
    n_estimators = _positive_integer(config.model.n_estimators, "model.n_estimators")
    max_depth = config.model.max_depth
    if max_depth is not None:
        max_depth = _positive_integer(max_depth, "model.max_depth")

    n_jobs = config.model.get("n_jobs", -1)
    if n_jobs is not None:
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs == 0:
            raise ValueError("model.n_jobs must be a nonzero integer or null.")
        n_jobs = int(n_jobs)

    seed = config.get("seed", None)
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32) or null.")
        seed = int(seed)

    quantiles = np.asarray(quantiles, dtype=float)
    if (
        quantiles.ndim != 1
        or quantiles.size == 0
        or not np.all(np.isfinite(quantiles))
        or np.any((quantiles < 0) | (quantiles > 1))
        or np.any(np.diff(quantiles) <= 0)
    ):
        raise ValueError("quantiles must be a nonempty, strictly increasing array in [0, 1].")

    forest_type = (
        RobustSampleRandomForestQuantileRegressor
        if n_train_samples > 10_000
        else RandomForestQuantileRegressor
    )
    return forest_type(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=config.model.criterion,
        n_jobs=n_jobs,
        q=quantiles,
        random_state=seed,
    )
