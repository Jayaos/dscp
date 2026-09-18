"""Quantile-forest construction shared by SPCI evaluation and tuning."""

from numbers import Integral
import warnings

import numpy as np
from sklearn.utils import check_random_state
from sklearn_quantile import (
    RandomForestQuantileRegressor,
    SampleRandomForestQuantileRegressor,
)


class RobustRandomForestQuantileRegressor(RandomForestQuantileRegressor):
    """Exact forest with endpoint quantiles taken from conditional support.

    sklearn_quantile 0.1.1 accumulates its weighted CDF in float32. Its last
    cumulative weight can fall short of one, leaving Q(1) unset. Q(0) can also
    encounter a zero interpolation denominator. The endpoint quantiles are
    exactly the minimum and maximum positive-weight residuals in the reached
    leaves, so calculate those directly and leave interior quantiles unchanged.
    """

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        self.leaf_minima_ = []
        self.leaf_maxima_ = []
        for estimator in self.estimators_:
            # The parent sorts these shared arrays together during fit.
            leaves = estimator.y_train_leaves_
            residuals = estimator.y_train_[:, 0]
            positive = (leaves >= 0) & (estimator.y_weights_ > 0)
            minima = np.full(estimator.tree_.node_count, np.inf, dtype=np.float32)
            maxima = np.full(estimator.tree_.node_count, -np.inf, dtype=np.float32)
            np.minimum.at(minima, leaves[positive], residuals[positive])
            np.maximum.at(maxima, leaves[positive], residuals[positive])
            self.leaf_minima_.append(minima)
            self.leaf_maxima_.append(maxima)
        return self

    def predict(self, X):
        predictions = super().predict(X)
        quantiles = self.validate_quantiles()
        lower_rows = quantiles == 0
        upper_rows = quantiles == 1
        if not (lower_rows.any() or upper_rows.any()):
            return predictions

        reached_leaves = self.apply(X)
        lower = np.full(len(reached_leaves), np.inf, dtype=np.float32)
        upper = np.full(len(reached_leaves), -np.inf, dtype=np.float32)
        for index, leaves in enumerate(reached_leaves.T):
            np.minimum(lower, self.leaf_minima_[index][leaves], out=lower)
            np.maximum(upper, self.leaf_maxima_[index][leaves], out=upper)
        if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
            raise ValueError("Exact quantile-forest leaves require finite positive-weight support.")

        if quantiles.size == 1:
            predictions[...] = lower if lower_rows[0] else upper
        else:
            predictions[lower_rows] = lower
            predictions[upper_rows] = upper
        return predictions


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
        else RobustRandomForestQuantileRegressor
    )
    return forest_type(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=config.model.criterion,
        n_jobs=n_jobs,
        q=quantiles,
        random_state=seed,
    )
