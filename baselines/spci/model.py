"""Quantile-forest construction shared by SPCI evaluation and tuning."""

from numbers import Integral

import numpy as np
from sklearn_quantile import (
    RandomForestQuantileRegressor,
    SampleRandomForestQuantileRegressor,
)


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
        SampleRandomForestQuantileRegressor
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
