"""Fit shared partitions using frozen, normalized calibration representations."""

from numbers import Integral

import numpy as np
from sklearn.cluster import KMeans


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _embeddings(values, name):
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] < 1 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real two-dimensional embedding array.")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite embeddings.")
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{name} must contain nonzero L2-normalized embeddings.")
    return array


def fit_groups(calibration_embeddings, test_embeddings, k=5, seed=2026, n_init=10):
    """Fit one K-means partition on calibration and assign the test contexts.

    The fitted arithmetic-mean centroids are retained without normalization.
    Empty test groups are valid and retain their zero counts. Calibration
    partitions that cannot support K groups are rejected. Return a single
    K-keyed entry using the existing artifact membership format.
    """
    calibration = _embeddings(calibration_embeddings, "calibration_embeddings")
    test = _embeddings(test_embeddings, "test_embeddings")
    if calibration.shape[1] != test.shape[1]:
        raise ValueError("Calibration and test embedding dimensions must match.")
    n_init = _positive_integer(n_init, "n_init")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral)
            or not 0 <= seed < 2**32):
        raise ValueError("seed must be an integer in [0, 2**32).")
    k = _positive_integer(k, "K")
    distinct = len(np.unique(calibration, axis=0))
    if k > distinct:
        raise ValueError(
            f"K={k} requires at least that many distinct calibration "
            f"embeddings; found {distinct} among {len(calibration)} observations."
        )

    clustering = KMeans(n_clusters=k, random_state=int(seed), n_init=n_init, algorithm="lloyd")
    calibration_labels = clustering.fit_predict(calibration).astype(np.int64)
    calibration_counts = np.bincount(calibration_labels, minlength=k)
    if np.any(calibration_counts == 0):
        raise ValueError(f"K-means collapsed to fewer than {k} nonempty calibration groups.")
    test_labels = (
        clustering.predict(test).astype(np.int64)
        if len(test) else np.empty(0, dtype=np.int64)
    )
    return {k: {
        "centroids": clustering.cluster_centers_.astype(np.float32),
        "calibration_labels": calibration_labels,
        "test_labels": test_labels,
        "calibration_counts": calibration_counts.astype(np.int64),
        "test_counts": np.bincount(test_labels, minlength=k).astype(np.int64),
        "inertia": float(clustering.inertia_),
        "seed": int(seed),
        "n_init": n_init,
    }}
