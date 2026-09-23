"""Causal DistMatch residual intervals adapted from the official implementation.

The partition and interval aggregation follow upstream commit
d9fd84dde4a2b92cf5340a3cae578bd0fe489ccd. See UPSTREAM.md for the deliberately
different observation bookkeeping, random streams, and dependency boundary.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


UPSTREAM_COMMIT = "d9fd84dde4a2b92cf5340a3cae578bd0fe489ccd"
_CACHE_VERSION = "distmatch-equal-window-strict-ks-v1"


class DistMatchCrossedBoundsError(ValueError):
    """A failed QRF interval, with enough context to record an exclusion."""

    def __init__(self, message="DistMatch QRF returned crossed interval bounds.", **diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}.")
    return int(value)


def _number(name, value, minimum=0.0, maximum=None, strict_min=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if not np.isfinite(value) or (value <= minimum if strict_min else value < minimum):
        raise ValueError(f"{name} must be finite and {'greater than' if strict_min else 'at least'} {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return value


def _tie_ranks(sorted_rows):
    """Integer counts strictly below/at each atom, including repeated atoms."""
    left = np.empty(sorted_rows.shape, dtype=np.int64)
    right = np.empty_like(left)
    for index, row in enumerate(sorted_rows):
        left[index] = np.searchsorted(row, row, side="left")
        right[index] = np.searchsorted(row, row, side="right")
    return left, right


def _ks_distances_sorted(anchor, sorted_rows, left_ranks=None, right_ranks=None):
    """Exact two-sample KS distances for equal-length, already sorted windows.

    A negative CDF difference reaches its extremum after a candidate's jump;
    a positive difference reaches its extremum before a candidate's jump.
    Integer counts at those two boundaries handle ties without roundoff from
    subtracting floating-point empirical CDF values. Temporary arrays have
    shape (block_size, window_length), never (sample_count, sample_count).
    """
    if left_ranks is None or right_ranks is None:
        left_ranks, right_ranks = _tie_ranks(sorted_rows)
    before = np.searchsorted(anchor, sorted_rows, side="left") - left_ranks
    after = right_ranks - np.searchsorted(anchor, sorted_rows, side="right")
    counts = np.maximum(before.max(axis=1), after.max(axis=1))
    return counts.astype(np.float64) / len(anchor)


@dataclass
class _Node:
    # Member IDs index the shared chronological patch/target store. Bootstrap
    # multiplicity and ancestor order are retained in this list.
    member_ids: list | None = None
    anchor: np.ndarray | None = None
    anchor_local_id: int | None = None
    left: object = None
    right: object = None


@dataclass
class _Tree:
    root: _Node
    bootstrap_indices: np.ndarray
    depth: int
    leaves: list


class DistMatchResidualIntervalEstimator:
    """Fit fixed distribution-matching trees and update their leaf memories.

    ``fit`` accepts a chronological calibration prefix of signed residuals.
    Inputs, targets, observations, and returned quantiles share the supplied
    residual units. The data loader handles optional scaling before ``fit``.
    ``predict_intervals`` uses only its last observed residual window and does
    not advance memory or random streams. Call ``observe`` once per newly
    available outcome, after making all intervals for that time step.
    """

    def __init__(
        self,
        past_window_len=100,
        match_threshold=0.1,
        n_trees=10,
        bagging_ratio=0.9,
        beta_bins=10,
        qrf_n_estimators=10,
        qrf_max_depth=2,
        min_samples_per_node=0,
        use_beta_search=True,
        seed=2026,
        cache_dir=None,
        ks_block_size=256,
        max_cache_memory_mb=256,
    ):
        self.past_window_len = _integer("past_window_len", past_window_len)
        self.match_threshold = _number("match_threshold", match_threshold, maximum=1, strict_min=True)
        self.n_trees = _integer("n_trees", n_trees)
        self.bagging_ratio = _number("bagging_ratio", bagging_ratio, maximum=1, strict_min=True)
        self.beta_bins = _integer("beta_bins", beta_bins)
        self.qrf_n_estimators = _integer("qrf_n_estimators", qrf_n_estimators)
        if self.qrf_n_estimators == 1:
            raise ValueError(
                "qrf_n_estimators must be at least 2: sklearn-quantile 0.1.1 "
                "can return nonfinite quantiles with a single estimator."
            )
        self.qrf_max_depth = None if qrf_max_depth is None else _integer("qrf_max_depth", qrf_max_depth)
        self.min_samples_per_node = _integer("min_samples_per_node", min_samples_per_node, minimum=0)
        if not isinstance(use_beta_search, (bool, np.bool_)):
            raise ValueError("use_beta_search must be a boolean.")
        self.use_beta_search = bool(use_beta_search)
        self.seed = _integer("seed", seed, minimum=0)
        if self.seed >= 2 ** 32:
            raise ValueError("seed must be less than 2**32.")
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.ks_block_size = _integer("ks_block_size", ks_block_size)
        self.max_cache_memory_mb = _number("max_cache_memory_mb", max_cache_memory_mb, strict_min=True)
        self._fitted = False

    def _seed_for(self, stream, tree_index, step=0):
        return int(np.random.SeedSequence([self.seed, stream, tree_index, step]).generate_state(1)[0])

    @staticmethod
    def _load_qrf():
        try:
            from sklearn_quantile import RandomForestQuantileRegressor
        except ImportError as exc:
            raise ImportError(
                "DistMatch requires sklearn-quantile. Install the DistMatch optional "
                "dependencies in the experiment environment before running this baseline."
            ) from exc
        return RandomForestQuantileRegressor

    def fit(self, residuals, *, progress=None):
        """Build the initial partition in the supplied residual units.

        Optional ``progress(stage, completed, total)`` reports matching pair
        comparisons and completed trees. Cached matching is skipped.
        """
        residuals = np.asarray(residuals)
        if residuals.ndim == 2 and residuals.shape[1] == 1:
            residuals = residuals[:, 0]
        if residuals.ndim != 1 or residuals.dtype.kind not in "iuf" or not residuals.size:
            raise ValueError("residuals must be a nonempty numeric residual vector.")
        residuals = residuals.astype(np.float64)
        if not np.isfinite(residuals).all():
            raise ValueError("residuals must contain only finite values.")
        n_pairs = len(residuals) - self.past_window_len
        if n_pairs < 1 or int(n_pairs * self.bagging_ratio) < 1:
            raise ValueError(
                "Calibration is too short: len(residuals) - past_window_len must "
                "yield at least one bootstrapped training pair."
            )
        self._fitted = False
        patches = np.ascontiguousarray(sliding_window_view(residuals, self.past_window_len)[:-1])
        self._patches = [row.copy() for row in patches]
        # Data preparation supplies both windows and targets in the same units.
        self._targets = residuals[self.past_window_len:].tolist()
        self._history = residuals[-self.past_window_len:].copy()
        self._calibration_size = len(residuals)
        self._training_pair_count = n_pairs
        self._observed_updates = 0
        self._trees = []
        sorted_patches = np.sort(patches, axis=1)
        with self._match_matrix(sorted_patches, progress=progress) as (mask, cache_info):
            if progress is not None:
                progress("trees", 0, self.n_trees)
            for tree_index in range(self.n_trees):
                rng = np.random.RandomState(self._seed_for(0, tree_index))
                bootstrap = rng.choice(n_pairs, int(n_pairs * self.bagging_ratio), replace=True)
                self._trees.append(self._build_tree(bootstrap, sorted_patches, mask))
                if progress is not None:
                    progress("trees", tree_index + 1, self.n_trees)
            self._cache_info = cache_info
        self._fitted = True
        return self

    def _fill_match_matrix(self, mask, sorted_patches, progress=None):
        total = len(sorted_patches) * (len(sorted_patches) + 1) // 2
        completed = 0
        if progress is not None:
            progress("matching", completed, total)
        left_ranks, right_ranks = _tie_ranks(sorted_patches)
        for row_index, anchor in enumerate(sorted_patches):
            for start in range(row_index, len(sorted_patches), self.ks_block_size):
                stop = min(start + self.ks_block_size, len(sorted_patches))
                distances = _ks_distances_sorted(
                    anchor, sorted_patches[start:stop], left_ranks[start:stop], right_ranks[start:stop]
                )
                matches = distances < self.match_threshold
                mask[row_index, start:stop] = matches
                mask[start:stop, row_index] = matches
                if progress is not None:
                    completed += stop - start
                    progress("matching", completed, total)

    @staticmethod
    def _close_matrix(matrix):
        if isinstance(matrix, np.memmap):
            matrix._mmap.close()

    @contextmanager
    def _match_matrix(self, sorted_patches, progress=None):
        settings = dict(
            version=_CACHE_VERSION, window=self.past_window_len,
            threshold=self.match_threshold,
            shape=sorted_patches.shape,
        )
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8"))
        digest.update(np.ascontiguousarray(sorted_patches, dtype="<f8").tobytes())
        fingerprint = digest.hexdigest()
        shape = (len(sorted_patches), len(sorted_patches))
        byte_count = int(np.prod(shape, dtype=np.int64))
        info = dict(fingerprint=fingerprint, bytes=byte_count, path=None, hit=False)
        if self.cache_dir is None and byte_count <= self.max_cache_memory_mb * 1024 ** 2:
            matrix = np.empty(shape, dtype=np.bool_)
            self._fill_match_matrix(matrix, sorted_patches, progress=progress)
            info["backend"] = "memory"
            yield matrix, info
            return

        temporary_dir = None
        if self.cache_dir is None:
            temporary_dir = tempfile.TemporaryDirectory(prefix="dscp-distmatch-")
            cache_dir = Path(temporary_dir.name)
            info["backend"] = "temporary_memmap"
        else:
            cache_dir = self.cache_dir
            cache_dir.mkdir(parents=True, exist_ok=True)
            info["backend"] = "disk_memmap"
        path = cache_dir / f"ks-{fingerprint}.npy"
        if self.cache_dir is not None:
            info["path"] = str(path.resolve())
        matrix = None
        partial_path = None
        try:
            if path.is_file():
                try:
                    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
                    if matrix.shape != shape or matrix.dtype != np.bool_:
                        self._close_matrix(matrix)
                        matrix = None
                    else:
                        info["hit"] = True
                except (ValueError, OSError):
                    matrix = None
            if matrix is None:
                descriptor, filename = tempfile.mkstemp(prefix=f"ks-{fingerprint}-", suffix=".npy", dir=cache_dir)
                os.close(descriptor)
                partial_path = Path(filename)
                matrix = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.bool_, shape=shape)
                self._fill_match_matrix(matrix, sorted_patches, progress=progress)
                matrix.flush()
                self._close_matrix(matrix)
                matrix = None
                try:
                    os.replace(partial_path, path)
                except PermissionError:
                    # Another worker may already be reading the same completed
                    # cache on Windows. Its content has this same fingerprint.
                    if not path.is_file():
                        raise
                matrix = np.load(path, mmap_mode="r", allow_pickle=False)
                if matrix.shape != shape or matrix.dtype != np.bool_:
                    raise ValueError(f"Invalid DistMatch cache: {path}")
            yield matrix, info
        finally:
            if matrix is not None:
                self._close_matrix(matrix)
            if partial_path is not None and partial_path.exists():
                partial_path.unlink()
            if temporary_dir is not None:
                temporary_dir.cleanup()

    def _build_tree(self, bootstrap, sorted_patches, mask):
        root = _Node()
        leaves = []
        depth = 0
        stack = [(root, np.arange(len(bootstrap)), (), 0)]
        while stack:
            node, subset, ancestors, level = stack.pop()
            depth = max(depth, level)
            max_split = 0
            chosen = matched = unmatched = None
            original_ids = bootstrap[subset]
            for local_id in subset:
                right_mask = np.asarray(mask[bootstrap[local_id], original_ids], dtype=bool)
                count = int(right_mask.sum())
                if count < self.min_samples_per_node or len(subset) - count < self.min_samples_per_node:
                    continue
                if count > max_split:
                    max_split = count
                    chosen = int(local_id)
                    matched = np.setdiff1d(subset[right_mask], np.array([local_id]))
                    unmatched = subset[~right_mask]
                if max_split == len(subset):
                    break
            if unmatched is None or len(unmatched) == 0:
                local_members = np.concatenate([subset, np.asarray(ancestors, dtype=int)])
                node.member_ids = bootstrap[local_members].tolist()
                leaves.append(node)
                continue
            node.anchor_local_id = chosen
            node.anchor = sorted_patches[bootstrap[chosen]].copy()
            node.left, node.right = _Node(), _Node()
            next_ancestors = (chosen,) + ancestors
            stack.append((node.left, unmatched, next_ancestors, level + 1))
            stack.append((node.right, matched, next_ancestors, level + 1))
        return _Tree(root=root, bootstrap_indices=bootstrap.copy(), depth=depth, leaves=leaves)

    def _require_fit(self):
        if not self._fitted:
            raise RuntimeError("Fit the DistMatch estimator before using it.")

    def _route(self, tree, sorted_query, ranks):
        node = tree.root
        while node.member_ids is None:
            distance = _ks_distances_sorted(node.anchor, sorted_query, *ranks)[0]
            node = node.right if distance < self.match_threshold else node.left
        return node

    @staticmethod
    def _quantile_pair(pair):
        try:
            values = tuple(pair)
        except TypeError as exc:
            raise ValueError("Each target quantile pair must contain two numbers.") from exc
        if len(values) != 2:
            raise ValueError("Each target quantile pair must contain two numbers.")
        lower = _number("lower quantile", values[0], maximum=1)
        upper = _number("upper quantile", values[1], maximum=1)
        if lower >= upper:
            raise ValueError("Target quantiles must satisfy 0 <= lower < upper <= 1.")
        return lower, upper

    @staticmethod
    def _correct_qrf_endpoints(forest, query, quantiles, predictions):
        """Evaluate q=0/1 exactly on the fitted forest's conditional support.

        sklearn-quantile 0.1.1 accumulates its CDF in float32. A sum just below
        one can leave q=1 at an unfilled zero, despite all supported targets
        being positive. The empirical distribution's endpoints are its minimum
        and maximum positive-weight targets. Interior quantiles remain exactly
        as returned by the dependency; this is not a quantile rearrangement.
        """
        endpoints = (quantiles == 0) | (quantiles == 1)
        if not endpoints.any():
            return predictions
        query_leaves = forest.apply(query).T
        support = np.any(
            (forest.y_train_leaves_ == query_leaves) & (forest.y_weights_ > 0),
            axis=0,
        )
        values = np.asarray(forest.y_train_).reshape(-1)[support]
        if not values.size or not np.isfinite(values).all():
            raise ValueError("DistMatch QRF has no finite positive-weight conditional support.")
        corrected = predictions.copy()
        corrected[quantiles == 0] = values.min()
        corrected[quantiles == 1] = values.max()
        return corrected

    def predict_intervals(self, target_quantiles):
        """Return ``{(q_low, q_high): (low, high, per_tree_betas)}`` in raw units.

        All quantile pairs share one QRF fit per tree at this observation count.
        Seeds depend only on the configured seed, tree, and observation count,
        so repeated predictions and a different quantile-pair order agree.
        """
        self._require_fit()
        pairs = sorted({self._quantile_pair(pair) for pair in target_quantiles})
        if not pairs:
            raise ValueError("At least one target quantile pair is required.")
        grids = {}
        for pair in pairs:
            width = pair[1] - pair[0]
            alpha = 1.0 - width
            betas = np.linspace(0, alpha, self.beta_bins) if self.use_beta_search else np.array([pair[0]])
            grids[pair] = (betas, np.clip(width + betas, 0, 1))
        quantiles = np.unique(np.concatenate([values for grid in grids.values() for values in grid]))
        qrf_class = self._load_qrf()
        sorted_query = np.sort(self._history)[None, :]
        ranks = _tie_ranks(sorted_query)
        per_pair = {pair: [] for pair in pairs}
        for tree_index, tree in enumerate(self._trees):
            leaf = self._route(tree, sorted_query, ranks)
            xs = np.asarray([self._patches[index] for index in leaf.member_ids], dtype=np.float64)
            ys = np.asarray([self._targets[index] for index in leaf.member_ids], dtype=np.float64)
            model = qrf_class(
                n_estimators=self.qrf_n_estimators, max_depth=self.qrf_max_depth,
                criterion="squared_error", q=quantiles, n_jobs=1,
                random_state=self._seed_for(1, tree_index, self._observed_updates),
            )
            query = self._history[None, :]
            predictions = np.asarray(model.fit(xs, ys).predict(query), dtype=np.float64).reshape(-1)
            if predictions.size != quantiles.size:
                raise ValueError("DistMatch QRF returned invalid quantile predictions.")
            predictions = self._correct_qrf_endpoints(model, query, quantiles, predictions)
            if not np.isfinite(predictions).all():
                raise ValueError("DistMatch QRF returned invalid quantile predictions.")
            for pair, (betas, highs) in grids.items():
                lows = predictions[np.searchsorted(quantiles, betas)]
                uppers = predictions[np.searchsorted(quantiles, highs)]
                best = int(np.argmin(uppers - lows))
                if uppers[best] < lows[best]:
                    raise DistMatchCrossedBoundsError(
                        tree_index=tree_index, triggering_quantile_pair=list(pair),
                        beta=float(betas[best]), lower_quantile=float(betas[best]),
                        upper_quantile=float(highs[best]), lower_bound=float(lows[best]),
                        upper_bound=float(uppers[best]),
                    )
                per_pair[pair].append((float(lows[best]), float(uppers[best]), float(betas[best])))
        return {
            pair: (float(np.mean([v[0] for v in values])), float(np.mean([v[1] for v in values])), [v[2] for v in values])
            for pair, values in per_pair.items()
        }

    def observe(self, residual):
        """Insert one new window/target pair in the same residual units as fit."""
        self._require_fit()
        if isinstance(residual, (bool, np.bool_)) or not isinstance(residual, Real) or not np.isfinite(residual):
            raise ValueError("residual must be a finite number.")
        residual = float(residual)
        sorted_query = np.sort(self._history)[None, :]
        ranks = _tie_ranks(sorted_query)
        leaves = [self._route(tree, sorted_query, ranks) for tree in self._trees]
        new_id = len(self._targets)
        self._patches.append(self._history.copy())
        self._targets.append(residual)
        for leaf in leaves:
            leaf.member_ids.append(new_id)
        self._history[:-1] = self._history[1:]
        self._history[-1] = residual
        self._observed_updates += 1
        return self

    @property
    def memory_size(self):
        self._require_fit()
        return len(self._targets)

    @property
    def tree_depths(self):
        self._require_fit()
        return [tree.depth for tree in self._trees]

    def diagnostics(self):
        self._require_fit()
        return dict(
            upstream_commit=UPSTREAM_COMMIT, calibration_size=self._calibration_size,
            training_pair_count=self._training_pair_count, memory_size=self.memory_size,
            observed_updates=self._observed_updates, tree_depths=self.tree_depths,
            tree_leaf_counts=[len(tree.leaves) for tree in self._trees],
            leaf_sizes=[[len(leaf.member_ids) for leaf in tree.leaves] for tree in self._trees],
            cache=dict(self._cache_info), fixed_tree_structure=True,
            match_comparison="strict_less_than", rng_policy="local_seed_by_tree_and_observation",
        )
