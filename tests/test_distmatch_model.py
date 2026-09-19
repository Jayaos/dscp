"""Independent KS/tree reference and causal streaming tests for DistMatch."""

import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import ks_2samp

from baselines.distmatch.model import (
    DistMatchResidualIntervalEstimator,
    _ks_distances_sorted,
)


def reference_tree(patches, bootstrap, gamma, minimum=0):
    """Slow direct SciPy reference of upstream's anchor/ancestor rules."""
    def build(members, ancestors):
        best_size, best_anchor, best_right, best_left = 0, None, None, None
        for anchor_id in members:
            matching = [
                member for member in members
                if ks_2samp(patches[bootstrap[anchor_id]], patches[bootstrap[member]]).statistic < gamma
            ]
            other = [member for member in members if member not in matching]
            if len(matching) < minimum or len(other) < minimum:
                continue
            if len(matching) > best_size:
                best_size = len(matching)
                best_anchor = anchor_id
                best_right = sorted(member for member in matching if member != anchor_id)
                best_left = other
            if best_size == len(members):
                break
        if not best_left:
            return {"ids": [int(bootstrap[member]) for member in members + ancestors]}
        return {
            "anchor_id": best_anchor,
            "anchor": patches[bootstrap[best_anchor]],
            "left": build(best_left, [best_anchor] + ancestors),
            "right": build(best_right, [best_anchor] + ancestors),
        }
    return build(list(range(len(bootstrap))), [])


def reference_leaf(tree, query, gamma):
    while "ids" not in tree:
        matching = ks_2samp(query, tree["anchor"]).statistic < gamma
        tree = tree["right"] if matching else tree["left"]
    return tree["ids"]


class DistMatchModelTests(unittest.TestCase):
    RESIDUALS = np.array([
        -4, -3, -4, -2, -3, -1, 0, 1, 0, 2, 1, 4, 6, 5, 7,
        4, 6, 3, 2, -1, -2, -4, -2, 0, 2, 3, 5, 3, 1, 2,
    ], dtype=float)

    @staticmethod
    def model(**overrides):
        config = dict(
            past_window_len=3, match_threshold=0.34, n_trees=3,
            bagging_ratio=0.9, beta_bins=5, qrf_n_estimators=4,
            qrf_max_depth=2, ks_block_size=4, seed=17,
        )
        config.update(overrides)
        return DistMatchResidualIntervalEstimator(**config)

    def assert_tree_matches(self, actual, expected):
        if "ids" in expected:
            self.assertEqual(actual.member_ids, expected["ids"])
            self.assertIsNone(actual.anchor)
        else:
            self.assertEqual(actual.anchor_local_id, expected["anchor_id"])
            np.testing.assert_array_equal(actual.anchor, np.sort(expected["anchor"]))
            self.assert_tree_matches(actual.left, expected["left"])
            self.assert_tree_matches(actual.right, expected["right"])

    def test_ks_matches_scipy_with_ties_disjoint_and_continuous_windows(self):
        rng = np.random.default_rng(36)
        for width in (1, 2, 3, 7, 100):
            rows = np.concatenate([
                rng.integers(-3, 4, size=(13, width)).astype(float),
                rng.normal(size=(9, width)),
                np.full((1, width), -100.0), np.full((1, width), 100.0),
            ])
            rows = np.sort(rows, axis=1)
            for anchor in rows:
                expected = [ks_2samp(anchor, row).statistic for row in rows]
                np.testing.assert_array_equal(_ks_distances_sorted(anchor, rows), expected)

    def test_ks_threshold_is_strict_at_exact_boundary(self):
        rows = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]], dtype=float)
        model = self.model(past_window_len=4, match_threshold=0.25)
        matrix = np.empty((3, 3), dtype=bool)
        model._fill_match_matrix(matrix, rows)
        np.testing.assert_array_equal(matrix, np.eye(3, dtype=bool))
        self.assertEqual(_ks_distances_sorted(rows[0], rows)[1], 0.25)

    def test_topology_and_leaf_order_match_independent_upstream_reference(self):
        for minimum in (0, 2):
            model = self.model(min_samples_per_node=minimum).fit(self.RESIDUALS)
            patches = np.asarray(model._patches)
            for tree in model._trees:
                expected = reference_tree(patches, tree.bootstrap_indices, model.match_threshold, minimum)
                self.assert_tree_matches(tree.root, expected)
            self.assertTrue(any(depth > 0 for depth in model.tree_depths))

    def test_topology_matches_execution_of_cloned_upstream_tree(self):
        source_path = (
            Path(__file__).resolve().parents[1] / "dist_match_conformal" /
            "code/models/uncertainty/dist_match/tree.py"
        )
        if not source_path.is_file():
            self.skipTest("Optional official DistMatch clone is unavailable.")
        source = ast.parse(source_path.read_text(encoding="utf-8"))
        # The QRF tree never uses the optional XGBoost branch. Remove only that
        # import; execute the original upstream partition methods unchanged.
        source.body = [
            statement for statement in source.body
            if not (isinstance(statement, ast.Import) and any(
                name.name == "xgboost" for name in statement.names
            ))
        ]
        namespace = {"__name__": "_distmatch_upstream_tree_fixture"}
        exec(compile(source, str(source_path), "exec"), namespace)
        upstream_tree_class = namespace["DistMatchTree"]
        model = self.model(n_trees=2).fit(self.RESIDUALS)
        patches = np.asarray(model._patches)
        targets = np.asarray(model._targets)

        def compare(actual, upstream_node, bootstrap):
            split = upstream_node.get_split_value()
            if split is None:
                ids = upstream_node.get_values()[2]
                self.assertEqual(actual.member_ids, bootstrap[ids].tolist())
            else:
                self.assertEqual(actual.anchor_local_id, split[2])
                np.testing.assert_array_equal(actual.anchor, np.sort(split[0]))
                compare(actual.left, upstream_node.left, bootstrap)
                compare(actual.right, upstream_node.right, bootstrap)

        def matches(first, second):
            return ks_2samp(first, second).statistic < model.match_threshold

        for tree in model._trees:
            bootstrap = tree.bootstrap_indices
            sampled = patches[bootstrap]
            mask = np.array([[matches(first, second) for second in sampled] for first in sampled])
            upstream = upstream_tree_class(
                matcher=matches, feature_dim=-1, quantiles=np.array([0.1, 0.9]),
                match_mask=mask[None, ...], min_samples_per_node=0,
            )
            # A univariate tree always chooses dimension zero; avoid touching
            # process RNG while retaining the same upstream partition path.
            with patch.object(upstream_tree_class, "_sample_dim", return_value=0):
                upstream.fit(sampled[..., None], targets[bootstrap], preserve_match_mask=True)
            compare(tree.root, upstream.root, bootstrap)
            self.assertEqual(tree.depth, upstream.depth)

    def test_windows_targets_and_pre_observation_update_are_aligned(self):
        model = self.model().fit(self.RESIDUALS)
        width = model.past_window_len
        np.testing.assert_array_equal(model._patches, np.array([
            self.RESIDUALS[i:i + width] for i in range(len(self.RESIDUALS) - width)
        ]))
        np.testing.assert_array_equal(model._targets, self.RESIDUALS[width:])
        original_sizes = [sum(len(leaf.member_ids) for leaf in tree.leaves) for tree in model._trees]
        original_depths = model.tree_depths
        before = model._history.copy()
        model.observe(123.0)
        self.assertEqual(model.memory_size, len(self.RESIDUALS) - width + 1)
        np.testing.assert_array_equal(model._patches[-1], before)
        self.assertEqual(model._targets[-1], 123.0)
        np.testing.assert_array_equal(model._history, [1.0, 2.0, 123.0])
        self.assertEqual(model.tree_depths, original_depths)
        for tree, original in zip(model._trees, original_sizes):
            self.assertEqual(sum(len(leaf.member_ids) for leaf in tree.leaves), original + 1)
            self.assertEqual(sum(leaf.member_ids.count(model.memory_size - 1) for leaf in tree.leaves), 1)

    def test_supplied_residual_units_are_preserved_during_fit_and_observe(self):
        residuals = self.RESIDUALS * 10 + 70
        model = self.model().fit(residuals)
        np.testing.assert_array_equal(model._patches[0], residuals[:3])
        np.testing.assert_array_equal(model._targets, residuals[3:])
        np.testing.assert_array_equal(model._history, residuals[-3:])
        model.observe(10000.0)
        np.testing.assert_array_equal(model._patches[-1], residuals[-3:])
        self.assertEqual(model._targets[-1], 10000.0)
        self.assertEqual(model._history[-1], 10000.0)

    def test_qrf_endpoints_match_independent_scipy_tree_reference(self):
        from sklearn_quantile import RandomForestQuantileRegressor
        model = self.model().fit(self.RESIDUALS)
        pair = (0.1, 0.9)
        alpha = 1 - (pair[1] - pair[0])
        betas = np.linspace(0, alpha, model.beta_bins)
        upper_levels = pair[1] - pair[0] + betas
        levels = np.unique(np.concatenate([betas, upper_levels]))
        patches = np.asarray(model._patches)
        targets = np.asarray(model._targets)
        expected = []
        for index, tree in enumerate(model._trees):
            reference = reference_tree(patches, tree.bootstrap_indices, model.match_threshold)
            ids = reference_leaf(reference, model._history, model.match_threshold)
            forest = RandomForestQuantileRegressor(
                n_estimators=4, max_depth=2, criterion="squared_error", q=levels,
                n_jobs=1, random_state=model._seed_for(1, index, 0),
            ).fit(patches[ids], targets[ids])
            quantiles = forest.predict(model._history[None, :]).reshape(-1)
            lower = quantiles[np.searchsorted(levels, betas)]
            upper = quantiles[np.searchsorted(levels, upper_levels)]
            chosen = np.argmin(upper - lower)
            expected.append((lower[chosen], upper[chosen], betas[chosen]))
        actual = model.predict_intervals([pair])[pair]
        np.testing.assert_allclose(actual[:2], np.mean(np.asarray(expected)[:, :2], axis=0), rtol=0, atol=1e-7)
        np.testing.assert_array_equal(actual[2], np.asarray(expected)[:, 2])

    def test_repeated_and_reordered_alpha_predictions_leave_state_unchanged(self):
        model = self.model().fit(self.RESIDUALS)
        before = model.diagnostics()
        history = model._history.copy()
        pairs = [(0.1, 0.9), (0.05, 0.95), (0.02, 0.82)]
        together = model.predict_intervals(pairs)
        self.assertEqual(together, model.predict_intervals(list(reversed(pairs))))
        for pair in pairs:
            self.assertEqual(together[pair], model.predict_intervals([pair])[pair])
        self.assertEqual(model.diagnostics(), before)
        np.testing.assert_array_equal(model._history, history)

    def test_fixed_tail_keeps_requested_asymmetry(self):
        model = self.model(use_beta_search=False).fit(self.RESIDUALS)
        actual = model.predict_intervals([(0.02, 0.82)])[(0.02, 0.82)]
        self.assertEqual(actual[2], [0.02] * model.n_trees)
        self.assertLessEqual(actual[0], actual[1])

    def test_single_qrf_estimator_is_rejected_for_dependency_compatibility(self):
        with self.assertRaisesRegex(ValueError, "sklearn-quantile.*single estimator"):
            self.model(qrf_n_estimators=1)
        model = self.model(qrf_n_estimators=2).fit(self.RESIDUALS)
        interval = model.predict_intervals([(0.1, 0.9)])[(0.1, 0.9)]
        self.assertTrue(np.isfinite(interval[:2]).all())
        self.assertLessEqual(interval[0], interval[1])

    def test_qrf_endpoint_cdf_underflow_uses_exact_support_without_changing_interior(self):
        from sklearn_quantile import RandomForestQuantileRegressor
        xs = np.zeros((1000, 3))
        ys = np.arange(1, 1001, dtype=float)
        quantiles = np.array([0, 0.05, 0.95, 1])
        forest = RandomForestQuantileRegressor(
            n_estimators=10, max_depth=2, q=quantiles, random_state=1, n_jobs=1,
        ).fit(xs, ys)
        raw = forest.predict(xs[:1]).reshape(-1).astype(float)
        # In sklearn-quantile 0.1.1 this ordinary positive-target fixture returns
        # zero at q=1. A future fixed version may already return the endpoint.
        corrected = self.model()._correct_qrf_endpoints(forest, xs[:1], quantiles, raw)
        support = []
        for target_index, target in enumerate(forest.y_train_.reshape(-1)):
            for tree_index, leaf_id in enumerate(forest.apply(xs[:1])[0]):
                if (forest.y_train_leaves_[tree_index, target_index] == leaf_id
                        and forest.y_weights_[tree_index, target_index] > 0):
                    support.append(target)
                    break
        self.assertGreater(min(support), 0)
        self.assertEqual(corrected[0], min(support))
        self.assertEqual(corrected[-1], max(support))
        np.testing.assert_array_equal(corrected[1:-1], raw[1:-1])
        self.assertTrue(np.all(np.diff(corrected) >= 0))
        self.assertFalse(np.shares_memory(corrected, raw))

    def test_qrf_endpoints_use_conditional_support_instead_of_all_training_targets(self):
        from sklearn_quantile import RandomForestQuantileRegressor
        xs = np.arange(120, dtype=float)[:, None]
        ys = 10 + xs[:, 0] ** 2
        quantiles = np.array([0, 0.1, 0.9, 1])
        forest = RandomForestQuantileRegressor(
            n_estimators=10, max_depth=2, q=quantiles, random_state=3, n_jobs=1,
        ).fit(xs, ys)
        query = xs[-1:]
        raw = forest.predict(query).reshape(-1).astype(float)
        corrected = self.model()._correct_qrf_endpoints(forest, query, quantiles, raw)
        support = np.any(
            (forest.y_train_leaves_ == forest.apply(query).T) & (forest.y_weights_ > 0), axis=0,
        )
        supported_targets = forest.y_train_[support, 0]
        self.assertGreater(supported_targets.min(), ys.min())
        self.assertEqual(corrected[0], supported_targets.min())
        self.assertEqual(corrected[-1], supported_targets.max())
        np.testing.assert_array_equal(corrected[1:-1], raw[1:-1])

    def test_causal_prefix_and_extra_predictions_do_not_change_future_state(self):
        first = self.model().fit(self.RESIDUALS)
        second = self.model().fit(self.RESIDUALS)
        pairs = [(0.1, 0.9), (0.05, 0.95)]
        for residual in (10.0, -2.0, 0.0):
            self.assertEqual(first.predict_intervals(pairs), second.predict_intervals(pairs))
            first.predict_intervals([pairs[1]])
            first.observe(residual)
            second.observe(residual)
        self.assertEqual(first.predict_intervals(pairs), second.predict_intervals(pairs))
        first.observe(-1000.0)
        second.observe(1000.0)
        self.assertFalse(np.array_equal(first._history, second._history))

    def test_refit_is_reproducible_without_process_rng_side_effects(self):
        state = np.random.get_state()
        model = self.model().fit(self.RESIDUALS)
        first = model.predict_intervals([(0.1, 0.9)])
        model.observe(5.0)
        refit = model.fit(self.RESIDUALS).predict_intervals([(0.1, 0.9)])
        self.assertEqual(first, refit)
        afterwards = np.random.get_state()
        self.assertEqual(state[0], afterwards[0])
        np.testing.assert_array_equal(state[1], afterwards[1])
        self.assertEqual(state[2:], afterwards[2:])

    def test_boolean_cache_reuse_fingerprint_and_memory_backend_parity(self):
        memory = self.model(ks_block_size=1).fit(self.RESIDUALS)
        with tempfile.TemporaryDirectory() as directory:
            disk = self.model(cache_dir=directory, ks_block_size=7).fit(self.RESIDUALS)
            info = disk.diagnostics()["cache"]
            self.assertEqual(info["backend"], "disk_memmap")
            self.assertFalse(info["hit"])
            self.assertEqual(np.load(info["path"]).dtype, np.bool_)
            self.assertEqual(info["bytes"], memory.memory_size ** 2)
            with patch.object(DistMatchResidualIntervalEstimator, "_fill_match_matrix", side_effect=AssertionError("cache miss")):
                reused = self.model(cache_dir=directory).fit(self.RESIDUALS)
            self.assertTrue(reused.diagnostics()["cache"]["hit"])
            changed = self.model(cache_dir=directory).fit(self.RESIDUALS + 1)
            self.assertNotEqual(info["fingerprint"], changed.diagnostics()["cache"]["fingerprint"])
            for actual, reference in zip(disk._trees, memory._trees):
                self.assertEqual([leaf.member_ids for leaf in actual.leaves], [leaf.member_ids for leaf in reference.leaves])
            self.assertFalse(any("mask" in key for key in disk.__dict__))
            self.assertFalse(any(path.name.endswith(".tmp") for path in Path(directory).iterdir()))

    def test_memory_budget_uses_temporary_memmap(self):
        model = self.model(max_cache_memory_mb=0.00001).fit(self.RESIDUALS)
        info = model.diagnostics()["cache"]
        self.assertEqual(info["backend"], "temporary_memmap")
        self.assertIsNone(info["path"])
        self.assertEqual(model.memory_size, len(self.RESIDUALS) - 3)

    def test_one_qrf_fit_per_tree_for_many_alphas_and_no_fit_on_observe(self):
        from sklearn_quantile import RandomForestQuantileRegressor
        fit_calls = []

        class CountingQRF(RandomForestQuantileRegressor):
            def fit(self, *args, **kwargs):
                fit_calls.append(self)
                return super().fit(*args, **kwargs)

        with patch.object(DistMatchResidualIntervalEstimator, "_load_qrf", return_value=CountingQRF) as load:
            model = self.model().fit(self.RESIDUALS)
            load.assert_not_called()
            model.predict_intervals([(0.1, 0.9), (0.05, 0.95)])
            self.assertEqual(len(fit_calls), model.n_trees)
            model.observe(8.0)
            self.assertEqual(len(fit_calls), model.n_trees)

    def test_validation_and_short_constant_calibration(self):
        for values in ([1, 2, 3], [1, 2, 3, 4], [1, 2, np.nan, 4, 5]):
            with self.assertRaises(ValueError):
                self.model().fit(values)
        model = self.model().fit([5.0] * 10)
        result = model.predict_intervals([(0.1, 0.9)])[(0.1, 0.9)]
        self.assertEqual(result[:2], (5.0, 5.0))
        with self.assertRaises(ValueError):
            model.observe(float("inf"))
        with self.assertRaises(ValueError):
            model.predict_intervals([(0.9, 0.1)])
        with self.assertRaises(RuntimeError):
            self.model().observe(1.0)


if __name__ == "__main__":
    unittest.main()
