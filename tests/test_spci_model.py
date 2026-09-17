"""Forest choice, configuration, and repeatability for SPCI."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
from sklearn_quantile import RandomForestQuantileRegressor, SampleRandomForestQuantileRegressor
from sklearn_quantile.ensemble import quantile as upstream_quantile

from baselines.spci.model import (
    RobustSampleRandomForestQuantileRegressor,
    _repair_sampled_leaves,
    build_quantile_forest,
)


def _fake_sampled_tree(seed=0):
    # Leaf 1 is already valid; leaf 2 needs repair. The zero-weight observation
    # and out-of-bag observation must never enter its sampled support.
    return SimpleNamespace(
        random_state=seed,
        tree_=SimpleNamespace(
            node_count=3,
            children_left=np.array([1, -1, -1], dtype=np.intp),
            children_right=np.array([2, -1, -1], dtype=np.intp),
            value=np.array([np.nan, 7, np.nan], dtype=np.float64).reshape(3, 1, 1),
        ),
        y_train_leaves_=np.array([1, 2, 2, 2, -1, 2], dtype=np.intp),
        y_train_=np.array([7, 11, 22, 9999, -8888, 33], dtype=np.float32)[:, None],
        y_weights_=np.array([1, 0.2, 0.3, 0, 100, 0.5], dtype=np.float32),
    )


class SPCIForestTests(unittest.TestCase):
    @staticmethod
    def config():
        return OmegaConf.create({
            "model": {"n_estimators": 5, "max_depth": 2, "criterion": "squared_error"},
        })

    def test_exact_to_sampled_boundary_and_legacy_defaults(self):
        for count, expected_type in ((10_000, RandomForestQuantileRegressor), (10_001, SampleRandomForestQuantileRegressor)):
            with self.subTest(count=count):
                forest = build_quantile_forest(self.config(), count, [0.05, 0.95])
                self.assertIsInstance(forest, expected_type)
                if count > 10_000:
                    self.assertIsInstance(forest, RobustSampleRandomForestQuantileRegressor)
                self.assertEqual(forest.n_estimators, 5)
                self.assertEqual(forest.max_depth, 2)
                self.assertEqual(forest.criterion, "squared_error")
                self.assertEqual(forest.n_jobs, -1)
                self.assertIsNone(forest.random_state)
                np.testing.assert_array_equal(forest.q, [0.05, 0.95])

    def test_seed_threads_and_unlimited_depth_are_forwarded_to_both_forests(self):
        config = self.config()
        config.seed = 17
        config.model.n_jobs = 2
        config.model.max_depth = None
        for count in (20, 10_001):
            forest = build_quantile_forest(config, count, [0.1, 0.9])
            self.assertEqual(forest.random_state, 17)
            self.assertEqual(forest.n_jobs, 2)
            self.assertIsNone(forest.max_depth)

    def test_explicit_seed_repeats_predictions_without_relying_on_global_rng(self):
        config = self.config()
        config.seed = 17
        config.model.n_jobs = 1
        generator = np.random.default_rng(8)
        features = generator.normal(size=(80, 4)).astype(np.float32)
        residuals = (features[:, 0] + generator.normal(size=80)).astype(np.float32)
        first = build_quantile_forest(config, len(features), [0.1, 0.9]).fit(features, residuals)
        np.random.seed(999)
        second = build_quantile_forest(config, len(features), [0.1, 0.9]).fit(features, residuals)
        np.testing.assert_allclose(first.predict(features[:8]), second.predict(features[:8]))

    def test_float32_upstream_sampling_failure_is_repaired_from_observed_support(self):
        # Repeated float32 subtraction misses the last observation despite
        # sufficient positive mass. The fixture requires no forest training.
        size = 11_000
        estimator = SimpleNamespace(
            random_state=18_985,
            tree_=SimpleNamespace(
                node_count=1,
                children_left=np.array([-1], dtype=np.intp),
                children_right=np.array([-1], dtype=np.intp),
                value=np.zeros((1, 1, 1), dtype=np.float64),
            ),
            y_train_leaves_=np.zeros(size, dtype=np.intp),
            y_train_=np.arange(size, dtype=np.float32)[:, None],
            y_weights_=np.full(size, 1 / size, dtype=np.float32),
        )
        draw = np.random.RandomState(estimator.random_state).rand(1).astype(np.float32)[0]
        self.assertGreater(estimator.y_weights_.sum(dtype=np.float64), draw)
        upstream_quantile._fit_sample_tree(estimator)
        self.assertTrue(np.isnan(estimator.tree_.value[0, 0, 0]))
        self.assertEqual(_repair_sampled_leaves(estimator), 1)
        self.assertEqual(estimator.tree_.value[0, 0, 0], size - 1)

    def test_repair_preserves_valid_leaves_and_uses_original_weighted_node_draw(self):
        # These seeds place node 2's draw in the first, second, and third
        # positive-weight observations. This also catches using draw[0].
        for seed, expected in ((1, 11), (3, 22), (0, 33)):
            with self.subTest(seed=seed):
                estimator = _fake_sampled_tree(seed)
                original_weights = estimator.y_weights_.copy()
                original_targets = estimator.y_train_.copy()
                self.assertEqual(_repair_sampled_leaves(estimator), 1)
                self.assertEqual(estimator.tree_.value[1, 0, 0], 7.0)
                self.assertEqual(estimator.tree_.value[2, 0, 0], expected)
                self.assertTrue(np.isnan(estimator.tree_.value[0, 0, 0]))
                np.testing.assert_array_equal(estimator.y_weights_, original_weights)
                np.testing.assert_array_equal(estimator.y_train_, original_targets)
                self.assertEqual(_repair_sampled_leaves(estimator), 0)

    def test_repair_rejects_empty_nonfinite_or_nonpositive_training_support(self):
        for damage in ("empty", "zero_weights", "negative_weights", "nonfinite_weights", "nonfinite_targets"):
            with self.subTest(damage=damage):
                estimator = _fake_sampled_tree()
                in_leaf = estimator.y_train_leaves_ == 2
                if damage == "empty":
                    estimator.y_train_leaves_[in_leaf] = -1
                elif damage == "zero_weights":
                    estimator.y_weights_[in_leaf] = 0
                elif damage == "negative_weights":
                    estimator.y_weights_[in_leaf] = -1
                elif damage == "nonfinite_weights":
                    estimator.y_weights_[in_leaf] = np.nan
                else:
                    estimator.y_train_[in_leaf] = np.nan
                with self.assertRaisesRegex(ValueError, "Cannot repair sampled quantile-forest leaf"):
                    _repair_sampled_leaves(estimator)

    def test_sampled_fit_repairs_failed_leaf_draws_and_is_reproducible_across_threads(self):
        generator = np.random.default_rng(28)
        features = generator.normal(size=(80, 4)).astype(np.float32)
        residuals = (features[:, 0] + generator.normal(size=80)).astype(np.float32)
        original_sampler = upstream_quantile._fit_sample_tree

        def sample_with_unresolved_leaf(estimator):
            original_sampler(estimator)
            estimator.test_original_values = estimator.tree_.value.copy()
            leaf = np.flatnonzero(estimator.tree_.children_left == -1)[0]
            estimator.test_damaged_leaf = leaf
            estimator.tree_.value[leaf, 0, 0] = np.nan

        predictions = []
        for jobs, global_seed in ((1, 29), (2, 700)):
            config = self.config()
            config.seed = 2029
            config.model.n_jobs = jobs
            config.model.max_depth = 10
            np.random.seed(global_seed)
            # Force the sampled factory branch while keeping this regression
            # small; the actual sampled forest still fits and predicts.
            forest = build_quantile_forest(config, 10_001, [0.1, 0.5, 0.9])
            with patch.object(upstream_quantile, "_fit_sample_tree", side_effect=sample_with_unresolved_leaf), \
                    self.assertWarnsRegex(RuntimeWarning, "Repaired 5"):
                returned = forest.fit(features, residuals)
            self.assertIs(returned, forest)
            self.assertEqual(forest.n_repaired_leaves_, 5)
            for estimator in forest.estimators_:
                leaf = estimator.test_damaged_leaf
                valid = np.arange(estimator.tree_.node_count) != leaf
                np.testing.assert_array_equal(
                    estimator.tree_.value[valid], estimator.test_original_values[valid]
                )
                positive_support = (
                    (estimator.y_train_leaves_ == leaf) & (estimator.y_weights_ > 0)
                )
                self.assertIn(estimator.tree_.value[leaf, 0, 0], estimator.y_train_[positive_support, 0])
            predicted = forest.predict(features)
            self.assertEqual(predicted.shape, (3, len(features)))
            self.assertTrue(np.isfinite(predicted).all())
            self.assertTrue((np.diff(predicted, axis=0) >= 0).all())
            self.assertGreaterEqual(predicted.min(), residuals.min())
            self.assertLessEqual(predicted.max(), residuals.max())
            predictions.append(predicted)
        np.testing.assert_array_equal(predictions[0], predictions[1])

    def test_sampled_fit_without_failed_draws_matches_upstream_predictions(self):
        generator = np.random.default_rng(5)
        features = generator.normal(size=(80, 3)).astype(np.float32)
        targets = (features[:, 0] + features[:, 1]).astype(np.float32)
        config = self.config()
        config.seed = 17
        config.model.n_jobs = 1
        forest = build_quantile_forest(config, 10_001, [0.05, 0.95])
        upstream = SampleRandomForestQuantileRegressor(**forest.get_params())
        upstream.fit(features, targets)
        forest.fit(features, targets)
        self.assertEqual(forest.n_repaired_leaves_, 0)
        self.assertTrue(np.isfinite(forest.predict(features)).all())
        np.testing.assert_array_equal(forest.predict(features), upstream.predict(features))

    def test_invalid_core_options_are_rejected_before_fit(self):
        for key, values in {
            "n_estimators": (0, -1, True, 1.5),
            "max_depth": (0, -1, True, "None"),
            "n_jobs": (0, True, 1.5),
        }.items():
            for value in values:
                config = self.config()
                config.model[key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    build_quantile_forest(config, 20, [0.1, 0.9])
        for seed in (-1, 2**32, True, 1.5):
            config = self.config()
            config.seed = seed
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                build_quantile_forest(config, 20, [0.1, 0.9])
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "n_train_samples"):
                build_quantile_forest(self.config(), count, [0.1, 0.9])

    def test_quantiles_must_be_sorted_finite_unique_probabilities(self):
        for quantiles in ([], [np.nan], [0.9, 0.1], [0.1, 0.1], [-0.1, 0.9], [0.1, 1.1], [[0.1, 0.9]]):
            with self.subTest(quantiles=quantiles), self.assertRaisesRegex(ValueError, "quantiles"):
                build_quantile_forest(self.config(), 20, quantiles)


if __name__ == "__main__":
    unittest.main()
