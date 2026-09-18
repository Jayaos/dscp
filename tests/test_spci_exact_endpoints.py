"""Exact SPCI forest endpoint quantiles use the reached leaves' support."""

import unittest

import numpy as np
from omegaconf import OmegaConf
from sklearn.ensemble._forest import _generate_sample_indices
from sklearn_quantile import RandomForestQuantileRegressor

from baselines.spci.model import RobustRandomForestQuantileRegressor, build_quantile_forest


class SPCIExactEndpointTests(unittest.TestCase):
    def test_factory_repairs_large_leaf_endpoints_without_changing_interior_quantiles(self):
        config = OmegaConf.create({
            "seed": 2026,
            "model": {
                "n_estimators": 10, "max_depth": 2,
                "criterion": "squared_error", "n_jobs": 1,
            },
        })
        features = np.zeros((1000, 1), dtype=np.float32)
        for low, high in ((1, 2), (-5, -2), (-1.5, 1.5)):
            with self.subTest(low=low, high=high):
                targets = np.linspace(low, high, len(features), dtype=np.float32)
                forest = build_quantile_forest(config, len(features), [0, .05, .95, 1])
                self.assertIsInstance(forest, RobustRandomForestQuantileRegressor)
                self.assertIsInstance(forest, RandomForestQuantileRegressor)
                self.assertIs(forest.fit(features, targets), forest)
                original = RandomForestQuantileRegressor.predict(forest, features[:2])
                predictions = forest.predict(features[:2])
                np.testing.assert_array_equal(predictions[1:-1], original[1:-1])
                supported = forest.y_train_[np.any(forest.y_weights_ > 0, axis=0), 0]
                np.testing.assert_array_equal(predictions[0], [supported.min()] * 2)
                np.testing.assert_array_equal(predictions[-1], [supported.max()] * 2)
                self.assertTrue(np.isfinite(predictions).all())
                self.assertTrue((np.diff(predictions, axis=0) >= 0).all())

    def test_endpoints_follow_each_query_leaf_and_exclude_zero_weight_and_out_of_bag_rows(self):
        random = np.random.default_rng(47)
        features = np.repeat([-1., 1.], 40).astype(np.float32)[:, None]
        targets = np.concatenate((np.linspace(-10, -2, 40), np.linspace(3, 12, 40))).astype(np.float32)
        weights = np.ones(80)
        targets[[0, 79]] = [-10_000, 10_000]
        weights[[0, 79]] = 0
        order = random.permutation(80)
        features, targets, weights = features[order], targets[order], weights[order]
        queries = np.array([[-1.], [1.]], dtype=np.float32)
        forest = RobustRandomForestQuantileRegressor(
            n_estimators=3, max_depth=1, q=[0, .25, .75, 1], random_state=13, n_jobs=1,
        ).fit(features, targets, sample_weight=weights)
        predictions = forest.predict(queries)
        original = RandomForestQuantileRegressor.predict(forest, queries)
        np.testing.assert_array_equal(predictions[1:-1], original[1:-1])

        # Reconstruct support using original (unsorted) targets and bootstrap
        # draws, independently of the sorted arrays used by the implementation.
        for column, query in enumerate(queries):
            support = []
            for estimator in forest.estimators_:
                draws = _generate_sample_indices(estimator.random_state, len(targets), len(targets))
                sampled = np.bincount(draws, minlength=len(targets)) > 0
                reached = estimator.apply(query[None])[0]
                same_leaf = estimator.apply(features) == reached
                support.extend(targets[sampled & (weights > 0) & same_leaf])
            self.assertEqual(predictions[0, column], min(support))
            self.assertEqual(predictions[-1, column], max(support))
        self.assertLess(predictions[-1, 0], 0)
        self.assertGreater(predictions[0, 1], 0)

    def test_single_quantile_shapes_and_later_quantile_changes_match_parent_api(self):
        features = np.zeros((1000, 1), dtype=np.float32)
        targets = np.linspace(-3, 4, 1000, dtype=np.float32)
        forest = RobustRandomForestQuantileRegressor(
            n_estimators=10, max_depth=2, q=.5, random_state=2026, n_jobs=1,
        ).fit(features, targets)
        for quantile in (.5, 0., 1.):
            with self.subTest(quantile=quantile):
                forest.q = quantile
                predictions = forest.predict(features[:3])
                self.assertEqual(predictions.shape, (3,))
                if quantile == .5:
                    np.testing.assert_array_equal(
                        predictions, RandomForestQuantileRegressor.predict(forest, features[:3])
                    )
                else:
                    supported = forest.y_train_[np.any(forest.y_weights_ > 0, axis=0), 0]
                    expected = supported.min() if quantile == 0 else supported.max()
                    np.testing.assert_array_equal(predictions, np.full(3, expected))


if __name__ == "__main__":
    unittest.main()
