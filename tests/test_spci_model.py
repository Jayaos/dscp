"""Forest choice, configuration, and repeatability for SPCI."""

import unittest

import numpy as np
from omegaconf import OmegaConf
from sklearn_quantile import RandomForestQuantileRegressor, SampleRandomForestQuantileRegressor

from baselines.spci.model import build_quantile_forest


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
                self.assertIs(type(forest), expected_type)
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
