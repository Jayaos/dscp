"""SPCI beta selection uses predicted widths independently at each time."""

import unittest

import numpy as np
from omegaconf import OmegaConf

from baselines.spci.intervals import build_interval_plan, select_intervals
from baselines.spci.model import build_quantile_forest


HALF_PAIR = (0.75, 0.25)
QUARTER_PAIR = (0.625, 0.375)
CURVE_QUANTILES = np.linspace(0.0, 1.0, 9)
CURVES = np.array([
    [0, 1, 2, 3, 4, 10, 20, 30, 40],
    [0, 5, 10, 11, 12, 13, 14, 19, 24],
    [0, 10, 20, 30, 36, 37, 38, 39, 40],
], dtype=float).T


def _config(**overrides):
    values = {"target_quantiles": [list(HALF_PAIR)], "optimize_beta": True, "beta_bins": 5}
    values.update(overrides)
    return OmegaConf.create(values)


def _predictions(quantiles):
    return np.column_stack([
        np.interp(quantiles, CURVE_QUANTILES, column) for column in CURVES.T
    ])


class SPCIIntervalTests(unittest.TestCase):
    def test_missing_opt_in_preserves_fixed_asymmetric_endpoints_and_pair_order(self):
        pairs = [(0.875, 0.125), (0.125, 0.625)]
        plan = build_interval_plan(OmegaConf.create({"target_quantiles": pairs}))
        self.assertFalse(plan.optimize_beta)
        np.testing.assert_array_equal(plan.quantiles, [0.125, 0.625, 0.875])
        values = np.array([[-4, -3], [2, 3], [8, 9]], dtype=float)
        intervals = select_intervals(values, plan)
        self.assertEqual(list(intervals), pairs)
        for pair, lower, upper, alpha in (
            (pairs[0], [-4, -3], [8, 9], 0.25),
            (pairs[1], [-4, -3], [2, 3], 0.5),
        ):
            interval = intervals[pair]
            np.testing.assert_array_equal(interval.lower, lower)
            np.testing.assert_array_equal(interval.upper, upper)
            np.testing.assert_array_equal(interval.beta, [min(pair)] * 2)
            np.testing.assert_array_equal(interval.upper_quantile, [max(pair)] * 2)
            self.assertEqual(interval.alpha, alpha)

    def test_beta_minimizes_predicted_width_separately_at_every_time(self):
        plan = build_interval_plan(_config())
        np.testing.assert_array_equal(plan.quantiles, CURVE_QUANTILES)
        interval = select_intervals(CURVES.copy(), plan)[HALF_PAIR]
        np.testing.assert_array_equal(interval.beta, [0.0, 0.25, 0.5])
        np.testing.assert_array_equal(interval.upper_quantile, [0.5, 0.75, 1.0])
        np.testing.assert_array_equal(interval.lower, [0, 10, 36])
        np.testing.assert_array_equal(interval.upper, [4, 14, 40])
        np.testing.assert_array_equal(interval.upper - interval.lower, [4, 4, 4])
        self.assertEqual(interval.alpha, 0.5)

    def test_multiple_coverages_share_unique_quantiles_but_select_independently(self):
        # The third pair has the same coverage as the first, with different tails.
        other_half = (0.625, 0.125)
        plan = build_interval_plan(_config(
            target_quantiles=[HALF_PAIR, QUARTER_PAIR, other_half]
        ))
        self.assertEqual(len(plan.quantiles), len(np.unique(plan.quantiles)))
        self.assertTrue((np.diff(plan.quantiles) > 0).all())
        self.assertEqual(plan.quantiles[0], 0.0)
        self.assertEqual(plan.quantiles[-1], 1.0)
        intervals = select_intervals(_predictions(plan.quantiles), plan)
        np.testing.assert_array_equal(intervals[HALF_PAIR].beta, [0, 0.25, 0.5])
        np.testing.assert_array_equal(intervals[QUARTER_PAIR].beta, [0, 0.375, 0.5625])
        np.testing.assert_array_equal(intervals[QUARTER_PAIR].upper - intervals[QUARTER_PAIR].lower,
                                      [2, 2, 2])
        np.testing.assert_array_equal(intervals[HALF_PAIR].lower, intervals[other_half].lower)
        np.testing.assert_array_equal(intervals[HALF_PAIR].upper, intervals[other_half].upper)

    def test_width_ties_choose_lowest_beta_deterministically(self):
        plan = build_interval_plan(_config())
        predictions = np.repeat(plan.quantiles[:, None], 4, axis=1)
        for _ in range(2):
            interval = select_intervals(predictions, plan)[HALF_PAIR]
            np.testing.assert_array_equal(interval.beta, np.zeros(4))
            np.testing.assert_array_equal(interval.upper_quantile, np.full(4, 0.5))

    def test_two_bin_search_includes_both_boundary_candidates(self):
        plan = build_interval_plan(_config(beta_bins=2))
        self.assertEqual(plan.beta_bins, 2)
        np.testing.assert_array_equal(plan.quantiles, [0, 0.5, 1])
        interval = select_intervals(np.array([[0, 0], [2, 8], [10, 10]]), plan)[HALF_PAIR]
        np.testing.assert_array_equal(interval.beta, [0, 0.5])
        np.testing.assert_array_equal(interval.upper - interval.lower, [2, 2])

    def test_invalid_flags_bin_counts_and_nominal_pairs_are_rejected(self):
        for value in (None, 0, 1, "true", "false"):
            with self.subTest(flag=value), self.assertRaisesRegex(ValueError, "optimize_beta"):
                build_interval_plan(_config(optimize_beta=value))
        for value in (None, True, False, 0, 1, -1, 2.5, "5"):
            with self.subTest(bins=value), self.assertRaisesRegex(ValueError, "beta_bins"):
                build_interval_plan(_config(beta_bins=value))
        for pairs in ([], [[0.5]], [[0.1, 0.5, 0.9]], [[0.5, 0.5]],
                      [[0, 0.9]], [[0.1, 1]], [[-0.1, 0.9]], [[0.1, 1.1]],
                      [[np.nan, 0.9]], [[0.1, np.inf]], [[True, 0.9]]):
            with self.subTest(pairs=pairs), self.assertRaises((TypeError, ValueError)):
                build_interval_plan(_config(target_quantiles=pairs))

    def test_invalid_prediction_shape_nonfinite_and_crossed_quantiles_fail(self):
        plan = build_interval_plan(_config())
        malformed = [
            CURVES[:, 0], CURVES.T, CURVES[:-1], CURVES[:, :, None],
        ]
        for values in malformed:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                select_intervals(values, plan)
        for value in (np.nan, np.inf, -np.inf):
            values = CURVES.copy()
            values[2, 1] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                select_intervals(values, plan)
        values = CURVES.copy()
        values[1, 0] = values[-1, 0] + 1
        with self.assertRaises(ValueError):
            select_intervals(values, plan)

    def test_boundary_candidates_work_with_exact_and_sampled_forests(self):
        config = OmegaConf.create({
            "seed": 7,
            "model": {
                "target_quantiles": [HALF_PAIR], "optimize_beta": True, "beta_bins": 5,
                "n_estimators": 10, "max_depth": 2, "criterion": "squared_error", "n_jobs": 1,
            },
        })
        plan = build_interval_plan(config.model)
        random = np.random.default_rng(12)
        features = random.normal(size=(60, 3)).astype(np.float32)
        targets = (features[:, 0] + random.normal(size=60)).astype(np.float32)
        # Factory sample-count controls the branch; both fits remain deliberately tiny.
        for factory_count in (60, 10_001):
            with self.subTest(factory_count=factory_count):
                forest = build_quantile_forest(config, factory_count, plan.quantiles)
                forest.fit(features, targets)
                interval = select_intervals(forest.predict(features[:6]), plan)[HALF_PAIR]
                self.assertTrue(np.isfinite(interval.lower).all())
                self.assertTrue(np.isfinite(interval.upper).all())
                self.assertTrue((interval.lower <= interval.upper).all())


if __name__ == "__main__":
    unittest.main()
