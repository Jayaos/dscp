import unittest

import numpy as np

from baselines.split_cp.model import (
    SplitCPResidualIntervalEstimator,
    conformal_quantile,
)


class SplitCPQuantileTests(unittest.TestCase):
    def test_corrected_order_statistic_does_not_interpolate(self):
        scores = np.array([100.0, 1.0, 4.0, 2.0, 3.0])
        original = scores.copy()
        # ceil((5 + 1) * 0.8) = 5: the fifth score is 100, not an
        # interpolated empirical 80th percentile between 4 and 100.
        self.assertEqual(conformal_quantile(scores, alpha=0.2), 100.0)
        self.assertEqual(conformal_quantile(scores, alpha=0.5), 3.0)
        np.testing.assert_array_equal(scores, original)

    def test_integer_rank_is_not_rounded_up_by_floating_point_error(self):
        # Binary arithmetic gives 25 * (1 - .44) = 14.000000000000002.
        self.assertEqual(conformal_quantile(np.arange(1.0, 25.0), 0.44), 14.0)
        self.assertEqual(conformal_quantile(np.arange(1.0, 20.0), 0.1), 18.0)

    def test_infinity_augmented_rank_handles_small_calibration_sets(self):
        self.assertEqual(conformal_quantile([3.0], 0.5), 3.0)
        self.assertTrue(np.isposinf(conformal_quantile([3.0], 0.1)))
        self.assertTrue(np.isposinf(conformal_quantile([1.0, 2.0, 3.0], 0.2)))
        self.assertEqual(conformal_quantile([1.0, 2.0, 3.0], 0.25), 3.0)

    def test_ties_zero_scores_and_unsorted_scores(self):
        self.assertEqual(conformal_quantile([2.0, 0.0, 2.0, 1.0], 0.4), 2.0)
        self.assertEqual(conformal_quantile([0.0] * 19, 0.05), 0.0)
        self.assertEqual(conformal_quantile([[1.0], [2.0], [3.0]], 0.5), 2.0)

    def test_invalid_scores_and_alpha_are_rejected(self):
        for scores in ([], [np.nan], [np.inf], [-1.0], [[1.0, 2.0]], 1.0):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                conformal_quantile(scores, 0.1)
        for alpha in (0.0, 1.0, -0.1, 1.1, np.nan, np.inf):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                conformal_quantile([1.0, 2.0], alpha)


class SplitCPModelTests(unittest.TestCase):
    def test_absolute_residuals_produce_response_scale_symmetric_intervals(self):
        model = SplitCPResidualIntervalEstimator().fit([-1.0, 2.0, -4.0, 3.0, 100.0])
        self.assertEqual(model.quantile(0.5), 3.0)
        predictions = np.array([1000.0, -20.0, 0.0])
        lower, upper = model.predict_interval(predictions, alpha=0.5)
        np.testing.assert_array_equal(lower, [997.0, -23.0, -3.0])
        np.testing.assert_array_equal(upper, [1003.0, -17.0, 3.0])

    def test_vector_and_column_inputs_produce_scalar_output_without_broadcasting(self):
        residuals = np.arange(-9.0, 10.0)
        predictions = np.arange(5.0)
        expected = SplitCPResidualIntervalEstimator().fit(residuals).predict_interval(predictions, 0.1)
        for calibration in (residuals.tolist(), residuals[:, None]):
            model = SplitCPResidualIntervalEstimator().fit(calibration)
            for forecasts in (predictions.tolist(), predictions[:, None]):
                with self.subTest(calibration=np.shape(calibration), forecasts=np.shape(forecasts)):
                    actual = model.predict_interval(forecasts, 0.1)
                    for endpoint, reference in zip(actual, expected):
                        self.assertEqual(endpoint.shape, (5,))
                        np.testing.assert_array_equal(endpoint, reference)

    def test_calibration_is_frozen_and_does_not_alias_the_callers_data(self):
        residuals = np.arange(1.0, 20.0)
        model = SplitCPResidualIntervalEstimator().fit(residuals)
        residuals[:] = 10000.0
        first = model.predict_interval([100.0, 200.0], 0.1)
        self.assertEqual(model.quantile(0.1), 18.0)
        for _ in range(3):
            model.predict_interval([-999999.0], 0.2)
            actual = model.predict_interval([100.0, 200.0], 0.1)
            for endpoint, expected in zip(actual, first):
                np.testing.assert_array_equal(endpoint, expected)

    def test_higher_confidence_gives_nested_intervals(self):
        model = SplitCPResidualIntervalEstimator().fit(np.arange(1.0, 20.0))
        previous_lower, previous_upper = model.predict_interval([0.0, 100.0], 0.4)
        for alpha in (0.2, 0.1, 0.05, 0.01):
            lower, upper = model.predict_interval([0.0, 100.0], alpha)
            self.assertTrue(np.all(lower <= previous_lower))
            self.assertTrue(np.all(upper >= previous_upper))
            previous_lower, previous_upper = lower, upper

    def test_unattainable_confidence_gives_unbounded_intervals(self):
        model = SplitCPResidualIntervalEstimator().fit([0.0])
        lower, upper = model.predict_interval([0.0, 100.0], 0.1)
        self.assertTrue(np.isneginf(lower).all())
        self.assertTrue(np.isposinf(upper).all())

    def test_unfitted_model_and_invalid_scalar_sequences_are_rejected(self):
        model = SplitCPResidualIntervalEstimator()
        with self.assertRaises(RuntimeError):
            model.quantile(0.1)
        with self.assertRaises(RuntimeError):
            model.predict_interval([0.0], 0.1)
        for values in ([], 1.0, [[1.0, 2.0]], np.zeros((2, 2)), np.zeros((2, 1, 1)), [np.nan], [np.inf]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                model.fit(values)
            fitted = SplitCPResidualIntervalEstimator().fit([1.0])
            with self.subTest(predictions=values), self.assertRaises(ValueError):
                fitted.predict_interval(values, 0.1)


if __name__ == "__main__":
    unittest.main()
