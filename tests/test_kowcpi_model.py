import contextlib
import io
import unittest
import warnings

import numpy as np

from baselines.kowcpi.model import (
    KOWCPIResidualIntervalEstimator,
    WeightedNadarayaWatson,
)


class KOWCPIModelTests(unittest.TestCase):
    def setUp(self):
        warning_context = warnings.catch_warnings()
        warning_context.__enter__()
        self.addCleanup(warning_context.__exit__, None, None, None)
        warnings.simplefilter("error", RuntimeWarning)

    @staticmethod
    def model(x, bandwidth=100.0, kernel="epanechnikov"):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        return WeightedNadarayaWatson(bandwidth=bandwidth, kernel=kernel).fit(
            x, np.arange(len(x), dtype=float),
        )

    def test_invalid_initial_barrier_regression_has_exact_two_point_weights(self):
        # At the former initial lambda=0.1, the positive point makes the
        # log barrier infinite and SciPy's numerical derivative warns.
        for kernel in ("epanechnikov", "gaussian"):
            with self.subTest(kernel=kernel):
                model = self.model([-40.0, 20.0], kernel=kernel)
                query = np.array([0.0])
                kernel_values = (
                    np.array([0.63, 0.72]) if kernel == "epanechnikov"
                    else np.exp(np.array([-0.08, -0.02]))
                )
                moments = np.array([-40.0, 20.0]) * kernel_values
                expected_p = np.array([moments[1], -moments[0]]) / (
                    moments[1] - moments[0]
                )
                np.testing.assert_allclose(
                    model.get_p_t_values(query), expected_p, rtol=1e-10, atol=1e-12,
                )
                np.testing.assert_allclose(
                    model.get_weights(query), [1.0 / 3.0, 2.0 / 3.0],
                    rtol=1e-10, atol=1e-12,
                )

    def test_mixed_sign_probabilities_satisfy_empirical_likelihood_constraints(self):
        for kernel in ("epanechnikov", "gaussian"):
            for direction in (-1.0, 1.0):
                with self.subTest(kernel=kernel, direction=direction):
                    model = self.model(
                        direction * np.array([-40.0, -3.0, 0.0, 2.0, 20.0, 250.0]),
                        kernel=kernel,
                    )
                    query = np.array([0.0])
                    p_values = model.get_p_t_values(query)
                    kernel_values = np.array([
                        model.kernel_function(query, row) for row in model.X_
                    ])
                    moments = model.X_[:, 0] * kernel_values
                    self.assertTrue(np.isfinite(p_values).all())
                    self.assertTrue((p_values > 0.0).all())
                    self.assertAlmostEqual(float(p_values.sum()), 1.0, places=10)
                    self.assertAlmostEqual(float(p_values @ moments), 0.0, places=9)
                    np.testing.assert_allclose(
                        p_values[moments == 0.0], 1.0 / len(p_values), atol=1e-12,
                    )

    def test_rescaling_inputs_and_bandwidth_preserves_probabilities_and_weights(self):
        x = np.array([-40.0, -3.0, 0.0, 2.0, 20.0])
        query = np.array([0.0])
        for kernel in ("epanechnikov", "gaussian"):
            reference = self.model(x, kernel=kernel)
            expected_p = reference.get_p_t_values(query)
            expected_weights = reference.get_weights(query)
            for scale in (1e-12, 1e12):
                with self.subTest(kernel=kernel, scale=scale):
                    scaled = self.model(x * scale, bandwidth=100.0 * scale, kernel=kernel)
                    np.testing.assert_allclose(
                        scaled.get_p_t_values(query), expected_p, rtol=1e-10, atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        scaled.get_weights(query), expected_weights, rtol=1e-10, atol=1e-12,
                    )

    def test_zero_moments_keep_uniform_probabilities_and_kernel_weights(self):
        for kernel in ("epanechnikov", "gaussian"):
            with self.subTest(kernel=kernel):
                model = self.model([[0.0, 0.0], [0.0, 0.5], [0.0, 1.5]],
                                   bandwidth=2.0, kernel=kernel)
                query = np.array([0.0, 0.0])
                kernel_values = np.array([
                    model.kernel_function(query, row) for row in model.X_
                ])
                np.testing.assert_allclose(model.get_p_t_values(query), np.full(3, 1.0 / 3.0))
                np.testing.assert_allclose(model.get_weights(query), kernel_values / kernel_values.sum())

    def test_absent_kernel_support_uses_uniform_weights(self):
        for kernel, x in (("epanechnikov", [-2.0, 2.0]),
                          ("gaussian", [-10000.0, 20000.0])):
            with self.subTest(kernel=kernel):
                model = self.model(x, bandwidth=1.0, kernel=kernel)
                query = np.array([0.0])
                np.testing.assert_allclose(model.get_p_t_values(query), [0.5, 0.5])
                np.testing.assert_allclose(model.get_weights(query), [0.5, 0.5])

    def test_one_sided_moments_fall_back_to_uncorrected_kernel_weights(self):
        for kernel in ("epanechnikov", "gaussian"):
            for x in ([1.0, 2.0], [-1.0, -2.0], [0.0, 1.0, 2.0], [0.0, -1.0, -2.0]):
                with self.subTest(kernel=kernel, x=x):
                    model = self.model(x, bandwidth=10.0, kernel=kernel)
                    query = np.array([0.0])
                    kernel_values = np.array([
                        model.kernel_function(query, row) for row in model.X_
                    ])
                    np.testing.assert_allclose(
                        model.get_p_t_values(query), np.full(len(x), 1.0 / len(x)),
                    )
                    np.testing.assert_allclose(
                        model.get_weights(query), kernel_values / kernel_values.sum(),
                    )

    def test_short_online_residual_intervals_are_finite_and_ordered(self):
        residuals = np.array([
            -40.0, 20.0, -10.0, 30.0, -20.0, 5.0, 15.0, -30.0,
            25.0, 10.0, -5.0, 35.0, -15.0, 0.0, 20.0, 40.0,
        ])
        for kernel in ("epanechnikov", "gaussian"):
            with self.subTest(kernel=kernel), contextlib.redirect_stdout(io.StringIO()):
                model = KOWCPIResidualIntervalEstimator(bandwidth=100.0, kernel=kernel)
                lower, upper = model.predict_residual_intervals(
                    residuals, calibration_size=12, test_size=4,
                    alpha=0.2, block_size=2, history_window=12,
                )
                self.assertEqual(lower.shape, (4,))
                self.assertEqual(upper.shape, (4,))
                self.assertTrue(np.isfinite(lower).all())
                self.assertTrue(np.isfinite(upper).all())
                self.assertTrue((lower <= upper).all())


if __name__ == "__main__":
    unittest.main()
