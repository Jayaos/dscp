import unittest

import numpy as np

from utils.plotting import _resolve_logged_interval_endpoints


class LoggedIntervalEndpointTests(unittest.TestCase):
    def test_new_schema_uses_saved_response_scale_endpoints_directly(self):
        result = {
            "lower_interval": [90.0],
            "upper_interval": [115.0],
            "lower_residual_quantile": [-10.0],
            "upper_residual_quantile": [15.0],
            "target_predictions": [100.0],
        }

        lower, upper = _resolve_logged_interval_endpoints(result)

        np.testing.assert_allclose(lower, [90.0])
        np.testing.assert_allclose(upper, [115.0])

    def test_new_normalized_schema_does_not_transform_endpoints_twice(self):
        result = {
            "lower_interval": [108.0],
            "upper_interval": [114.0],
            "lower_residual_quantile": [-1.0],
            "upper_residual_quantile": [2.0],
            "target_predictions": [100.0],
            "train_residuals_mu": 10.0,
            "train_residuals_std": 2.0,
        }

        lower, upper = _resolve_logged_interval_endpoints(result)

        np.testing.assert_allclose(lower, [108.0])
        np.testing.assert_allclose(upper, [114.0])

    def test_legacy_unnormalized_schema_adds_prediction(self):
        result = {
            "lower_interval": [-10.0],
            "upper_interval": [15.0],
            "target_predictions": [100.0],
        }

        lower, upper = _resolve_logged_interval_endpoints(result)

        np.testing.assert_allclose(lower, [90.0])
        np.testing.assert_allclose(upper, [115.0])

    def test_legacy_normalized_schema_denormalizes_and_adds_prediction(self):
        result = {
            "lower_interval": [-1.0],
            "upper_interval": [2.0],
            "target_predictions": [100.0],
            "train_residuals_mu": 10.0,
            "train_residuals_std": 2.0,
        }

        lower, upper = _resolve_logged_interval_endpoints(result)

        np.testing.assert_allclose(lower, [108.0])
        np.testing.assert_allclose(upper, [114.0])


if __name__ == "__main__":
    unittest.main()
