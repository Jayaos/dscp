import unittest

import numpy as np
import torch

from utils.reporting import compute_winkler_score, construct_interval_endpoints


class WinklerScoreTests(unittest.TestCase):
    def setUp(self):
        self.lower = torch.tensor([-1.0, -1.0, -1.0])
        self.upper = torch.tensor([1.0, 1.0, 1.0])
        self.y = torch.tensor([-2.0, 0.0, 2.0])
        self.preds = torch.zeros(3)

    def test_equal_tailed_pair_matches_classic_winkler_score(self):
        actual = compute_winkler_score(
            self.upper, self.lower, self.y, self.preds, (0.05, 0.95)
        )

        np.testing.assert_allclose(actual, [22.0, 2.0, 22.0])

    def test_asymmetric_pair_uses_separate_tail_penalties(self):
        actual = compute_winkler_score(
            self.upper, self.lower, self.y, self.preds, (0.025, 0.925)
        )

        np.testing.assert_allclose(
            actual,
            [42.0, 2.0, 2.0 + 1.0 / (1.0 - 0.925)],
            rtol=1e-6,
        )

    def test_pair_order_does_not_change_score(self):
        ascending = compute_winkler_score(
            self.upper, self.lower, self.y, self.preds, (0.025, 0.925)
        )
        descending = compute_winkler_score(
            self.upper, self.lower, self.y, self.preds, (0.925, 0.025)
        )

        np.testing.assert_allclose(ascending, descending)

    def test_scalar_alpha_retains_equal_tailed_behavior(self):
        pair_score = compute_winkler_score(
            self.upper, self.lower, self.y, self.preds, (0.05, 0.95)
        )
        alpha_score = compute_winkler_score(
            self.upper, self.lower, self.y, self.preds, 0.1
        )

        np.testing.assert_allclose(pair_score, alpha_score)

    def test_zero_dimensional_alpha_retains_equal_tailed_behavior(self):
        expected = compute_winkler_score(
            self.upper, self.lower, self.y, self.preds, 0.1
        )

        for alpha in (np.array(0.1), torch.tensor(0.1)):
            with self.subTest(alpha_type=type(alpha)):
                actual = compute_winkler_score(
                    self.upper, self.lower, self.y, self.preds, alpha
                )
                np.testing.assert_allclose(actual, expected)

    def test_numpy_interval_endpoints_are_supported(self):
        actual = compute_winkler_score(
            self.upper.numpy(),
            self.lower.numpy(),
            self.y,
            self.preds,
            (0.025, 0.925),
        )

        np.testing.assert_allclose(
            actual,
            [42.0, 2.0, 2.0 + 1.0 / (1.0 - 0.925)],
            rtol=1e-6,
        )

    def test_normalized_asymmetric_score_uses_final_endpoints(self):
        lower = torch.tensor([-1.0, -1.0, -1.0])
        upper = torch.tensor([2.0, 2.0, 2.0])
        y = torch.tensor([106.0, 110.0, 117.0])
        preds = torch.tensor([100.0, 100.0, 100.0])

        actual = compute_winkler_score(
            upper,
            lower,
            y,
            preds,
            (0.2, 0.9),
            normalized_params=(10.0, 2.0),
        )

        np.testing.assert_allclose(actual, [16.0, 6.0, 36.0])

    def test_invalid_quantile_levels_are_rejected(self):
        invalid_pairs = [
            (0.0, 0.9),
            (0.1, 1.0),
            (0.5, 0.5),
            (0.1, 0.5, 0.9),
            (np.nan, 0.9),
        ]

        for confidence_pair in invalid_pairs:
            with self.subTest(confidence_pair=confidence_pair):
                with self.assertRaises(ValueError):
                    compute_winkler_score(
                        self.upper,
                        self.lower,
                        self.y,
                        self.preds,
                        confidence_pair,
                    )

    def test_invalid_scalar_alpha_is_rejected(self):
        for alpha in (0.0, 1.0, np.nan):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    compute_winkler_score(
                        self.upper, self.lower, self.y, self.preds, alpha
                    )


class IntervalEndpointTests(unittest.TestCase):
    def test_unnormalized_residual_quantiles_are_added_to_predictions(self):
        upper, lower = construct_interval_endpoints(
            torch.tensor([2.0, 3.0]),
            torch.tensor([-1.0, -2.0]),
            torch.tensor([[100.0], [200.0]]),
        )

        np.testing.assert_allclose(upper, [102.0, 203.0])
        np.testing.assert_allclose(lower, [99.0, 198.0])

    def test_normalized_residual_quantiles_are_denormalized_first(self):
        upper, lower = construct_interval_endpoints(
            torch.tensor([2.0, 3.0]),
            torch.tensor([-1.0, -2.0]),
            torch.tensor([100.0, 200.0]),
            normalized_params=(10.0, 2.0),
        )

        np.testing.assert_allclose(upper, [114.0, 216.0])
        np.testing.assert_allclose(lower, [108.0, 206.0])


if __name__ == "__main__":
    unittest.main()
