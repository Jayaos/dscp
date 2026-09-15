"""Deterministic checks of Local-CP calibration weights and residual sampling."""

import math
import unittest
from unittest.mock import patch

import torch

from dscp.models.local_cp import LocalConformalPrediction
from utils.utils import negative_squared_euclidean


class LocalCPWeightTests(unittest.TestCase):
    def setUp(self):
        # Nonunit vectors distinguish raw Euclidean distances from normalized ones.
        self.calibration = torch.tensor([[2.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        self.queries = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        self.residuals = torch.tensor([[-3.0], [2.0], [8.0]])

    def model(self, similarity_fn="euclidean", temperature=2.0, **overrides):
        options = dict(
            encoded_rep=self.calibration,
            target=self.residuals,
            similarity_fn=similarity_fn,
            temperature=temperature,
            device="cpu",
        )
        options.update(overrides)
        return LocalConformalPrediction(**options)

    @staticmethod
    def expected_euclidean_weights(temperature):
        # These distances are calculated by hand from the fixture coordinates.
        rows = []
        for squared_distances in ([0.0, 5.0, 9.0], [13.0, 4.0, 10.0]):
            unnormalized = [math.exp(-distance / temperature) for distance in squared_distances]
            total = sum(unnormalized)
            rows.append([value / total for value in unnormalized])
        return torch.tensor(rows)

    def test_negative_squared_euclidean_scores_preserve_vector_lengths(self):
        scores = negative_squared_euclidean(self.queries, self.calibration)
        torch.testing.assert_close(
            scores, torch.tensor([[0.0, -5.0, -9.0], [-13.0, -4.0, -10.0]])
        )
        # The kernel must be symmetric when query and calibration sets coincide.
        self_scores = negative_squared_euclidean(self.calibration, self.calibration)
        torch.testing.assert_close(self_scores, self_scores.T)
        torch.testing.assert_close(self_scores.diag(), torch.zeros(3))

    def test_euclidean_weights_match_squared_distance_divided_by_temperature(self):
        weights = self.model().compute_weights(self.queries)
        self.assertEqual(weights.shape, (2, 3))
        torch.testing.assert_close(weights, self.expected_euclidean_weights(2.0))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights >= 0).all())
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
        torch.testing.assert_close(weights.argmax(dim=1), torch.tensor([0, 1]))

    def test_smaller_temperature_increases_nearest_neighbor_weight(self):
        cold = self.model(temperature=0.5).compute_weights(self.queries)
        warm = self.model(temperature=5.0).compute_weights(self.queries)
        self.assertGreater(cold[0, 0].item(), warm[0, 0].item())
        self.assertGreater(cold[1, 1].item(), warm[1, 1].item())
        for temperature, actual in ((0.5, cold), (5.0, warm)):
            torch.testing.assert_close(actual, self.expected_euclidean_weights(temperature))

    def test_equal_distances_give_uniform_weights(self):
        calibration = torch.tensor([[2.0, 1.0], [2.0, 5.0]])
        queries = torch.tensor([[2.0, 3.0], [-1.0, 3.0]])
        for temperature in (0.25, 1, 10.0):
            with self.subTest(temperature=temperature):
                weights = self.model(
                    encoded_rep=calibration,
                    target=torch.tensor([1.0, 2.0]),
                    temperature=temperature,
                ).compute_weights(queries)
                torch.testing.assert_close(weights, torch.full((2, 2), 0.5))

    def test_very_small_temperature_keeps_finite_weights(self):
        # All uncentered scores for the second query overflow to -inf on division.
        weights = self.model(temperature=1e-38).compute_weights(self.queries)
        self.assertTrue(torch.isfinite(weights).all())
        torch.testing.assert_close(weights, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    def test_euclidean_temperature_requires_a_finite_positive_scalar(self):
        invalid_temperatures = (
            0, -1.0, float("nan"), float("inf"), float("-inf"),
            None, True, False, "2.0", [], [2.0], [1.0, 2.0],
        )
        for temperature in invalid_temperatures:
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    self.model(temperature=temperature)

    def test_existing_similarity_modes_keep_multiplier_temperature(self):
        dot_scores = torch.tensor([[4.0, 0.0, -2.0], [0.0, 3.0, 0.0]])
        cosine_scores = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        for similarity_fn, scores in (
            ("dot_product", dot_scores), ("cos_similarity", cosine_scores)
        ):
            # Zero/negative values were previously supported by these modes.
            for temperature in (2.0, 0.1, 0.0, -1.0):
                with self.subTest(similarity_fn=similarity_fn, temperature=temperature):
                    weights = self.model(
                        similarity_fn=similarity_fn, temperature=temperature
                    ).compute_weights(self.queries)
                    expected = torch.softmax(temperature * scores, dim=1)
                    torch.testing.assert_close(weights, expected)

    def test_quantiles_share_one_sample_draw_from_euclidean_weights(self):
        # Samples yield sorted residuals [-3, -3, 2, 8, 8] and [-3, 2, 2, 8, 8].
        sampled_indices = torch.tensor([[2, 0, 1, 0, 2], [1, 2, 1, 2, 0]])
        expected_quantiles = torch.tensor([[-3.0, 2.0], [2.0, 2.0], [8.0, 8.0]])
        for target in (self.residuals, self.residuals.squeeze(1)):
            with self.subTest(target_shape=tuple(target.shape)):
                model = self.model(target=target)
                with patch(
                    "dscp.models.local_cp.torch.multinomial",
                    return_value=sampled_indices,
                ) as sample:
                    quantiles = model.approximate_quantile(
                        self.queries, target_quantiles=[0.25, 0.5, 0.75], sampling_num=5
                    )
                sample.assert_called_once()
                torch.testing.assert_close(
                    sample.call_args.args[0], self.expected_euclidean_weights(2.0)
                )
                self.assertEqual(sample.call_args.kwargs["num_samples"], 5)
                self.assertIs(sample.call_args.kwargs["replacement"], True)
                torch.testing.assert_close(quantiles, expected_quantiles)
                self.assertTrue((quantiles[1:] >= quantiles[:-1]).all())


if __name__ == "__main__":
    unittest.main()
