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

    def test_epanechnikov_chord_weights_and_support_boundary(self):
        # At bandwidth 2 the cosine scores 1, 0, -1 give raw weights 1, 1/2, 0.
        model = self.model(similarity_fn="cos_similarity", kernel="epanechnikov", bandwidth=2.0)
        expected = torch.tensor([[2 / 3, 1 / 3, 0.0], [0.25, 0.5, 0.25]])
        weights = model.compute_weights(self.queries)
        torch.testing.assert_close(weights, expected)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
        self.assertEqual(weights[0, 2].item(), 0.0)  # d == b has no positive weight.
        narrower = self.model(
            similarity_fn="cos_similarity", kernel="epanechnikov", bandwidth=1.0
        ).compute_weights(self.queries)
        torch.testing.assert_close(narrower, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    def test_epanechnikov_normalization_handles_tiny_and_large_nonzero_vectors(self):
        model = self.model(
            similarity_fn="cos_similarity", kernel="epanechnikov", bandwidth=2.0,
            encoded_rep=torch.tensor([[2e30, 0.0], [0.0, 1e-30], [-1e-30, 0.0]]),
        )
        weights = model.compute_weights(torch.tensor([[2e-30, 0.0], [0.0, 3e30]]))
        torch.testing.assert_close(
            weights, torch.tensor([[2 / 3, 1 / 3, 0.0], [0.25, 0.5, 0.25]])
        )

    def test_epanechnikov_weights_do_not_depend_on_temperature(self):
        for temperature in (-10.0, 0.0, 10.0, float("nan"), float("inf"), None):
            with self.subTest(temperature=temperature):
                weights = self.model(
                    similarity_fn="cos_similarity", temperature=temperature,
                    kernel="epanechnikov", bandwidth=2.0,
                ).compute_weights(self.queries)
                torch.testing.assert_close(
                    weights, torch.tensor([[2 / 3, 1 / 3, 0.0], [0.25, 0.5, 0.25]])
                )

    def test_epanechnikov_extreme_finite_bandwidths(self):
        for bandwidth, expected in (
            (1e-200, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])),
            (1e200, torch.full((2, 3), 1 / 3)),
        ):
            with self.subTest(bandwidth=bandwidth):
                weights = self.model(
                    similarity_fn="cos_similarity", kernel="epanechnikov", bandwidth=bandwidth,
                ).compute_weights(self.queries)
                torch.testing.assert_close(weights, expected)

    def test_epanechnikov_requires_cosine_and_positive_finite_scalar_bandwidth(self):
        for bandwidth in (
            None, 0, -1.0, float("nan"), float("inf"), float("-inf"),
            True, False, "2.0", [], [2.0], [1.0, 2.0], 1 + 2j,
        ):
            with self.subTest(bandwidth=bandwidth):
                with self.assertRaisesRegex(ValueError, "bandwidth"):
                    self.model(similarity_fn="cos_similarity", kernel="epanechnikov", bandwidth=bandwidth)
        for similarity_fn in ("dot_product", "euclidean"):
            with self.subTest(similarity_fn=similarity_fn):
                with self.assertRaisesRegex(ValueError, "cos_similarity"):
                    self.model(similarity_fn=similarity_fn, kernel="epanechnikov", bandwidth=1.0)
        with self.assertRaisesRegex(ValueError, "kernel"):
            self.model(kernel="unknown")

    def test_epanechnikov_rejects_invalid_representations(self):
        invalid_representations = (
            torch.tensor([[0.0, 0.0]]), torch.tensor([[float("nan"), 1.0]]),
            torch.tensor([[float("inf"), 1.0]]), torch.tensor([1.0, 2.0]),
            torch.empty((1, 0)), torch.tensor([[1, 2]]), torch.tensor([[1 + 1j, 0j]]),
        )
        for representation in invalid_representations:
            with self.subTest(representation=representation):
                model = self.model(similarity_fn="cos_similarity", kernel="epanechnikov", bandwidth=1.0)
                with self.assertRaisesRegex(ValueError, "representations"):
                    model.compute_weights(representation)
                model = self.model(
                    similarity_fn="cos_similarity", kernel="epanechnikov", bandwidth=1.0,
                    encoded_rep=representation,
                )
                with self.assertRaisesRegex(ValueError, "representations"):
                    model.compute_weights(self.queries)
        model = self.model(similarity_fn="cos_similarity", kernel="epanechnikov", bandwidth=1.0)
        with self.assertRaisesRegex(ValueError, "dimension"):
            model.compute_weights(torch.ones((1, 3)))

    def test_epanechnikov_mixed_neighborhoods_keep_sampling_and_zero_fallback(self):
        queries = torch.tensor([[3.0, 0.0], [-1.0, -1.0], [0.0, 2.0]])
        samples = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 0, 0]])
        for target in (torch.tensor([-3.0, 8.0]), torch.tensor([[-3.0], [8.0]])):
            with self.subTest(target_shape=tuple(target.shape)):
                model = self.model(
                    similarity_fn="cos_similarity", kernel="epanechnikov", bandwidth=1.5,
                    encoded_rep=torch.tensor([[1.0, 0.0], [0.0, 1.0]]), target=target,
                )
                torch.testing.assert_close(
                    model.compute_weights(queries),
                    torch.tensor([[0.9, 0.1], [0.0, 0.0], [0.1, 0.9]]),
                )
                with patch("dscp.models.local_cp.torch.multinomial", return_value=samples) as sample:
                    quantiles = model.approximate_quantile(queries, [0.375, 0.5, 0.75], 5)
                sample.assert_called_once()
                torch.testing.assert_close(
                    sample.call_args.args[0], torch.tensor([[0.9, 0.1], [1.0, 0.0], [0.1, 0.9]])
                )
                self.assertEqual(sample.call_args.kwargs["num_samples"], 5)
                self.assertIs(sample.call_args.kwargs["replacement"], True)
                # The 2.5 endpoint checks that sampled torch.quantile interpolation is retained.
                torch.testing.assert_close(
                    quantiles, torch.tensor([[-3.0, 0.0, 2.5], [-3.0, 0.0, 8.0], [8.0, 0.0, 8.0]])
                )

    def test_epanechnikov_all_empty_neighborhoods_keep_draw_shape_and_zero_quantiles(self):
        model = self.model(
            similarity_fn="cos_similarity", kernel="epanechnikov", bandwidth=0.5,
            encoded_rep=torch.tensor([[1.0, 0.0]]), target=torch.tensor([[9.0]]),
        )
        with patch("dscp.models.local_cp.torch.multinomial", wraps=torch.multinomial) as sample:
            quantiles = model.approximate_quantile(torch.tensor([[-1.0, 0.0], [0.0, 1.0]]), [0.05, 0.95], 1000)
        sample.assert_called_once()
        torch.testing.assert_close(sample.call_args.args[0], torch.ones((2, 1)))
        self.assertEqual(sample.call_args.kwargs["num_samples"], 1000)
        self.assertIs(sample.call_args.kwargs["replacement"], True)
        torch.testing.assert_close(quantiles, torch.zeros((2, 2)))

    def test_epanechnikov_fallback_preserves_rng_state_for_later_encoder_training(self):
        calibration = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        residuals = torch.tensor([-3.0, 8.0])
        for queries in (
            torch.tensor([[1.0, 0.0], [-1.0, -1.0], [0.0, 1.0]]),
            torch.tensor([[-1.0, -1.0], [-2.0, -2.0]]),
        ):
            with self.subTest(query_count=len(queries)):
                states = []
                for kernel in ("softmax", "epanechnikov"):
                    model = self.model(
                        similarity_fn="cos_similarity", kernel=kernel, bandwidth=1.5,
                        encoded_rep=calibration, target=residuals,
                    )
                    torch.manual_seed(12345)
                    model.approximate_quantile(queries, [0.05, 0.95], 1000)
                    states.append(torch.random.get_rng_state())
                self.assertTrue(torch.equal(*states))


if __name__ == "__main__":
    unittest.main()
