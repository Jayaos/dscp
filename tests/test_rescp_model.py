import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.rescp.model import ResCPResidualIntervalEstimator


class ResCPModelTests(unittest.TestCase):
    @staticmethod
    def model(**kwargs):
        settings = dict(reservoir_size=4, connectivity=0.5, calibration_size=8,
                        sampling_num=32, beta_bins=8, seed=17)
        settings.update(kwargs)
        return ResCPResidualIntervalEstimator(**settings)

    def test_upstream_recurrence_and_pre_observation_pairing(self):
        model = self.model(reservoir_size=2, leak_rate=0.3).fit([0.0], normalize=False)
        recurrent = np.array([[0.6, -0.2], [0.1, 0.4]], dtype=np.float32)
        inputs = np.array([0.8, -0.4], dtype=np.float32)
        model._internal_weights = torch.tensor(recurrent)
        model._input_weights = torch.tensor(inputs)
        expected = np.zeros(2, dtype=np.float32)
        for residual in (1.0, -0.2, 2.0):
            before = expected.copy()
            model.observe(residual)
            expected = 0.7 * expected + np.tanh(recurrent @ expected + inputs * residual)
            np.testing.assert_allclose(model._state.numpy(), expected, atol=1e-7)
            states, residuals = model._ordered_memory()
            normalized = before / np.linalg.norm(before) if np.linalg.norm(before) else before
            np.testing.assert_allclose(states[-1].numpy(), normalized, atol=1e-7)
            self.assertEqual(float(residuals[-1]), residual)
        self.assertFalse(np.isclose(np.linalg.norm(expected), 1.0))

    def test_similarity_matches_upstream_cosine_times_linear_ramp(self):
        model = self.model().fit([0.2, -1.0, 0.8, 0.4], normalize=False)
        states, _ = model._ordered_memory()
        query = model._state.numpy() / np.linalg.norm(model._state.numpy())
        similarities = states.numpy() @ query
        probabilities = np.exp((similarities.astype(float) - similarities.max()) / model.temperature)
        probabilities *= np.arange(model.memory_size)
        probabilities /= probabilities.sum()
        np.testing.assert_allclose(model._sampling_weights().numpy(), probabilities, rtol=1e-6, atol=1e-9)
        self.assertEqual(float(model._sampling_weights()[0]), 0.0)

    def test_quantiles_and_shortest_interval_match_independent_numpy_calculation(self):
        model = self.model(beta_bins=100, sampling_num=128).fit([-4.0, -1.0, 0.0, 2.0, 8.0])
        states, residuals = model._ordered_memory()
        # Reproduce the seeded draw without advancing the estimator's RNG.
        sampler = torch.Generator().set_state(model._sampling_generator.get_state())
        indices = torch.multinomial(model._sampling_weights(), 128, replacement=True, generator=sampler)
        samples = residuals[indices].numpy()
        beta_grid = np.linspace(0.001, 0.199, 100)
        lower = np.quantile(samples, beta_grid, method="linear")
        upper = np.quantile(samples, 0.8 + beta_grid, method="linear")
        selected = np.argmin(upper - lower)
        interval = model.predict_interval((0.1, 0.9))
        np.testing.assert_allclose(interval, (lower[selected], upper[selected], beta_grid[selected]), atol=1e-12)
        # Quantile estimation does not consume the current observation.
        np.testing.assert_array_equal(model._ordered_memory()[0], states)

    def test_fixed_tail_preserves_requested_asymmetry(self):
        model = self.model(use_beta_search=False, sampling_num=128).fit([-3.0, -1.0, 0.0, 4.0, 8.0])
        sampler = torch.Generator().set_state(model._sampling_generator.get_state())
        indices = torch.multinomial(model._sampling_weights(), 128, replacement=True, generator=sampler)
        samples = model._ordered_memory()[1][indices].numpy()
        low, high, beta = model.predict_interval((0.02, 0.82))
        np.testing.assert_allclose((low, high), np.quantile(samples, [0.02, 0.82]))
        self.assertEqual(beta, 0.02)

    def test_exact_window_cap_and_fixed_default_sampling_budget(self):
        for initial in ([1.0], [1.0, 2.0, 3.0, 4.0]):
            model = self.model(calibration_size=3, sampling_num=None).fit(initial)
            history = list(initial)
            for residual in range(5, 15):
                model.observe(float(residual))
                history.append(float(residual))
                self.assertEqual(model.memory_size, min(3, len(history)))
                self.assertEqual(model.effective_sampling_num, 3)
                np.testing.assert_array_equal(model._ordered_memory()[1], history[-3:])

    def test_unbounded_memory_grows_without_changing_sample_budget(self):
        model = self.model(calibration_size=None, sampling_num=None).fit([1.0, 2.0])
        for residual in range(3, 15):
            model.observe(float(residual))
        self.assertEqual(model.memory_size, 14)
        self.assertEqual(model.effective_sampling_num, 2)
        np.testing.assert_array_equal(model._ordered_memory()[1], np.arange(1.0, 15.0))

    def test_wrapped_prediction_matches_ordered_memory_without_copying_states(self):
        model = self.model(calibration_size=5, sampling_num=64, use_beta_search=False)
        model.fit([-2.0, 0.5, 1.0, 3.0, -1.0, 8.0, 2.0], normalize=False)
        self.assertNotEqual(model._memory_start, 0)
        states, residuals = model._ordered_memory()
        similarity = (states @ model._normalized(model._state)).to(torch.float64)
        # Reference the former contiguous calculation with the same log-space
        # recency weighting, then replay its exact seeded sampling operation.
        similarity -= similarity[1:].max()
        log_decay = torch.log(torch.arange(model.memory_size, dtype=torch.float64))
        expected_weights = torch.softmax(similarity / model.temperature + log_decay, dim=0)
        sampler = torch.Generator().set_state(model._sampling_generator.get_state())
        indices = torch.multinomial(expected_weights, 64, replacement=True, generator=sampler)
        expected = np.quantile(residuals[indices].numpy(), [0.05, 0.95])
        with patch.object(model, "_ordered_memory", side_effect=AssertionError("State copy during prediction")):
            torch.testing.assert_close(model._sampling_weights(), expected_weights)
            actual = model.predict_interval((0.05, 0.95))
        np.testing.assert_allclose(actual[:2], expected, atol=1e-12)
        self.assertEqual(actual[2], 0.05)

    def test_scaler_is_frozen_and_residual_pool_keeps_raw_units(self):
        model = self.model().fit([10.0, 20.0, 30.0])
        self.assertEqual(model.input_mean, 20.0)
        self.assertAlmostEqual(model.input_std, np.std([10.0, 20.0, 30.0]))
        model.observe(10000.0)
        self.assertEqual(model.input_mean, 20.0)
        self.assertAlmostEqual(model.input_std, np.std([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(model._ordered_memory()[1], [10.0, 20.0, 30.0, 10000.0])

    def test_future_observation_does_not_affect_current_prediction(self):
        first = self.model().fit([-2.0, 0.5, 1.0])
        second = self.model().fit([-2.0, 0.5, 1.0])
        before = first._state.clone()
        initial = first.predict_interval((0.05, 0.95))
        self.assertEqual(initial, second.predict_interval((0.05, 0.95)))
        torch.testing.assert_close(first._state, before)
        self.assertEqual(first.memory_size, 3)
        first.observe(-1000.0)
        second.observe(1000.0)
        self.assertFalse(torch.equal(first._state, second._state))

    def test_chunked_observation_matches_prefix_replay(self):
        residuals = [0.5, -2.0, 1.0, 3.0, -1.0]
        batch = self.model(calibration_size=3).fit(residuals, normalize=False)
        online = self.model(calibration_size=3).fit(residuals[:1], normalize=False)
        for residual in residuals[1:]:
            online.observe(residual)
        torch.testing.assert_close(batch._state, online._state, rtol=0, atol=0)
        for actual, expected in zip(batch._ordered_memory(), online._ordered_memory()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(batch.predict_interval((0.05, 0.95)), online.predict_interval((0.05, 0.95)))

    def test_reproducible_seeds_and_refit_without_global_rng_side_effects(self):
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        model = self.model().fit([-2.0, -1.0, 1.0, 4.0])
        first = model.predict_interval((0.05, 0.95))
        second = self.model().fit([-2.0, -1.0, 1.0, 4.0]).predict_interval((0.05, 0.95))
        self.assertEqual(first, second)
        self.assertEqual(first, model.fit([-2.0, -1.0, 1.0, 4.0]).predict_interval((0.05, 0.95)))
        self.assertEqual(np.random.get_state()[0], numpy_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_singleton_constant_zero_and_tiny_reservoirs_are_finite(self):
        for residuals in ([0.0], [3.0], [0.0] * 6, [3.0] * 6):
            for decay in ("linear", "none", "exponential"):
                with self.subTest(residuals=residuals, decay=decay):
                    model = self.model(reservoir_size=1, connectivity=0.01, decay=decay).fit(residuals)
                    low, high, beta = model.predict_interval((0.05, 0.95))
                    self.assertEqual(low, residuals[0])
                    self.assertEqual(high, residuals[0])
                    self.assertTrue(np.isfinite(beta))
                    self.assertAlmostEqual(float(model._sampling_weights().sum()), 1.0)

    def test_extreme_temperature_does_not_lose_all_sampling_mass(self):
        model = self.model(temperature=1e-300).fit([0.0, -1.0, 1.0, 2.0])
        weights = model._sampling_weights()
        self.assertTrue(torch.isfinite(weights).all())
        self.assertEqual(float(weights.sum()), 1.0)
        self.assertTrue(np.isfinite(model.predict_interval((0.05, 0.95))).all())

    def test_zero_spectral_radius_is_supported(self):
        model = self.model(spectral_radius=0).fit([0.1, 1.0])
        self.assertEqual(int(torch.count_nonzero(model._internal_weights)), 0)

    def test_small_alpha_has_valid_beta_grid(self):
        model = self.model().fit([-1.0, 0.0, 1.0])
        low, high, beta = model.predict_interval((0.0001, 0.9999))
        self.assertTrue(np.isfinite([low, high, beta]).all())
        self.assertLessEqual(low, high)
        self.assertTrue(0 <= beta <= 0.0002)

    def test_invalid_configuration_and_inputs_are_rejected(self):
        invalid = [dict(reservoir_size=2.5), dict(reservoir_size=True), dict(reservoir_size=0),
                   dict(calibration_size=-1), dict(calibration_size=2.2), dict(sampling_num=0),
                   dict(beta_bins=2.5), dict(seed=2.5), dict(seed=-1), dict(seed=2 ** 32),
                   dict(temperature=0), dict(temperature=float("nan")), dict(connectivity=0),
                   dict(leak_rate=1.1), dict(spectral_radius=-1), dict(decay="inverse_age"),
                   dict(decay_rate=0), dict(use_beta_search="false"), dict(recurrence="paper")]
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.model(**settings)
        for residuals in ([], [np.nan], [np.inf], [[1.0, 2.0]], ["1.0"]):
            with self.subTest(residuals=residuals), self.assertRaises(ValueError):
                self.model().fit(residuals)
        model = self.model()
        with self.assertRaises(RuntimeError):
            model.predict_interval((0.05, 0.95))
        model.fit([1.0])
        for pair in ((0, 1), (0.9, 0.1), (-0.1, 0.9), (0.1,), (0.1, np.nan)):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                model.predict_interval(pair)
        with self.assertRaises(ValueError):
            model.observe(float("inf"))
        self.assertEqual(model.memory_size, 1)


if __name__ == "__main__":
    unittest.main()
