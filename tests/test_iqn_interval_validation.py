import unittest
from unittest.mock import patch

import torch

from dscp.loss import (
    compute_iqn_interval_validation_loss,
    compute_loss_iqn_rnn,
    compute_loss_iqn_transformer,
)
from dscp.models.iqn_rnn import IQNRNN
from dscp.models.iqn_transformer import IQNTransformer


class _FixedIntervalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor([[0.0, 2.0], [4.0, 1.0]]))
        self.calls = []

    def predict_quantiles(self, src, quantiles, current_feature=None):
        self.calls.append((src, quantiles, current_feature, torch.is_grad_enabled()))
        return self.values * 1.0


class IQNIntervalValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(71)
        self.inputs = torch.randn(3, 4, 3)
        self.targets = torch.randn(3, 1)
        self.current_feature = torch.randn(3, 1, 2)
        self.quantiles = torch.tensor([0.05, 0.95])

    @staticmethod
    def _model(architecture, interval_mode, prediction_head="cosine_embedding"):
        common = dict(
            dim_feature=3,
            dim_model=6,
            num_layers=1,
            current_feature_dim=2,
            iqn_hidden_dim=5,
            iqn_num_layers=2,
            n_cos_embedding=7,
            dropout=0.0,
            prediction_head=prediction_head,
            interval_mode=interval_mode,
            sampling_num=11,
        )
        if architecture == "transformer":
            model = IQNTransformer(num_head=2, dim_ff=12, **common)
        else:
            model = IQNRNN(rnn_type="gru", **common)
        return model.eval()

    def test_exact_pinball_mean_and_no_gradients(self):
        model = _FixedIntervalModel()
        inputs = torch.zeros(2, 3, 1)
        current_feature = torch.zeros(2, 1, 2)
        loss = compute_iqn_interval_validation_loss(
            model,
            inputs,
            torch.tensor([[1.0], [3.0]]),
            [0.25, 0.75],
            current_feature,
        )
        # Errors [[1, -1], [-1, 2]] yield [.25, .25, .75, 1.5].
        torch.testing.assert_close(loss, torch.tensor(0.6875))
        self.assertFalse(loss.requires_grad)
        self.assertIsNone(model.values.grad)
        self.assertFalse(model.calls[0][3])
        self.assertIs(model.calls[0][2], current_feature)
        torch.testing.assert_close(
            model.calls[0][1], torch.tensor([[0.25, 0.75], [0.25, 0.75]])
        )

    def test_invalid_shapes_levels_and_seed_are_rejected(self):
        model = _FixedIntervalModel()
        inputs = torch.zeros(2, 3, 1)
        target = torch.zeros(2, 1)
        for invalid_target in (torch.zeros(2), torch.zeros(2, 2), torch.zeros(1, 1)):
            with self.subTest(target_shape=invalid_target.shape):
                with self.assertRaisesRegex(ValueError, "target shape"):
                    compute_iqn_interval_validation_loss(model, inputs, invalid_target, [0.1, 0.9])
        for levels in ([], [float("nan")], [-0.1, 0.9], [[0.1], [0.5], [0.9]]):
            with self.subTest(levels=levels):
                with self.assertRaises(ValueError):
                    compute_iqn_interval_validation_loss(model, inputs, target, levels)
        for seed in (True, 1.5, "7"):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(TypeError, "sampling_seed"):
                    compute_iqn_interval_validation_loss(
                        model, inputs, target, [0.1, 0.9], sampling_seed=seed
                    )

    def test_direct_mode_matches_raw_target_loss_for_both_encoders(self):
        for architecture in ("rnn", "transformer"):
            with self.subTest(architecture=architecture):
                model = self._model(architecture, "direct")
                raw_loss = (
                    compute_loss_iqn_transformer
                    if architecture == "transformer"
                    else compute_loss_iqn_rnn
                )
                expected = raw_loss(
                    model,
                    self.inputs,
                    self.targets,
                    num_taus=37,
                    current_feature=self.current_feature,
                    taus=self.quantiles,
                )
                actual = compute_iqn_interval_validation_loss(
                    model,
                    self.inputs,
                    self.targets,
                    self.quantiles,
                    self.current_feature,
                )
                torch.testing.assert_close(actual, expected)
                self.assertFalse(actual.requires_grad)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_seeded_validation_repeats_and_restores_rng_for_all_heads(self):
        for architecture in ("rnn", "transformer"):
            for prediction_head in ("cosine_embedding", "partially_monotonic"):
                for interval_mode in ("direct", "sampling"):
                    with self.subTest(
                        architecture=architecture,
                        prediction_head=prediction_head,
                        interval_mode=interval_mode,
                    ):
                        model = self._model(architecture, interval_mode, prediction_head)
                        rng_before = torch.get_rng_state().clone()
                        first = compute_iqn_interval_validation_loss(
                            model,
                            self.inputs,
                            self.targets,
                            self.quantiles,
                            self.current_feature,
                            sampling_seed=123,
                        )
                        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
                        torch.rand(19)
                        rng_later = torch.get_rng_state().clone()
                        second = compute_iqn_interval_validation_loss(
                            model,
                            self.inputs,
                            self.targets,
                            self.quantiles,
                            self.current_feature,
                            sampling_seed=123,
                        )
                        torch.testing.assert_close(first, second, rtol=0, atol=0)
                        self.assertTrue(torch.equal(rng_later, torch.get_rng_state()))

    def test_sampling_matches_deployed_endpoints_and_uses_configured_sample_count(self):
        for architecture in ("rnn", "transformer"):
            with self.subTest(architecture=architecture):
                model = self._model(architecture, "sampling")
                generator = torch.Generator(device="cpu").manual_seed(123)
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(generator.get_state())
                    predicted = model.predict_quantiles(
                        self.inputs, self.quantiles, current_feature=self.current_feature
                    )
                errors = self.targets - predicted
                expected = torch.maximum(
                    (self.quantiles - 1.0) * errors, self.quantiles * errors
                ).mean()
                with patch.object(model.iqn, "sample_taus", wraps=model.iqn.sample_taus) as sample:
                    actual = compute_iqn_interval_validation_loss(
                        model,
                        self.inputs,
                        self.targets,
                        self.quantiles,
                        self.current_feature,
                        sampling_seed=123,
                    )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(sample.call_count, 1)
                self.assertEqual(sample.call_args.kwargs["num_taus"], 11)
                self.assertFalse(actual.requires_grad)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_rng_is_restored_if_prediction_raises(self):
        model = _FixedIntervalModel()

        def fail(**kwargs):
            torch.rand(9)
            raise RuntimeError("prediction failure")

        before = torch.get_rng_state().clone()
        with patch.object(model, "predict_quantiles", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "prediction failure"):
                compute_iqn_interval_validation_loss(
                    model, torch.zeros(2, 3, 1), torch.zeros(2, 1), [0.1, 0.9], sampling_seed=1
                )
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_training_still_samples_full_range_and_computes_gradients(self):
        for architecture in ("rnn", "transformer"):
            for interval_mode in ("direct", "sampling"):
                with self.subTest(architecture=architecture, interval_mode=interval_mode):
                    model = self._model(architecture, interval_mode).train()
                    loss_fn = (
                        compute_loss_iqn_transformer
                        if architecture == "transformer"
                        else compute_loss_iqn_rnn
                    )
                    before = torch.get_rng_state().clone()
                    with patch.object(model.iqn, "sample_taus", wraps=model.iqn.sample_taus) as sample:
                        loss = loss_fn(
                            model,
                            self.inputs,
                            self.targets,
                            num_taus=7,
                            current_feature=self.current_feature,
                        )
                    self.assertEqual(sample.call_count, 1)
                    self.assertEqual(sample.call_args.kwargs["num_taus"], 7)
                    self.assertFalse(torch.equal(before, torch.get_rng_state()))
                    loss.backward()
                    self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_sampling_repeats_and_restores_cpu_and_all_cuda_rng_states(self):
        model = self._model("rnn", "sampling").cuda()
        inputs = self.inputs.cuda()
        target = self.targets.cuda()
        current_feature = self.current_feature.cuda()
        cpu_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all()
        first = compute_iqn_interval_validation_loss(
            model, inputs, target, self.quantiles, current_feature, sampling_seed=123
        )
        second = compute_iqn_interval_validation_loss(
            model, inputs, target, self.quantiles, current_feature, sampling_seed=123
        )
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.equal(cpu_before, torch.get_rng_state()))
        for before, after in zip(cuda_before, torch.cuda.get_rng_state_all()):
            self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
