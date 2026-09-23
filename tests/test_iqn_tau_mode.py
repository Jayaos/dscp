import unittest
import warnings
from unittest.mock import patch

import torch

from dscp.loss import (
    compute_loss_iqn_rnn,
    compute_loss_iqn_transformer,
    resolve_iqn_training_quantiles,
    resolve_iqn_validation_quantiles,
    warn_iqn_training_interval_mismatch,
)
from dscp.models.iqn_rnn import IQNRNN
from dscp.models.iqn_transformer import IQNTransformer


class IQNTrainingTauModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(97)
        self.inputs = torch.randn(3, 4, 3)
        self.targets = torch.randn(3, 1)
        self.levels = torch.tensor([0.05, 0.95])

    @staticmethod
    def _model(architecture, prediction_head):
        common = dict(
            dim_feature=3,
            dim_model=4,
            num_layers=1,
            iqn_hidden_dim=5,
            n_cos_embedding=6,
            dropout=0.0,
            prediction_head=prediction_head,
            interval_mode="direct",
        )
        if architecture == "transformer":
            return IQNTransformer(num_head=2, dim_ff=8, **common)
        return IQNRNN(rnn_type="gru", **common)

    def test_training_mode_defaults_to_sampled_without_validating_endpoints(self):
        for config in ({}, {"validation_loss": "target_quantiles"}):
            with self.subTest(config=config):
                self.assertEqual(
                    resolve_iqn_training_quantiles(config, None),
                    ("sampled_quantiles", None),
                )

    def test_target_mode_normalizes_and_deduplicates_interval_endpoints(self):
        mode, levels = resolve_iqn_training_quantiles(
            {"tau_mode": "  TARGET_QUANTILES  "},
            [[0.95, 0.05], [0.1, 0.9], [0.05, 0.95]],
        )
        self.assertEqual(mode, "target_quantiles")
        self.assertEqual(levels, [0.05, 0.1, 0.9, 0.95])

        self.assertEqual(
            resolve_iqn_training_quantiles(
                {"tau_mode": " SAMPLED_QUANTILES "},
                [[0.5, 0.5]],
            ),
            ("sampled_quantiles", None),
        )

    def test_invalid_modes_and_target_endpoints_are_rejected(self):
        for invalid_mode in ("", "uniform", None):
            with self.subTest(tau_mode=invalid_mode):
                with self.assertRaisesRegex(ValueError, "training.tau_mode"):
                    resolve_iqn_training_quantiles(
                        {"tau_mode": invalid_mode},
                        [[0.05, 0.95]],
                    )

        invalid_targets = (
            None,
            [],
            [[0.05]],
            [[0.05, 0.5, 0.95]],
            [[0.5, 0.5]],
            [[0.0, 0.95]],
            [[0.05, 1.0]],
            [[float("nan"), 0.95]],
            [[True, 0.95]],
            [["0.05", 0.95]],
        )
        for target_quantiles in invalid_targets:
            with self.subTest(target_quantiles=target_quantiles):
                with self.assertRaisesRegex(
                    ValueError,
                    "target_quantiles training",
                ):
                    resolve_iqn_training_quantiles(
                        {"tau_mode": "target_quantiles"},
                        target_quantiles,
                    )

    def test_training_and_validation_modes_are_independent(self):
        training = resolve_iqn_training_quantiles(
            {
                "tau_mode": "target_quantiles",
                "validation_loss": "invalid_validation_mode",
            },
            [[0.05, 0.95]],
        )
        self.assertEqual(training, ("target_quantiles", [0.05, 0.95]))

        validation = resolve_iqn_validation_quantiles(
            {
                "tau_mode": "invalid_training_mode",
                "validation_loss": "sampled_quantiles",
            },
            None,
        )
        self.assertEqual(validation, ("sampled_quantiles", None))

        with self.assertRaisesRegex(ValueError, "training.validation_loss"):
            resolve_iqn_validation_quantiles(
                {"validation_loss": "invalid"},
                [[0.05, 0.95]],
            )
        with self.assertRaisesRegex(ValueError, "target_quantiles validation"):
            resolve_iqn_validation_quantiles(
                {"validation_loss": "target_quantiles"},
                [[0.5, 0.5]],
            )

    def test_warning_only_for_fixed_cosine_training_with_sampled_inference(self):
        warning_cases = (
            ("target_quantiles", {}),
            (
                " TARGET_QUANTILES ",
                {
                    "prediction_head": " COSINE_EMBEDDING ",
                    "interval_mode": " SAMPLING ",
                },
            ),
        )
        for tau_mode, model_config in warning_cases:
            with self.subTest(tau_mode=tau_mode, model_config=model_config):
                with self.assertWarnsRegex(UserWarning, "interval_mode='direct'"):
                    warn_iqn_training_interval_mismatch(tau_mode, model_config)

        no_warning_cases = (
            ("sampled_quantiles", {}),
            (
                "target_quantiles",
                {"prediction_head": "cosine_embedding", "interval_mode": "direct"},
            ),
            (
                "target_quantiles",
                {
                    "prediction_head": "partially_monotonic",
                    "interval_mode": "sampling",
                },
            ),
        )
        for tau_mode, model_config in no_warning_cases:
            with self.subTest(tau_mode=tau_mode, model_config=model_config):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    warn_iqn_training_interval_mismatch(tau_mode, model_config)
                self.assertEqual(caught, [])

    def test_fixed_target_backprop_uses_no_tau_sampling_for_all_models(self):
        for architecture in ("rnn", "transformer"):
            for prediction_head in (
                "cosine_embedding",
                "partially_monotonic",
            ):
                with self.subTest(
                    architecture=architecture,
                    prediction_head=prediction_head,
                ):
                    model = self._model(architecture, prediction_head).train()
                    loss_fn = (
                        compute_loss_iqn_transformer
                        if architecture == "transformer"
                        else compute_loss_iqn_rnn
                    )
                    with patch.object(
                        model.iqn,
                        "sample_taus",
                        wraps=model.iqn.sample_taus,
                    ) as sample_taus:
                        loss = loss_fn(
                            model,
                            self.inputs,
                            self.targets,
                            num_taus=37,
                            taus=self.levels,
                        )

                    self.assertEqual(sample_taus.call_count, 0)
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                    self.assertIsNotNone(model.input_linear.weight.grad)
                    self.assertTrue(
                        torch.isfinite(model.input_linear.weight.grad).all()
                    )
                    for name, parameter in model.iqn.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(
                            torch.isfinite(parameter.grad).all(),
                            name,
                        )


if __name__ == "__main__":
    unittest.main()
