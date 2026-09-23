import contextlib
import importlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dscp.data import QuantileRegressionDataset


class IQNTauTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        sbatch_path = str(Path(__file__).resolve().parents[1] / "sbatch")
        with patch.object(sys, "path", [sbatch_path, *sys.path]):
            cls.tuning = importlib.import_module("sbatch_run_tuning.run_iqn_cp_tuning")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def _config(architecture="rnn", prediction_head="cosine_embedding", current=False):
        config = OmegaConf.create({
            "device": "cpu",
            "seed": 19,
            "data": {
                "normalize": False,
                "strided_features": "xr",
                "train_ratio": 0.5,
                "valid_ratio": 0.25,
            },
            "model": {
                "dim_model": 4,
                "num_heads": 2,
                "num_layers": 1,
                "dropout": 0.0,
                "use_current_feature": current,
                "prediction_head": prediction_head,
                "iqn_hidden_dim": 5,
                "iqn_num_layers": 2,
                "cos_emb_dim": 6,
                "monotonic_num_layers": 1,
                "monotonic_activation": "tanh",
                "interval_mode": "direct",
                "sampling_num": 11,
                "num_taus": 7,
                "target_quantiles": [[0.9, 0.1], [0.2, 0.8], [0.9, 0.1]],
                "window_size": 3,
                "prediction_step": 1,
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "epochs": 1,
                "early_stop": 1,
                "validation_loss": "target_quantiles",
            },
            "tuning": {"delta_threshold": 0.0},
        })
        if architecture == "rnn":
            config.model.rnn_type = "gru"
        return config

    @staticmethod
    def _dataset(size):
        rng = np.random.default_rng(23 + size)
        values = lambda *shape: rng.normal(size=shape).astype(np.float32)
        return QuantileRegressionDataset(
            values(size, 3, 2), values(size, 3), values(size, 3),
            values(size, 1, 2), values(size, 1), values(size, 1),
            values(size, 1),
        )

    @classmethod
    def _sequence(cls):
        return {
            "train_dataset": cls._dataset(5),
            "model_selection_valid_dataset": cls._dataset(3),
            "tuning_evaluation_dataset": cls._dataset(3),
        }

    def test_training_and_validation_modes_are_independent_for_both_encoders_and_heads(self):
        levels = [0.1, 0.2, 0.8, 0.9]
        for architecture in ("rnn", "transformer"):
            for head in ("cosine_embedding", "partially_monotonic"):
                for current in (False, True):
                    for tau_mode in (None, "sampled_quantiles", "target_quantiles"):
                        for validation_mode in ("sampled_quantiles", "target_quantiles"):
                            with self.subTest(
                                architecture=architecture, head=head, current=current,
                                tau_mode=tau_mode, validation_mode=validation_mode,
                            ):
                                config = self._config(architecture, head, current)
                                config.training.validation_loss = validation_mode
                                if tau_mode is not None:
                                    config.training.tau_mode = tau_mode
                                loss_name = f"compute_loss_iqn_{architecture}"
                                raw_loss = getattr(self.tuning, loss_name)
                                raw_build = self.tuning._build_model
                                raw_interval_loss = self.tuning.compute_iqn_interval_validation_loss
                                loss_calls, interval_calls, samples, built_models = [], [], [], []

                                def tracked_loss(model, x, target, num_taus, current_feature=None, *, taus=None):
                                    loss_calls.append((model.training, taus, current_feature))
                                    return raw_loss(
                                        model, x, target, num_taus,
                                        current_feature=current_feature, taus=taus,
                                    )

                                def tracked_interval(model, x, target, quantiles, current_feature=None, **kwargs):
                                    interval_calls.append((quantiles, current_feature))
                                    return raw_interval_loss(
                                        model, x, target, quantiles,
                                        current_feature=current_feature, **kwargs,
                                    )

                                with contextlib.ExitStack() as stack:
                                    stack.enter_context(torch.random.fork_rng(devices=[]))
                                    torch.manual_seed(31)

                                    def tracked_build(*args, **kwargs):
                                        built = raw_build(*args, **kwargs)
                                        model = built[0]
                                        built_models.append((model, {
                                            name: parameter.detach().clone()
                                            for name, parameter in model.named_parameters()
                                        }))
                                        raw_sample = model.iqn.sample_taus

                                        def tracked_sample(*sample_args, **sample_kwargs):
                                            samples.append((model.training, sample_kwargs["num_taus"]))
                                            return raw_sample(*sample_args, **sample_kwargs)

                                        stack.enter_context(patch.object(
                                            model.iqn, "sample_taus", side_effect=tracked_sample,
                                        ))
                                        return built

                                    stack.enter_context(patch.object(self.tuning, loss_name, side_effect=tracked_loss))
                                    stack.enter_context(patch.object(self.tuning, "_build_model", side_effect=tracked_build))
                                    stack.enter_context(patch.object(
                                        self.tuning, "compute_iqn_interval_validation_loss",
                                        side_effect=tracked_interval,
                                    ))
                                    result = self.tuning._run_single_trial(config, self._sequence(), None)

                                resolved_mode = tau_mode or "sampled_quantiles"
                                self.assertEqual(config.training.tau_mode, resolved_mode)
                                self.assertEqual(result["tau_mode"], resolved_mode)
                                self.assertEqual(
                                    result["training_quantiles"],
                                    levels if tau_mode == "target_quantiles" else None,
                                )
                                self.assertEqual(result["validation_loss"], validation_mode)
                                training_calls = [call for call in loss_calls if call[0]]
                                validation_calls = [call for call in loss_calls if not call[0]]
                                self.assertEqual(len(training_calls), 2)
                                for _, taus, feature in training_calls:
                                    self.assertEqual(feature is not None, current)
                                    if tau_mode == "target_quantiles":
                                        self.assertEqual(taus.dtype, torch.float32)
                                        self.assertEqual(taus.device.type, "cpu")
                                        torch.testing.assert_close(taus, torch.tensor(levels))
                                    else:
                                        self.assertIsNone(taus)
                                self.assertEqual(
                                    [count for training, count in samples if training],
                                    [] if tau_mode == "target_quantiles" else [7, 7],
                                )
                                if validation_mode == "target_quantiles":
                                    self.assertEqual(validation_calls, [])
                                    self.assertEqual(len(interval_calls), 1)
                                    torch.testing.assert_close(interval_calls[0][0], torch.tensor(levels))
                                    self.assertEqual(interval_calls[0][1] is not None, current)
                                else:
                                    self.assertEqual(interval_calls, [])
                                    self.assertEqual(len(validation_calls), 1)
                                    self.assertIsNone(validation_calls[0][1])
                                    self.assertEqual(validation_calls[0][2] is not None, current)
                                self.assertTrue(np.isfinite(result["selection_score"]))
                                self.assertTrue(np.isfinite(result["train_loss"]).all())
                                self.assertTrue(np.isfinite(result["valid_loss"]).all())
                                model, initial = built_models[0]
                                self.assertTrue(any(
                                    not torch.equal(initial[name], parameter.detach())
                                    for name, parameter in model.named_parameters()
                                ))

    def test_grid_records_materialize_training_mode_and_levels(self):
        for mode in (None, "target_quantiles"):
            with self.subTest(mode=mode):
                config = self._config()
                if mode is not None:
                    config.training.tau_mode = mode
                prepared = SimpleNamespace(
                    dataset={"series": self._sequence()}, data={"series": {}},
                )
                cache = {self.tuning._prepared_data_cache_key(config): prepared}
                with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
                    record = self.tuning._run_grid_trial(
                        1, config, {}, {}, ["series"], 37, cache,
                    )
                expected_mode = mode or "sampled_quantiles"
                expected_levels = [0.1, 0.2, 0.8, 0.9] if mode else None
                self.assertEqual(record["resolved_config"]["training"]["tau_mode"], expected_mode)
                self.assertEqual(record["result"]["tau_mode"], expected_mode)
                self.assertEqual(record["result"]["training_quantiles"], expected_levels)
                sequence_result = record["result"]["sequence_results"]["series"]
                self.assertEqual(sequence_result["tau_mode"], expected_mode)
                self.assertEqual(sequence_result["training_quantiles"], expected_levels)

    def test_invalid_training_mode_fails_before_building_model(self):
        config = self._config()
        config.training.tau_mode = "unknown"
        with patch.object(self.tuning, "_build_model") as build:
            with self.assertRaisesRegex(ValueError, "tau_mode"):
                self.tuning._run_single_trial(config, self._sequence(), None)
        build.assert_not_called()

    def test_sampling_interval_setting_is_not_silently_changed(self):
        config = self._config()
        config.training.tau_mode = "target_quantiles"
        config.model.interval_mode = "sampling"
        with (
            torch.random.fork_rng(devices=[]),
            patch.object(self.tuning, "warn_iqn_training_interval_mismatch") as warning,
        ):
            result = self.tuning._run_single_trial(config, self._sequence(), None)
        warning.assert_called_once_with("target_quantiles", config.model)
        self.assertEqual(config.model.interval_mode, "sampling")
        self.assertEqual(result["interval_mode"], "sampling")
        self.assertEqual(result["sampling_num"], 11)


if __name__ == "__main__":
    unittest.main()
