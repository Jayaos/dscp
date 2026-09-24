import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dscp import run_iqn_cp
from tests import test_iqn_head_runners as runner_fixtures


class IQNTauRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def _run(self, architecture, config):
        runner = getattr(run_iqn_cp, f"run_{architecture}_iqn_cp")
        loss_name = f"compute_loss_iqn_{architecture}"
        original_loss = getattr(run_iqn_cp, loss_name)
        original_interval_loss = run_iqn_cp.compute_iqn_interval_validation_loss
        records = []

        def raw_loss(model, inputs, target, num_taus, current_feature=None, *, taus=None):
            with patch.object(model.iqn, "sample_taus", wraps=model.iqn.sample_taus) as sampler:
                loss = original_loss(
                    model, inputs, target, num_taus, current_feature, taus=taus,
                )
            records.append({
                "training": model.training,
                "taus": None if taus is None else taus.detach().cpu(),
                "sample_counts": [c.kwargs["num_taus"] for c in sampler.call_args_list],
                "has_current": current_feature is not None,
                "requires_grad": loss.requires_grad,
            })
            return loss

        def interval_loss(model, inputs, target, quantiles, current_feature=None, **kwargs):
            records.append({
                "training": model.training,
                "taus": quantiles.detach().cpu(),
                "sample_counts": [],
                "has_current": current_feature is not None,
                "requires_grad": False,
            })
            return original_interval_loss(
                model, inputs, target, quantiles, current_feature, **kwargs,
            )

        with (
            torch.random.fork_rng(devices=[]),
            patch.object(run_iqn_cp, "load_experiment_config", return_value=config),
            patch.object(run_iqn_cp.OmegaConf, "save") as save_config,
            patch.object(
                run_iqn_cp, "load_data",
                return_value=runner_fixtures.IQNPredictionHeadRunnerTests._artifact(),
            ),
            patch.object(run_iqn_cp, "save_data") as save_data,
            patch.object(run_iqn_cp, "write_excluded_points"),
            patch.object(run_iqn_cp.os, "makedirs"),
            patch.object(run_iqn_cp.torch, "save") as save_model,
            patch.object(run_iqn_cp, loss_name, side_effect=raw_loss),
            patch.object(run_iqn_cp, "compute_iqn_interval_validation_loss", side_effect=interval_loss),
            patch.object(run_iqn_cp, "tqdm", side_effect=lambda items, **kwargs: items),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            torch.manual_seed(41)
            runner("unused_config.yaml")

        logs = [c.args[1] for c in save_data.call_args_list if Path(c.args[0]).name == "log.pkl"]
        self.assertEqual(len(logs), 1)
        self.assertEqual(save_config.call_count, 1)
        self.assertEqual(save_model.call_count, 1)
        return logs[0]["series"], records, save_model.call_args.args[0]

    def test_training_and_validation_modes_are_independent_for_both_encoders(self):
        expected_taus = torch.tensor([0.1, 0.2, 0.8, 0.9])
        for architecture in ("rnn", "transformer"):
            for head in ("cosine_embedding", "partially_monotonic"):
                for use_current in (False, True):
                    for tau_mode in ("sampled_quantiles", "target_quantiles"):
                        for validation_mode in ("sampled_quantiles", "target_quantiles"):
                            with self.subTest(
                                architecture=architecture, head=head, use_current=use_current,
                                tau_mode=tau_mode, validation_mode=validation_mode,
                            ):
                                config = runner_fixtures.IQNPredictionHeadRunnerTests._config(
                                    architecture, head, use_current,
                                )
                                config.model.interval_mode = "direct"
                                # Shared endpoints must be trained once, not reweighted by pairs.
                                config.model.target_quantiles.append([0.9, 0.2])
                                config.training.tau_mode = tau_mode
                                config.training.validation_loss = validation_mode
                                log, records, _ = self._run(architecture, config)

                                training_records = [r for r in records if r["training"]]
                                validation_records = [r for r in records if not r["training"]]
                                self.assertTrue(training_records)
                                self.assertTrue(validation_records)
                                for group, selected_mode, is_training in (
                                    (training_records, tau_mode, True),
                                    (validation_records, validation_mode, False),
                                ):
                                    for record in group:
                                        self.assertEqual(record["has_current"], use_current)
                                        self.assertEqual(record["requires_grad"], is_training)
                                        if selected_mode == "target_quantiles":
                                            torch.testing.assert_close(record["taus"], expected_taus)
                                            self.assertEqual(record["sample_counts"], [])
                                        else:
                                            self.assertIsNone(record["taus"])
                                            self.assertEqual(record["sample_counts"], [7])

                                self.assertEqual(log["tau_mode"], tau_mode)
                                self.assertEqual(config.training.tau_mode, tau_mode)
                                self.assertEqual(log["validation_loss"], validation_mode)
                                self.assertEqual(config.model.interval_mode, "direct")
                                if tau_mode == "target_quantiles":
                                    np.testing.assert_allclose(log["training_quantiles"], expected_taus)
                                else:
                                    self.assertIsNone(log["training_quantiles"])
                                for key in ("train_loss", "valid_loss"):
                                    self.assertTrue(np.isfinite(log[key]).all())

    def test_omitted_mode_matches_explicit_sampled_training(self):
        for architecture in ("rnn", "transformer"):
            with self.subTest(architecture=architecture):
                results = []
                for tau_mode in (None, "sampled_quantiles"):
                    config = runner_fixtures.IQNPredictionHeadRunnerTests._config(
                        architecture, "cosine_embedding", False,
                    )
                    config.model.interval_mode = "direct"
                    config.training.validation_loss = "target_quantiles"
                    if tau_mode is not None:
                        config.training.tau_mode = tau_mode
                    results.append(self._run(architecture, config))
                    self.assertEqual(config.training.tau_mode, "sampled_quantiles")
                for key in ("train_loss", "valid_loss", "training_quantiles", "tau_mode"):
                    self.assertEqual(results[0][0][key], results[1][0][key])
                for key, value in results[0][2].items():
                    torch.testing.assert_close(value, results[1][2][key], rtol=0, atol=0)

    def test_target_training_with_sampling_intervals_warns_without_overriding(self):
        config = runner_fixtures.IQNPredictionHeadRunnerTests._config(
            "rnn", "cosine_embedding", False,
        )
        config.training.tau_mode = "target_quantiles"
        config.training.validation_loss = "target_quantiles"
        config.model.interval_mode = "sampling"
        config.model.sampling_num = 17
        with self.assertWarnsRegex(UserWarning, "interval_mode"):
            log, records, _ = self._run("rnn", config)
        self.assertEqual(log["interval_mode"], "sampling")
        self.assertEqual(config.model.interval_mode, "sampling")
        self.assertTrue(all(not r["sample_counts"] for r in records if r["training"]))

    def test_invalid_mode_fails_before_creating_outputs_or_loading_data(self):
        for architecture in ("rnn", "transformer"):
            with self.subTest(architecture=architecture):
                config = runner_fixtures.IQNPredictionHeadRunnerTests._config(
                    architecture, "cosine_embedding", False,
                )
                config.training.tau_mode = "invalid"
                with (
                    patch.object(run_iqn_cp, "load_experiment_config", return_value=config),
                    patch.object(run_iqn_cp.os, "makedirs") as mkdir,
                    patch.object(run_iqn_cp, "load_data") as load_data,
                    self.assertRaisesRegex(ValueError, "training.tau_mode"),
                ):
                    getattr(run_iqn_cp, f"run_{architecture}_iqn_cp")("unused_config.yaml")
                mkdir.assert_not_called()
                load_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
