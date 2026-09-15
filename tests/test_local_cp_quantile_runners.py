import contextlib
import importlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dscp import run_local_cp


class LocalCPQuantileRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sbatch_path = str(Path(__file__).resolve().parents[1] / "sbatch")
        with patch.object(sys, "path", [sbatch_path, *sys.path]):
            cls.tuning = importlib.import_module("sbatch_run_tuning.run_local_cp_tuning")

    @staticmethod
    def _config(architecture, use_current_feature=False):
        config = OmegaConf.create({
            "device": "cpu",
            "saving_dir": "unused_local_cp_results",
            "data": {
                "data_path": "tiny_air_data.pkl",
                "train_ratio": 0.6,
                "valid_ratio": 0.1,
                "calibration_ratio": 0.1,
                "test_ratio": 0.2,
                "normalize": use_current_feature,
                "strided_features": "xr",
            },
            "model": {
                "dim_model": 4,
                "num_heads": 2,
                "num_layers": 1,
                "dropout": 0.0,
                "use_current_feature": use_current_feature,
                # Deliberately different levels and counts from inference.
                "training_quantiles": [0.8, 0.2, 0.5],
                "target_quantiles": [[0.95, 0.05]],
                "prediction_step": 1,
                "window_size": 3,
                "calibration_size": 4,
                "similarity_fn": "cos_similarity",
                "temperature": 1.0,
                "sampling_num": 64,
            },
            "training": {
                "batch_size": 64,
                "learning_rate": 0.001,
                "epochs": 2,
                "early_stop": 1,
            },
            "tuning": {"model_selection_valid_ratio": 0.2},
            "plotting": {"plotting": False},
        })
        if architecture == "rnn":
            config.model.rnn_type = "gru"
        return config

    @staticmethod
    def _artifact():
        values = np.linspace(-2.0, 2.0, 40, dtype=np.float32)
        return {"series": {
            "heldout_x": np.column_stack([values, values**2]),
            "heldout_y": np.sin(values),
            "heldout_predictions": values * 0.2,
        }}

    def _run(self, architecture, config):
        runner = getattr(run_local_cp, f"run_{architecture}_local_cp")
        quantile_calls = []
        approximate = run_local_cp.LocalConformalPrediction.approximate_quantile

        def record_quantiles(local_cp, query, levels, sampling_num):
            quantile_calls.append(list(levels))
            self.assertFalse(query.requires_grad)
            return approximate(local_cp, query, levels, sampling_num)

        with (
            torch.random.fork_rng(devices=[]),
            patch.object(run_local_cp.OmegaConf, "load", return_value=config),
            patch.object(run_local_cp, "load_data", return_value=self._artifact()),
            patch.object(run_local_cp, "save_data") as save_data,
            patch.object(run_local_cp.os, "makedirs"),
            patch.object(run_local_cp.torch, "save") as save_model,
            patch.object(run_local_cp, "tqdm", side_effect=lambda items, **kwargs: items),
            patch.object(run_local_cp.LocalConformalPrediction, "approximate_quantile", record_quantiles),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            torch.manual_seed(7)
            runner("unused_config.yaml")

        save_model.assert_called_once()
        log = next(
            call.args[1]["series"] for call in save_data.call_args_list
            if Path(call.args[0]).name == "log.pkl"
        )
        return save_model.call_args.args[0], log, quantile_calls

    def test_inference_levels_do_not_change_encoder_training(self):
        for architecture in ("transformer", "rnn"):
            for current in (False, True):
                with self.subTest(architecture=architecture, current=current):
                    config = self._config(architecture, current)
                    config.model.similarity_fn = "euclidean" if current else "cos_similarity"
                    state, log, calls = self._run(architecture, config)
                    self.assertEqual(state["output_linear.weight"].shape[0], 3)
                    torch.testing.assert_close(
                        state["training_quantiles"], torch.tensor([0.2, 0.5, 0.8])
                    )
                    self.assertEqual(log["training_quantiles"], [0.2, 0.5, 0.8])
                    self.assertEqual(calls, [[0.05, 0.95]] * 8)
                    self.assertTrue(np.isfinite(log["train_loss"]).all())
                    self.assertTrue(np.isfinite(log["valid_loss"]).all())

                    config.model.target_quantiles = [[0.1, 0.9], [0.25, 0.75]]
                    changed_state, changed_log, changed_calls = self._run(architecture, config)
                    self.assertEqual(state.keys(), changed_state.keys())
                    for name in state:
                        torch.testing.assert_close(state[name], changed_state[name], rtol=0, atol=0)
                    self.assertEqual(log["train_loss"], changed_log["train_loss"])
                    self.assertEqual(log["valid_loss"], changed_log["valid_loss"])
                    self.assertEqual(changed_calls, [[0.1, 0.25, 0.75, 0.9]] * 8)
                    results = changed_log["evaluation_results"]
                    self.assertEqual(set(results), {(0.1, 0.9), (0.25, 0.75)})
                    outer, inner = results[(0.1, 0.9)], results[(0.25, 0.75)]
                    self.assertTrue(np.all(np.asarray(outer["lower_interval"]) <= inner["lower_interval"]))
                    self.assertTrue(np.all(np.asarray(inner["upper_interval"]) <= outer["upper_interval"]))
                    for result in results.values():
                        self.assertEqual(len(result["coverage"]), 8)
                        self.assertTrue(np.isfinite(result["lower_interval"]).all())
                        self.assertTrue(np.isfinite(result["upper_interval"]).all())

    def test_tuning_uses_training_levels_and_keeps_inference_independent(self):
        for architecture in ("transformer", "rnn"):
            with self.subTest(architecture=architecture):
                config = self._config(architecture, use_current_feature=True)
                config.model.similarity_fn = "euclidean"
                config.model.training_quantiles = [0.15, 0.35, 0.65, 0.85]
                model, _, model_type, uses_current = self.tuning._build_model(config, 3, 2)
                self.assertEqual(model_type, architecture)
                self.assertTrue(uses_current)
                self.assertEqual(model.output_linear.out_features, 4)
                torch.testing.assert_close(
                    model.training_quantiles, torch.tensor([0.15, 0.35, 0.65, 0.85])
                )
                cpd = self.tuning._prepare_trial_data(self._artifact(), config)
                normalized = self.tuning._normalization_params(config, cpd.data["series"])
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(11)
                    first = self.tuning._run_single_trial(config, cpd.dataset["series"], normalized)
                    config.model.target_quantiles = [[0.25, 0.75]]
                    torch.manual_seed(11)
                    second = self.tuning._run_single_trial(config, cpd.dataset["series"], normalized)
                self.assertEqual(first["train_loss"], second["train_loss"])
                self.assertEqual(first["model_selection_valid_loss"], second["model_selection_valid_loss"])
                self.assertEqual(first["training_quantiles"], second["training_quantiles"])
                np.testing.assert_allclose(first["training_quantiles"], [0.15, 0.35, 0.65, 0.85])
                self.assertTrue(np.isfinite(first["train_loss"]).all())
                self.assertTrue(np.isfinite(first["model_selection_valid_loss"]).all())
                self.assertEqual(first["num_tuning_evaluation_samples"], 4)
                self.assertFalse(first["final_test_evaluated"])

    def test_training_levels_are_required_without_inference_fallback(self):
        for architecture in ("transformer", "rnn"):
            with self.subTest(architecture=architecture):
                config = self._config(architecture)
                del config.model.training_quantiles
                with self.assertRaisesRegex(ValueError, "training_quantiles"):
                    self.tuning._build_model(config, 3, 2)
                with (
                    patch.object(run_local_cp.OmegaConf, "load", return_value=config),
                    patch.object(run_local_cp.os, "makedirs") as mkdir,
                    self.assertRaisesRegex(ValueError, "training_quantiles"),
                ):
                    getattr(run_local_cp, f"run_{architecture}_local_cp")("unused.yaml")
                mkdir.assert_not_called()


if __name__ == "__main__":
    unittest.main()
