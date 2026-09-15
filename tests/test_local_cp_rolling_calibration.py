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


class LocalCPRollingCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sbatch_path = str(Path(__file__).resolve().parents[1] / "sbatch")
        with patch.object(sys, "path", [sbatch_path, *sys.path]):
            cls.tuning = importlib.import_module("sbatch_run_tuning.run_local_cp_tuning")

    @staticmethod
    def _config(architecture, mode):
        config = OmegaConf.create({
            "device": "cpu",
            "saving_dir": "unused_local_cp_results",
            "data": {
                "data_path": "tiny_air_data.pkl",
                "train_ratio": 0.6,
                "valid_ratio": 0.1,
                "calibration_ratio": 0.1,
                "test_ratio": 0.2,
                "normalize": False,
                "strided_features": "xr",
            },
            "model": {
                "dim_model": 4,
                "num_heads": 2,
                "num_layers": 1,
                "dropout": 0.0,
                "use_current_feature": False,
                "training_quantiles": [0.1, 0.5, 0.9],
                "target_quantiles": [[0.05, 0.95]],
                "prediction_step": 1,
                "window_size": 3,
                "calibration_size": 2,
                "similarity_fn": "cos_similarity",
                "temperature": 1.0,
                "sampling_num": 16,
            },
            "training": {
                "batch_size": 64,
                "learning_rate": 0.001,
                "epochs": 1,
                "early_stop": 1,
            },
            "tuning": {"model_selection_valid_ratio": 0.2},
            "plotting": {"plotting": False},
        })
        if architecture == "rnn":
            config.model.rnn_type = "gru"
        if mode != "missing":
            config.model.rolling_calibration = mode
        return config

    @staticmethod
    def _artifact():
        # Each residual uniquely identifies its position in the chronology.
        values = np.arange(40, dtype=np.float32)
        return {"series": {
            "heldout_x": values[:, None],
            "heldout_y": values,
            "heldout_predictions": np.zeros_like(values),
        }}

    def _run(self, architecture, mode, tuning):
        config = self._config(architecture, mode)
        calls = []
        approximate = run_local_cp.LocalConformalPrediction.approximate_quantile

        def record_pool(local_cp, query, levels, sampling_num):
            self.assertFalse(query.requires_grad)
            calls.append({
                "representations": local_cp.encoded_rep.detach().cpu().clone(),
                "residuals": local_cp.target.detach().cpu().clone(),
                "query": query.detach().cpu().clone(),
            })
            return approximate(local_cp, query, levels, sampling_num)

        with (
            torch.random.fork_rng(devices=[]),
            patch.object(run_local_cp.LocalConformalPrediction, "approximate_quantile", record_pool),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            torch.manual_seed(19)
            if tuning:
                cpd = self.tuning._prepare_trial_data(self._artifact(), config)
                log = self.tuning._run_single_trial(config, cpd.dataset["series"], None)
            else:
                with (
                    patch.object(run_local_cp.OmegaConf, "load", return_value=config),
                    patch.object(run_local_cp, "load_data", return_value=self._artifact()),
                    patch.object(run_local_cp, "save_data") as save_data,
                    patch.object(run_local_cp.os, "makedirs"),
                    patch.object(run_local_cp.torch, "save"),
                    patch.object(run_local_cp, "tqdm", side_effect=lambda items, **kwargs: items),
                ):
                    getattr(run_local_cp, f"run_{architecture}_local_cp")("unused.yaml")
                log = next(
                    call.args[1]["series"] for call in save_data.call_args_list
                    if Path(call.args[0]).name == "log.pkl"
                )
        return log, calls

    def _assert_pool_sequence(self, calls, tuning, rolling):
        # Ordinary calibration is 28..31, tuning calibration is 24..27.
        # In either case the configured capacity keeps only the last two.
        first_target = 28 if tuning else 32
        num_predictions = 4 if tuning else 8
        self.assertEqual(len(calls), num_predictions)
        expected_residuals = torch.arange(first_target - 2, first_target).float().reshape(-1, 1)
        expected_repr = calls[0]["representations"]
        for offset, call in enumerate(calls):
            with self.subTest(prediction=offset):
                torch.testing.assert_close(call["residuals"], expected_residuals, rtol=0, atol=0)
                torch.testing.assert_close(call["representations"], expected_repr, rtol=0, atol=0)
                # The current outcome has not entered the pool for its own interval.
                self.assertTrue(torch.all(call["residuals"] < first_target + offset))
            if rolling:
                observed = torch.tensor([[float(first_target + offset)]])
                expected_residuals = torch.cat((expected_residuals[1:], observed))
                expected_repr = torch.cat((expected_repr[1:], call["query"]))

    def test_fixed_rolling_and_default_pools_in_both_runner_types(self):
        for architecture in ("transformer", "rnn"):
            for tuning in (False, True):
                recorded = {}
                for mode in (False, True, "missing"):
                    with self.subTest(architecture=architecture, tuning=tuning, mode=mode):
                        log, calls = self._run(architecture, mode, tuning)
                        rolling = mode is not False
                        self.assertIs(log["rolling_calibration"], rolling)
                        self._assert_pool_sequence(calls, tuning, rolling)
                        recorded[mode] = calls

                # Omitting the option preserves precisely the enabled behavior.
                for explicit, default in zip(recorded[True], recorded["missing"]):
                    for field in explicit:
                        torch.testing.assert_close(explicit[field], default[field], rtol=0, atol=0)
                # Changing calibration policy does not change training or encoding.
                for fixed, rolling in zip(recorded[False], recorded[True]):
                    torch.testing.assert_close(fixed["query"], rolling["query"], rtol=0, atol=0)
                torch.testing.assert_close(
                    recorded[False][0]["representations"],
                    recorded[True][0]["representations"], rtol=0, atol=0,
                )

    def test_non_boolean_options_fail_before_training(self):
        for architecture in ("transformer", "rnn"):
            for invalid in (None, "false", "true", 0, 1, [], {}):
                with self.subTest(architecture=architecture, invalid=invalid):
                    config = self._config(architecture, invalid)
                    with (
                        patch.object(run_local_cp.OmegaConf, "load", return_value=config),
                        patch.object(run_local_cp, "load_data") as load_data,
                        patch.object(run_local_cp.os, "makedirs") as mkdir,
                        patch.object(run_local_cp.torch.optim, "AdamW") as optimizer,
                        self.assertRaisesRegex(ValueError, "rolling_calibration"),
                    ):
                        getattr(run_local_cp, f"run_{architecture}_local_cp")("unused.yaml")
                    load_data.assert_not_called()
                    mkdir.assert_not_called()
                    optimizer.assert_not_called()

                    with (
                        patch.object(self.tuning, "_build_model") as build_model,
                        self.assertRaisesRegex(ValueError, "rolling_calibration"),
                    ):
                        self.tuning._run_single_trial(config, {}, None)
                    build_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
