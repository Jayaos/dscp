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

from dscp import run_qr_cp


class QuantileHeadRunnerTests(unittest.TestCase):
    @staticmethod
    def _config(architecture, head_type, use_current_feature):
        config = OmegaConf.create(
            {
                "device": "cpu",
                "saving_dir": "unused_qr_test_results",
                "data": {
                    "data_path": "tiny_air_data.pkl",
                    "train_ratio": 0.5,
                    "valid_ratio": 0.25,
                    "normalize": False,
                    "strided_features": "xr",
                },
                "model": {
                    "dim_model": 4,
                    "num_heads": 2,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "use_current_feature": use_current_feature,
                    "target_quantiles": [[0.9, 0.1], [0.2, 0.8]],
                    "prediction_step": 1,
                    "window_size": 3,
                },
                "training": {
                    "batch_size": 64,
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "early_stop": 1,
                },
                "plotting": {"plotting": False},
            }
        )
        if architecture == "rnn":
            config.model.rnn_type = "gru"
        if head_type is not None:
            config.model.head_type = head_type
        return config

    @staticmethod
    def _artifact():
        values = np.linspace(-1.0, 1.0, 24, dtype=np.float32)
        return {
            "series": {
                "heldout_x": np.column_stack([values, values**2]),
                "heldout_y": np.sin(values),
                "heldout_predictions": values * 0.2,
            }
        }

    def _assert_head_state(self, state, head_type):
        independent = head_type == "independent"
        self.assertEqual("base_head.weight" in state, not independent)
        self.assertEqual("increment_head.weight" in state, not independent)
        self.assertEqual("quantile_heads.0.weight" in state, independent)

    def test_runners_train_and_evaluate_each_head_configuration(self):
        runners = {
            "transformer": run_qr_cp.run_transformer_quantile_regression_cp,
            "rnn": run_qr_cp.run_rnn_quantile_regression_cp,
        }
        for architecture, runner in runners.items():
            for head_type in (None, "nondecreasing", "independent"):
                for use_current_feature in (False, True):
                    with self.subTest(
                        architecture=architecture,
                        head_type=head_type,
                        use_current_feature=use_current_feature,
                    ):
                        config = self._config(
                            architecture, head_type, use_current_feature
                        )
                        with (
                            torch.random.fork_rng(devices=[]),
                            patch.object(run_qr_cp.OmegaConf, "load", return_value=config),
                            patch.object(run_qr_cp, "load_data", return_value=self._artifact()),
                            patch.object(run_qr_cp, "save_data") as save_data,
                            patch.object(run_qr_cp, "write_excluded_points"),
                            patch.object(run_qr_cp.os, "makedirs"),
                            patch.object(run_qr_cp.torch, "save") as save_model,
                            patch.object(run_qr_cp, "tqdm", side_effect=lambda items, **kwargs: items),
                            contextlib.redirect_stdout(io.StringIO()),
                        ):
                            torch.manual_seed(7)
                            runner("unused_config.yaml")

                        save_model.assert_called_once()
                        self._assert_head_state(save_model.call_args.args[0], head_type)
                        logs = [
                            call.args[1]
                            for call in save_data.call_args_list
                            if Path(call.args[0]).name == "log.pkl"
                        ]
                        self.assertEqual(len(logs), 1)
                        sequence_log = logs[0]["series"]
                        for key in ("train_loss", "valid_loss"):
                            self.assertEqual(len(sequence_log[key]), 1)
                            self.assertTrue(np.isfinite(sequence_log[key]).all())
                        retained = 6
                        if head_type == "independent":
                            metadata = sequence_log["metadata"]
                            self.assertEqual(len(metadata["valid_prediction_mask"]), 6)
                            retained = sum(metadata["valid_prediction_mask"])
                            self.assertEqual(metadata["evaluated_points"], retained)
                        for result in sequence_log["evaluation_results"].values():
                            for key in (
                                "lower_interval",
                                "upper_interval",
                                "lower_residual_quantile",
                                "upper_residual_quantile",
                            ):
                                endpoints = np.asarray(result[key])
                                self.assertEqual(endpoints.shape, (retained,))
                                self.assertTrue(np.isfinite(endpoints).all())

    def test_tuning_builder_selects_heads_and_defaults_for_both_encoders(self):
        sbatch_path = str(Path(__file__).resolve().parents[1] / "sbatch")
        with patch.object(sys, "path", [sbatch_path, *sys.path]):
            tuning = importlib.import_module("sbatch_run_tuning.run_qr_cp_tuning")

        for architecture in ("transformer", "rnn"):
            for head_type in (None, "nondecreasing", "independent"):
                for use_current_feature in (False, True):
                    with self.subTest(
                        architecture=architecture,
                        head_type=head_type,
                        use_current_feature=use_current_feature,
                    ), torch.random.fork_rng(devices=[]):
                        config = self._config(
                            architecture, head_type, use_current_feature
                        )
                        model, loss_fn, model_type, uses_current = tuning._build_model(
                            config, dim_feature=3, dim_x=2
                        )
                        self.assertEqual(model_type, architecture)
                        self.assertEqual(uses_current, use_current_feature)
                        self._assert_head_state(model.state_dict(), head_type)
                        context = torch.zeros(2, 3, 3)
                        current = torch.zeros(2, 1, 2) if uses_current else None
                        loss = loss_fn(
                            model,
                            context,
                            torch.zeros(2, 1),
                            config.model.target_quantiles,
                            current,
                        )
                        self.assertEqual(loss.ndim, 0)
                        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
