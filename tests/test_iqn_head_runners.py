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

from dscp import run_iqn_cp


class IQNPredictionHeadRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def _config(architecture, prediction_head, use_current_feature):
        config = OmegaConf.create(
            {
                "device": "cpu",
                "saving_dir": "unused_iqn_test_results",
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
                    "iqn_hidden_dim": 5,
                    "cos_emb_dim": 6,
                    "monotonic_num_layers": 2,
                    "monotonic_activation": "tanh",
                    "num_taus": 7,
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
        if prediction_head is not None:
            config.model.prediction_head = prediction_head
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

    def _assert_head_state(self, state, prediction_head):
        monotonic = prediction_head == "partially_monotonic"
        self.assertEqual("iqn.raw_tau_weight" in state, monotonic)
        self.assertEqual(
            "iqn.quantile_embedding.output_layer.0.weight" in state,
            not monotonic,
        )

    def test_runners_train_and_evaluate_each_prediction_head(self):
        runners = {
            "transformer": run_iqn_cp.run_transformer_iqn_cp,
            "rnn": run_iqn_cp.run_rnn_iqn_cp,
        }
        for architecture, runner in runners.items():
            for prediction_head, interval_mode in (
                (None, None),
                ("cosine_embedding", "sampling"),
                ("cosine_embedding", "direct"),
                ("partially_monotonic", None),
            ):
                for use_current_feature in (False, True):
                    with self.subTest(
                        architecture=architecture,
                        prediction_head=prediction_head,
                        interval_mode=interval_mode,
                        use_current_feature=use_current_feature,
                    ):
                        config = self._config(
                            architecture,
                            prediction_head,
                            use_current_feature,
                        )
                        if interval_mode is not None:
                            config.model.interval_mode = interval_mode
                            config.model.sampling_num = 23
                            config.model.iqn_num_layers = 3 if interval_mode == "direct" else 2
                            config.training.validation_loss = "target_quantiles"
                        with (
                            torch.random.fork_rng(devices=[]),
                            patch.object(run_iqn_cp.OmegaConf, "load", return_value=config),
                            patch.object(run_iqn_cp.OmegaConf, "save") as save_config,
                            patch.object(run_iqn_cp, "load_data", return_value=self._artifact()),
                            patch.object(run_iqn_cp, "save_data") as save_data,
                            patch.object(run_iqn_cp.os, "makedirs"),
                            patch.object(run_iqn_cp.torch, "save") as save_model,
                            patch.object(run_iqn_cp, "tqdm", side_effect=lambda items, **kwargs: items),
                            contextlib.redirect_stdout(io.StringIO()),
                        ):
                            torch.manual_seed(7)
                            runner("unused_config.yaml")

                        selected = prediction_head or "cosine_embedding"
                        save_config.assert_called_once()
                        self.assertEqual(
                            Path(save_config.call_args.kwargs["f"]).name,
                            "resolved_config.yaml",
                        )
                        self.assertEqual(
                            save_config.call_args.kwargs[
                                "config"
                            ].model.prediction_head,
                            selected,
                        )
                        save_model.assert_called_once()
                        self._assert_head_state(save_model.call_args.args[0], selected)
                        logs = [
                            call.args[1]
                            for call in save_data.call_args_list
                            if Path(call.args[0]).name == "log.pkl"
                        ]
                        self.assertEqual(len(logs), 1)
                        sequence_log = logs[0]["series"]
                        self.assertEqual(sequence_log["prediction_head"], selected)
                        expected_mode = (
                            "direct" if selected == "partially_monotonic"
                            else interval_mode or "sampling"
                        )
                        self.assertEqual(sequence_log["interval_mode"], expected_mode)
                        expected_depth = (
                            None if selected == "partially_monotonic"
                            else config.model.get("iqn_num_layers", 1)
                        )
                        self.assertEqual(sequence_log["iqn_num_layers"], expected_depth)
                        if selected == "cosine_embedding":
                            state = save_model.call_args.args[0]
                            context_dim = 4 + (2 if use_current_feature else 0)
                            self.assertEqual(
                                state["iqn.quantile_embedding.output_layer.0.weight"].shape,
                                (context_dim, 6),
                            )
                            self.assertEqual(state["iqn.output_layer.0.weight"].shape, (5, context_dim))
                            prediction_weights = [
                                value for name, value in state.items()
                                if name.startswith("iqn.output_layer.") and name.endswith(".weight")
                            ]
                            self.assertEqual(len(prediction_weights), expected_depth + 1)
                            self.assertEqual(sequence_log["model_config"]["iqn_num_layers"], expected_depth)
                        if interval_mode is not None:
                            self.assertEqual(sequence_log["sampling_num"], 23)
                        self.assertEqual(
                            sequence_log["model_config"].get(
                                "prediction_head",
                                "cosine_embedding",
                            ),
                            selected,
                        )
                        for key in ("train_loss", "valid_loss"):
                            self.assertEqual(len(sequence_log[key]), 1)
                            self.assertTrue(np.isfinite(sequence_log[key]).all())
                        for result in sequence_log["evaluation_results"].values():
                            lower = np.asarray(result["lower_interval"])
                            upper = np.asarray(result["upper_interval"])
                            self.assertEqual(lower.shape, (6,))
                            self.assertEqual(upper.shape, (6,))
                            self.assertTrue(np.isfinite(lower).all())
                            self.assertTrue(np.isfinite(upper).all())
                            if selected == "partially_monotonic" or expected_mode == "sampling":
                                self.assertTrue(np.all(lower <= upper + 1e-6))

    def test_tuning_builder_propagates_prediction_head(self):
        sbatch_path = str(Path(__file__).resolve().parents[1] / "sbatch")
        with patch.object(sys, "path", [sbatch_path, *sys.path]):
            tuning = importlib.import_module("sbatch_run_tuning.run_iqn_cp_tuning")

        for architecture in ("transformer", "rnn"):
            for prediction_head, interval_mode in (
                (None, None),
                ("cosine_embedding", "direct"),
                ("cosine_embedding", "sampling"),
                ("partially_monotonic", None),
            ):
                with self.subTest(
                    architecture=architecture,
                    prediction_head=prediction_head,
                    interval_mode=interval_mode,
                ):
                    config = self._config(
                        architecture,
                        prediction_head,
                        use_current_feature=True,
                    )
                    if interval_mode is not None:
                        config.model.interval_mode = interval_mode
                        config.model.sampling_num = 19
                        config.model.iqn_num_layers = 3 if interval_mode == "direct" else 2
                    model, loss_fn, model_type, uses_current = tuning._build_model(
                        config,
                        dim_feature=3,
                        dim_x=2,
                    )
                    selected = prediction_head or "cosine_embedding"
                    self.assertEqual(model_type, architecture)
                    self.assertTrue(uses_current)
                    self.assertEqual(model.prediction_head, selected)
                    self._assert_head_state(model.state_dict(), selected)
                    if selected == "cosine_embedding":
                        expected_depth = config.model.get("iqn_num_layers", 1)
                        self.assertEqual(model.iqn.iqn_num_layers, expected_depth)
                        self.assertEqual(model.iqn.quantile_embedding.output_layer[0].weight.shape, (6, 6))
                        prediction_linears = [
                            layer for layer in model.iqn.output_layer
                            if isinstance(layer, torch.nn.Linear)
                        ]
                        self.assertEqual(len(prediction_linears), expected_depth + 1)
                        self.assertEqual(prediction_linears[0].weight.shape, (5, 6))

                    context = torch.zeros(2, 3, 3)
                    current = torch.zeros(2, 1, 2)
                    loss = loss_fn(
                        model,
                        context,
                        torch.zeros(2, 1),
                        config.model.num_taus,
                        current,
                    )
                    self.assertEqual(loss.ndim, 0)
                    self.assertTrue(torch.isfinite(loss).item())
                    expected_mode = (
                        "direct" if selected == "partially_monotonic"
                        else interval_mode or "sampling"
                    )
                    self.assertEqual(model.iqn.interval_mode, expected_mode)
                    model.eval()
                    with patch.object(
                        model.iqn, "sample_taus", wraps=model.iqn.sample_taus,
                    ) as sample_taus:
                        predictions = model.get_predicted_quantile_values(
                            model, context, torch.tensor([0.1, 0.9]), current,
                        )
                    self.assertEqual(predictions.shape, (2, 2))
                    if expected_mode == "sampling":
                        sample_taus.assert_called_once()
                        self.assertEqual(
                            sample_taus.call_args.kwargs["num_taus"],
                            19 if interval_mode is not None else 1000,
                        )
                    else:
                        sample_taus.assert_not_called()

    def test_monotonic_config_ignores_legacy_embedding_dimensions(self):
        config = self._config(
            "rnn",
            "partially_monotonic",
            use_current_feature=False,
        )
        config.model.monotonic_hidden_dims = [6, 4]
        config.model.iqn_hidden_dim = "unused-by-explicit-widths"
        config.model.cos_emb_dim = "unused-by-monotonic-head"
        config.model.interval_mode = "unused-by-monotonic-head"
        config.model.sampling_num = "unused-by-monotonic-head"
        config.model.iqn_num_layers = "unused-by-monotonic-head"

        self.assertEqual(
            run_iqn_cp._prediction_head_dimensions(config.model),
            (None, 64),
        )

        sbatch_path = str(Path(__file__).resolve().parents[1] / "sbatch")
        with patch.object(sys, "path", [sbatch_path, *sys.path]):
            tuning = importlib.import_module("sbatch_run_tuning.run_iqn_cp_tuning")
        model, _, _, _ = tuning._build_model(
            config,
            dim_feature=3,
            dim_x=2,
        )
        self.assertEqual(model.iqn.hidden_dims, (6, 4))

        cosine_config = self._config(
            "rnn",
            "cosine_embedding",
            use_current_feature=False,
        )
        cosine_config.model.iqn_hidden_dim = None
        self.assertEqual(
            run_iqn_cp._prediction_head_dimensions(cosine_config.model),
            (None, 6),
        )
        cosine_model, _, _, _ = tuning._build_model(
            cosine_config,
            dim_feature=3,
            dim_x=2,
        )
        self.assertEqual(cosine_model.prediction_head, "cosine_embedding")
        self.assertEqual(cosine_model.iqn.hidden_dim, 4)

    def test_checked_configs_route_each_head_to_a_distinct_directory(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs" / "iqn_cp_configs"
        config_names = (
            "iqn_rnn_lr_air_config.yaml",
            "iqn_transformer_lr_air_config.yaml",
            "iqn_rnn_lstm_sapflux_config.yaml",
            "iqn_transformer_lstm_sapflux_config.yaml",
        )
        for config_name in config_names:
            with self.subTest(config=config_name):
                config = OmegaConf.load(config_dir / config_name)
                self.assertIn(
                    config.model.prediction_head,
                    ("partially_monotonic", "cosine_embedding"),
                )
                # Exercise both routes without constraining the experiment's
                # currently selected head.
                config.model.prediction_head = "partially_monotonic"
                monotonic_dir = config.saving_dir
                self.assertIn("partially_monotonic", monotonic_dir)

                config.model.prediction_head = "cosine_embedding"
                cosine_dir = config.saving_dir
                self.assertIn("cosine_embedding", cosine_dir)
                self.assertNotEqual(monotonic_dir, cosine_dir)


if __name__ == "__main__":
    unittest.main()
