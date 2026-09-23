import contextlib
import importlib
import inspect
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dscp import run_iqn_cp
from dscp.data import QuantileRegressionDataset
from dscp.loss import (
    compute_loss_iqn_rnn,
    compute_loss_iqn_transformer,
    resolve_iqn_validation_quantiles,
)
from dscp.models.iqn_rnn import IQNRNN
from dscp.models.iqn_transformer import IQNTransformer


class _FixedQuantileModel(torch.nn.Module):
    """Small model that records whether fixed levels reach the IQN wrapper."""

    def __init__(self, quantile_values):
        super().__init__()
        self.register_buffer(
            "quantile_values",
            torch.as_tensor(quantile_values, dtype=torch.float32),
        )
        self.calls = []

    def forward(self, inputs, **kwargs):
        self.calls.append(kwargs)
        taus = torch.as_tensor(
            kwargs["taus"],
            dtype=inputs.dtype,
            device=inputs.device,
        )
        if taus.ndim == 1:
            taus = taus.unsqueeze(0).expand(inputs.shape[0], -1)
        return self.quantile_values.to(inputs), taus


class IQNValidationLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)

    def test_fixed_levels_have_exact_pinball_loss_for_both_encoders(self):
        # Errors are [[1, -1], [-1, 2]].  At taus [.25, .75], the four
        # pinball terms are [.25, .25, .75, 1.5], whose mean is .6875.
        expected = torch.tensor(0.6875)
        inputs = torch.zeros(2, 3, 1)
        targets = torch.tensor([[1.0], [3.0]])
        current_feature = torch.zeros(2, 1, 1)
        taus = torch.tensor([0.25, 0.75])

        for loss_fn in (compute_loss_iqn_rnn, compute_loss_iqn_transformer):
            with self.subTest(loss_fn=loss_fn.__name__):
                model = _FixedQuantileModel([[0.0, 2.0], [4.0, 1.0]])
                actual = loss_fn(
                    model,
                    inputs,
                    targets,
                    num_taus=19,
                    current_feature=current_feature,
                    taus=taus,
                )

                torch.testing.assert_close(actual, expected)
                self.assertEqual(len(model.calls), 1)
                torch.testing.assert_close(model.calls[0]["taus"], taus)
                self.assertEqual(model.calls[0]["num_taus"], 19)
                self.assertIs(
                    model.calls[0]["current_feature"],
                    current_feature,
                )

        for loss_fn in (compute_loss_iqn_rnn, compute_loss_iqn_transformer):
            self.assertEqual(
                inspect.signature(loss_fn).parameters["taus"].kind,
                inspect.Parameter.KEYWORD_ONLY,
            )

    @staticmethod
    def _real_model(architecture, prediction_head):
        common = dict(
            dim_feature=3,
            dim_model=6,
            num_layers=1,
            current_feature_dim=2,
            iqn_hidden_dim=5,
            n_cos_embedding=7,
            dropout=0.0,
            prediction_head=prediction_head,
            monotonic_num_layers=2,
        )
        if architecture == "transformer":
            return IQNTransformer(num_head=2, dim_ff=12, **common)
        return IQNRNN(rnn_type="gru", **common)

    def test_fixed_levels_are_deterministic_and_do_not_consume_rng(self):
        inputs = torch.randn(3, 4, 3)
        targets = torch.randn(3, 1)
        current_feature = torch.randn(3, 1, 2)
        taus = torch.tensor([0.1, 0.4, 0.9])

        for architecture in ("rnn", "transformer"):
            for prediction_head in (
                "cosine_embedding",
                "partially_monotonic",
            ):
                with self.subTest(
                    architecture=architecture,
                    prediction_head=prediction_head,
                ):
                    model = self._real_model(architecture, prediction_head)
                    model.eval()
                    loss_fn = (
                        compute_loss_iqn_transformer
                        if architecture == "transformer"
                        else compute_loss_iqn_rnn
                    )

                    rng_before = torch.random.get_rng_state().clone()
                    first = loss_fn(
                        model,
                        inputs,
                        targets,
                        num_taus=1,
                        current_feature=current_feature,
                        taus=taus,
                    )
                    rng_middle = torch.random.get_rng_state().clone()
                    second = loss_fn(
                        model,
                        inputs,
                        targets,
                        num_taus=101,
                        current_feature=current_feature,
                        taus=taus,
                    )
                    rng_after = torch.random.get_rng_state()

                    torch.testing.assert_close(first, second)
                    torch.testing.assert_close(rng_before, rng_middle)
                    torch.testing.assert_close(rng_before, rng_after)

    def test_omitting_fixed_levels_still_samples_during_training(self):
        inputs = torch.randn(3, 4, 3)
        targets = torch.randn(3, 1)
        current_feature = torch.randn(3, 1, 2)

        for architecture in ("rnn", "transformer"):
            for prediction_head in (
                "cosine_embedding",
                "partially_monotonic",
            ):
                with self.subTest(
                    architecture=architecture,
                    prediction_head=prediction_head,
                ):
                    model = self._real_model(architecture, prediction_head)
                    model.eval()
                    loss_fn = (
                        compute_loss_iqn_transformer
                        if architecture == "transformer"
                        else compute_loss_iqn_rnn
                    )

                    rng_before = torch.random.get_rng_state().clone()
                    loss = loss_fn(
                        model,
                        inputs,
                        targets,
                        num_taus=7,
                        current_feature=current_feature,
                    )
                    rng_after = torch.random.get_rng_state()

                    self.assertTrue(torch.isfinite(loss).item())
                    self.assertFalse(torch.equal(rng_before, rng_after))


class IQNValidationResolverTests(unittest.TestCase):
    def test_target_levels_are_sorted_and_deduplicated(self):
        mode, levels = resolve_iqn_validation_quantiles(
            OmegaConf.create({"validation_loss": "target_quantiles"}),
            [[0.9, 0.1], [0.8, 0.2], [0.9, 0.1]],
        )
        self.assertEqual(mode, "target_quantiles")
        self.assertEqual(levels, [0.1, 0.2, 0.8, 0.9])

    def test_sampled_mode_and_legacy_config_have_no_fixed_levels(self):
        for training_config in (
            {},
            OmegaConf.create({}),
            {"validation_loss": "sampled_quantiles"},
        ):
            with self.subTest(training_config=training_config):
                self.assertEqual(
                    resolve_iqn_validation_quantiles(
                        training_config,
                        [[0.9, 0.1]],
                    ),
                    ("sampled_quantiles", None),
                )

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_iqn_validation_quantiles(
                {"validation_loss": "validation_typo"},
                [[0.9, 0.1]],
            )

    def test_invalid_target_levels_are_rejected(self):
        invalid_targets = (
            [],
            [[0.1]],
            [[0.1, 0.5, 0.9]],
            [[0.5, 0.5]],
            [[0.0, 0.9]],
            [[0.1, 1.0]],
            [[float("nan"), 0.9]],
            [[0.1, float("inf")]],
            [[True, 0.9]],
            [["0.1", 0.9]],
        )
        for target_quantiles in invalid_targets:
            with self.subTest(target_quantiles=target_quantiles):
                with self.assertRaises(ValueError):
                    resolve_iqn_validation_quantiles(
                        {"validation_loss": "target_quantiles"},
                        target_quantiles,
                    )


class IQNValidationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def _runner_config(architecture):
        config = OmegaConf.create(
            {
                "device": "cpu",
                "saving_dir": "unused_iqn_validation_results",
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
                    "use_current_feature": False,
                    "prediction_head": "partially_monotonic",
                    "iqn_hidden_dim": 5,
                    "monotonic_num_layers": 2,
                    "monotonic_activation": "tanh",
                    "cos_emb_dim": 6,
                    "num_taus": 7,
                    "target_quantiles": [[0.9, 0.1], [0.2, 0.8]],
                    "prediction_step": 1,
                    "window_size": 3,
                },
                "training": {
                    "batch_size": 4,
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "epochs": 4,
                    "early_stop": 1,
                    "validation_loss": "target_quantiles",
                },
                "plotting": {"plotting": False},
            }
        )
        if architecture == "rnn":
            config.model.rnn_type = "gru"
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

    @staticmethod
    def _criterion_loss_spy():
        records = []
        validation_snapshots = []
        # Validation has batches of four and two.  Sample-weighted epoch means
        # are 2 and 7/3, selecting epoch one.  Equal batch averaging gives 3
        # and 7/4, incorrectly selecting epoch two.
        validation_batch_losses = (0.0, 6.0, 3.5, 0.0, 10.0, 10.0)

        def loss_fn(
            model,
            inputs,
            target,
            num_taus,
            current_feature=None,
            *,
            taus=None,
        ):
            is_training = torch.is_grad_enabled()
            prepared_taus = (
                None
                if taus is None
                else torch.as_tensor(taus).detach().cpu().flatten().tolist()
            )
            records.append(
                {
                    "training": is_training,
                    "batch_size": inputs.shape[0],
                    "taus": prepared_taus,
                }
            )
            if is_training:
                # Produce a real parameter update so the two epoch checkpoints
                # are distinguishable when the runner saves its best state.
                return next(model.parameters()).reshape(-1)[0]

            validation_index = sum(not record["training"] for record in records) - 1
            if validation_index % 2 == 0:
                name, parameter = next(model.named_parameters())
                validation_snapshots.append(
                    (name, parameter.detach().cpu().clone())
                )
            return torch.tensor(
                validation_batch_losses[validation_index],
                device=inputs.device,
                dtype=inputs.dtype,
            )

        return loss_fn, records, validation_snapshots

    @staticmethod
    def _interval_loss_spy(raw_loss_fn):
        def interval_loss(model, inputs, target, quantiles, current_feature=None, *, sampling_seed=None):
            return raw_loss_fn(
                model, inputs, target, 0, current_feature, taus=quantiles,
            )
        return interval_loss

    def test_runners_use_fixed_sample_weighted_validation_for_checkpoint(self):
        runners = {
            "transformer": (
                run_iqn_cp.run_transformer_iqn_cp,
                "compute_loss_iqn_transformer",
            ),
            "rnn": (
                run_iqn_cp.run_rnn_iqn_cp,
                "compute_loss_iqn_rnn",
            ),
        }
        expected_levels = [0.1, 0.2, 0.8, 0.9]

        for architecture, (runner, loss_name) in runners.items():
            with self.subTest(architecture=architecture):
                config = self._runner_config(architecture)
                fake_loss, records, snapshots = self._criterion_loss_spy()
                with (
                    torch.random.fork_rng(devices=[]),
                    patch.object(
                        run_iqn_cp.OmegaConf,
                        "load",
                        return_value=config,
                    ),
                    patch.object(run_iqn_cp.OmegaConf, "save"),
                    patch.object(
                        run_iqn_cp,
                        "load_data",
                        return_value=self._artifact(),
                    ),
                    patch.object(run_iqn_cp, "save_data") as save_data,
                    patch.object(run_iqn_cp.os, "makedirs"),
                    patch.object(run_iqn_cp.torch, "save") as save_model,
                    patch.object(run_iqn_cp, loss_name, side_effect=fake_loss),
                    patch.object(
                        run_iqn_cp, "compute_iqn_interval_validation_loss",
                        side_effect=self._interval_loss_spy(fake_loss),
                    ),
                    patch.object(
                        run_iqn_cp,
                        "tqdm",
                        side_effect=lambda items, **kwargs: items,
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    torch.manual_seed(23)
                    runner("unused_config.yaml")

                logs = [
                    call.args[1]
                    for call in save_data.call_args_list
                    if Path(call.args[0]).name == "log.pkl"
                ]
                self.assertEqual(len(logs), 1)
                sequence_log = logs[0]["series"]
                np.testing.assert_allclose(
                    sequence_log["valid_loss"],
                    [2.0, 7.0 / 3.0],
                )
                self.assertEqual(sequence_log["best_epoch"], 1)
                self.assertAlmostEqual(sequence_log["best_valid_loss"], 2.0)
                self.assertEqual(
                    sequence_log["validation_loss"],
                    "target_quantiles",
                )
                self.assertEqual(
                    sequence_log["validation_quantiles"],
                    expected_levels,
                )

                training_records = [r for r in records if r["training"]]
                validation_records = [r for r in records if not r["training"]]
                self.assertTrue(training_records)
                self.assertTrue(all(r["taus"] is None for r in training_records))
                self.assertEqual(
                    [r["batch_size"] for r in validation_records],
                    [4, 2, 4, 2],
                )
                for record in validation_records:
                    np.testing.assert_allclose(record["taus"], expected_levels)

                self.assertEqual(len(snapshots), 2)
                parameter_name = snapshots[0][0]
                self.assertFalse(torch.equal(snapshots[0][1], snapshots[1][1]))
                saved_state = save_model.call_args.args[0]
                torch.testing.assert_close(
                    saved_state[parameter_name].cpu(),
                    snapshots[0][1],
                )

    @staticmethod
    def _dataset(size):
        zeros = lambda *shape: np.zeros(shape, dtype=np.float32)
        return QuantileRegressionDataset(
            zeros(size, 3, 2),
            zeros(size, 3),
            zeros(size, 3),
            zeros(size, 1, 2),
            zeros(size, 1),
            zeros(size, 1),
            zeros(size, 1),
        )

    def test_tuner_uses_fixed_sample_weighted_validation_for_checkpoint(self):
        sbatch_path = str(Path(__file__).resolve().parents[1] / "sbatch")
        with patch.object(sys, "path", [sbatch_path, *sys.path]):
            tuning = importlib.import_module(
                "sbatch_run_tuning.run_iqn_cp_tuning"
            )

        config = self._runner_config("rnn")
        config.tuning = {"delta_threshold": 0.0}
        sequence_item = {
            "train_dataset": self._dataset(5),
            "model_selection_valid_dataset": self._dataset(6),
            "tuning_evaluation_dataset": self._dataset(4),
        }
        fake_loss, records, snapshots = self._criterion_loss_spy()
        built_models = []
        original_build_model = tuning._build_model

        def capture_built_model(*args, **kwargs):
            built = original_build_model(*args, **kwargs)
            built_models.append(built[0])
            return built

        with (
            torch.random.fork_rng(devices=[]),
            patch.object(
                tuning,
                "compute_loss_iqn_rnn",
                side_effect=fake_loss,
            ),
            patch.object(
                tuning, "compute_iqn_interval_validation_loss",
                side_effect=self._interval_loss_spy(fake_loss),
            ),
            patch.object(
                tuning,
                "_build_model",
                side_effect=capture_built_model,
            ),
        ):
            torch.manual_seed(29)
            result = tuning._run_single_trial(
                config,
                sequence_item,
                normalization_params=None,
            )

        np.testing.assert_allclose(
            result["model_selection_valid_loss"],
            [2.0, 7.0 / 3.0],
        )
        self.assertEqual(result["best_epoch"], 1)
        self.assertAlmostEqual(result["best_valid_loss"], 2.0)
        self.assertEqual(result["validation_loss"], "target_quantiles")
        self.assertEqual(
            result["validation_quantiles"],
            [0.1, 0.2, 0.8, 0.9],
        )

        training_records = [r for r in records if r["training"]]
        validation_records = [r for r in records if not r["training"]]
        self.assertTrue(training_records)
        self.assertTrue(all(r["taus"] is None for r in training_records))
        self.assertEqual(
            [r["batch_size"] for r in validation_records],
            [4, 2, 4, 2],
        )
        for record in validation_records:
            np.testing.assert_allclose(
                record["taus"],
                [0.1, 0.2, 0.8, 0.9],
            )

        self.assertEqual(len(built_models), 1)
        self.assertEqual(len(snapshots), 2)
        parameter_name = snapshots[0][0]
        self.assertFalse(torch.equal(snapshots[0][1], snapshots[1][1]))
        torch.testing.assert_close(
            built_models[0].state_dict()[parameter_name].cpu(),
            snapshots[0][1],
        )

    def test_checked_experiment_configs_default_to_target_validation(self):
        config_dir = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "iqn_cp_configs"
        )
        experiment_configs = [
            path
            for path in config_dir.glob("iqn_*_config.yaml")
            if "_tuning_config" not in path.name
        ]
        self.assertTrue(experiment_configs)
        for config_path in experiment_configs:
            with self.subTest(config=config_path.name):
                config = OmegaConf.load(config_path)
                self.assertEqual(
                    config.training.validation_loss,
                    "target_quantiles",
                )
                self.assertEqual(config.training.tau_mode, "sampled_quantiles")
                self.assertIn(config.model.interval_mode, ("direct", "sampling"))
                self.assertEqual(config.model.sampling_num, 1000)
                self.assertEqual(config.model.iqn_num_layers, 1)

    def test_checked_tuning_grids_default_target_and_both_modes_expand(self):
        sbatch_path = str(Path(__file__).resolve().parents[1] / "sbatch")
        with patch.object(sys, "path", [sbatch_path, *sys.path]):
            common = importlib.import_module("sbatch_run_tuning.common")

        config_dir = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "iqn_cp_configs"
        )
        for architecture in ("rnn", "transformer"):
            with self.subTest(architecture=architecture):
                base_config = OmegaConf.load(
                    config_dir / f"iqn_{architecture}_lr_air_config.yaml"
                )
                grid, _ = common.load_grid(
                    config_dir / f"iqn_{architecture}_air_tuning_config.yaml"
                )
                validation_grid = grid["training.validation_loss"]
                self.assertEqual(
                    validation_grid,
                    ["target_quantiles"],
                )

                resolved = []
                tiny_grid = {
                    "training.validation_loss": [
                        "target_quantiles",
                        "sampled_quantiles",
                    ]
                }
                for trial_config, updates in common.iter_grid_configs(
                    base_config,
                    tiny_grid,
                ):
                    mode, levels = resolve_iqn_validation_quantiles(
                        trial_config.training,
                        trial_config.model.target_quantiles,
                    )
                    resolved.append((mode, levels))
                    self.assertEqual(
                        updates["training.validation_loss"],
                        mode,
                    )

                self.assertEqual(
                    [mode for mode, _ in resolved],
                    ["target_quantiles", "sampled_quantiles"],
                )
                self.assertIsNotNone(resolved[0][1])
                self.assertIsNone(resolved[1][1])


if __name__ == "__main__":
    unittest.main()
