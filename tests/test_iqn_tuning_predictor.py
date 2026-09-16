import contextlib
import importlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "iqn_cp_configs"
with patch.object(sys, "path", [str(REPO_ROOT / "sbatch"), *sys.path]):
    tuning = importlib.import_module("sbatch_run_tuning.run_iqn_cp_tuning")
    array_job = importlib.import_module("sbatch_run_iqn_cp.run_iqn_cp_job")


class IQNTuningPredictorTests(unittest.TestCase):
    def test_air_base_configs_resolve_each_predictor_and_output(self):
        for encoder in ("rnn", "transformer"):
            for predictor in ("lr", "lstm", "chronos"):
                with self.subTest(encoder=encoder, predictor=predictor):
                    initial = CONFIG_DIR / f"iqn_{encoder}_{predictor}_air_config.yaml"
                    config, selected_path, output = tuning.resolve_tuning_inputs(
                        initial, {},
                        REPO_ROOT / "unused" / f"iqn_{encoder}_{{base_predictor}}_air",
                    )
                    self.assertEqual(selected_path, initial)
                    self.assertEqual(config.base_predictor, predictor)
                    self.assertEqual(
                        Path(config.data.data_path),
                        Path("data/air-10_prediction") / predictor
                        / f"{predictor}_air-10_data.pkl",
                    )
                    prefix = "iqn_rnn" if encoder == "rnn" else "run_iqn_transformer"
                    self.assertEqual(
                        Path(config.saving_dir),
                        Path("results") / f"{prefix}_{predictor}_air_partially_monotonic",
                    )
                    self.assertEqual(
                        output, REPO_ROOT / "unused" / f"iqn_{encoder}_{predictor}_air"
                    )

    def test_generic_air_configs_keep_chronos_defaults(self):
        for encoder in ("rnn", "transformer"):
            with self.subTest(encoder=encoder):
                generic = OmegaConf.load(CONFIG_DIR / f"iqn_{encoder}_air_config.yaml")
                explicit = OmegaConf.load(
                    CONFIG_DIR / f"iqn_{encoder}_chronos_air_config.yaml"
                )
                self.assertEqual(
                    OmegaConf.to_container(generic, resolve=True),
                    OmegaConf.to_container(explicit, resolve=True),
                )

    def test_main_dispatch_preserves_selected_file_and_custom_head_settings(self):
        for encoder in ("rnn", "transformer"):
            for predictor in ("lr", "lstm", "chronos"):
                for head in ("partially_monotonic", "cosine_embedding"):
                    with self.subTest(encoder=encoder, predictor=predictor, head=head):
                        initial = CONFIG_DIR / f"iqn_{encoder}_chronos_air_config.yaml"
                        config = OmegaConf.load(initial)
                        config.base_predictor = f" {predictor.upper()} "
                        config.model.shared_dim = 128
                        config.model.prediction_head = head
                        config.model.monotonic_num_layers = 3
                        config.training.learning_rate = 0.0123
                        artifact = (
                            f"./data/air-10_prediction/{predictor}/{predictor}_air-10_data.pkl"
                        )
                        expected_output = (
                            REPO_ROOT / "unused" / f"iqn_{encoder}_{predictor}_air"
                        )
                        args = SimpleNamespace(
                            base_config=initial, grid_config=Path("unused_grid.yaml"),
                            save_dir=(
                                REPO_ROOT / "unused" / f"iqn_{encoder}_{{base_predictor}}_air"
                            ),
                            sequence_key=None, sequence_index=0, top_k=3, seed=2026,
                            num_gpus=1,
                        )
                        prepared = SimpleNamespace(
                            data={"series": {
                                "train_residuals_mu": 0.0, "train_residuals_std": 1.0
                            }},
                            dataset={"series": {}},
                        )

                        def sequence_result(trial_config, sequence_item, normalization):
                            return {
                                "prediction_head": trial_config.model.prediction_head,
                                "best_valid_loss": 0.1, "best_epoch": 1,
                                "selection_score": 1.0, "positive_delta_coverage": True,
                                "pair_metrics": {
                                    str(tuple(pair)): {
                                        "avg_coverage": 0.95,
                                        "target_coverage": max(pair) - min(pair),
                                        "avg_delta_coverage": 0.05,
                                        "avg_interval_width": 1.0, "avg_winkler_score": 1.0,
                                    }
                                    for pair in trial_config.model.target_quantiles
                                },
                            }

                        with (
                            patch.object(tuning, "parse_args", return_value=args),
                            patch.object(tuning.OmegaConf, "load", return_value=config) as load,
                            patch.object(tuning, "load_grid", return_value=(
                                {"model.window_size": [25]}, {"num_sequences": "all"}
                            )),
                            patch.object(tuning, "load_data", return_value={"series": {}}) as data,
                            patch.object(tuning, "_prepare_trial_data", return_value=prepared),
                            patch.object(tuning, "set_global_seed"),
                            patch.object(
                                tuning, "_run_single_trial", side_effect=sequence_result
                            ) as train,
                            patch.object(Path, "mkdir"),
                            patch.object(tuning, "write_trial_artifacts") as write_trial,
                            patch.object(tuning, "finalize_and_save_results") as finalize,
                            contextlib.redirect_stdout(io.StringIO()),
                        ):
                            tuning.main()

                        load.assert_called_once_with(initial)
                        data.assert_called_once_with(artifact)
                        train.assert_called_once()
                        trial_config = train.call_args.args[0]
                        for dimension in ("shared_dim", "dim_model", "iqn_hidden_dim", "cos_emb_dim"):
                            self.assertEqual(trial_config.model[dimension], 128)
                        self.assertEqual(trial_config.model.prediction_head, head)
                        self.assertEqual(trial_config.model.monotonic_num_layers, 3)
                        self.assertEqual(trial_config.training.learning_rate, 0.0123)
                        self.assertEqual(trial_config.model.window_size, 25)
                        write_trial.assert_called_once()
                        self.assertEqual(write_trial.call_args.args[0], expected_output)
                        finalize.assert_called_once()
                        self.assertEqual(finalize.call_args.args[0], expected_output)
                        payload = finalize.call_args.args[1]
                        self.assertEqual(Path(payload["base_config_path"]), initial)
                        self.assertFalse(payload["final_test_evaluated"])
                        resolved = payload["all_trials"][0]["resolved_config"]
                        self.assertEqual(resolved["base_predictor"], predictor)
                        self.assertEqual(resolved["data"]["data_path"], artifact)
                        self.assertEqual(resolved["model"]["prediction_head"], head)
                        self.assertEqual(
                            payload["all_trials"][0]["result"]["prediction_head"], head
                        )

    def test_legacy_explicit_paths_are_preserved_and_output_predictor_is_inferred(self):
        initial = CONFIG_DIR / "iqn_rnn_chronos_air_config.yaml"
        legacy = OmegaConf.to_container(OmegaConf.load(initial), resolve=True)
        del legacy["base_predictor"]
        legacy["data"]["data_path"] = "custom/predictions/lstm_custom_data.pkl"
        legacy["saving_dir"] = "custom/results"
        for output_name, expected_name in (
            ("explicit_output", "explicit_output"),
            ("iqn_{base_predictor}", "iqn_lstm"),
        ):
            with self.subTest(output_name=output_name):
                with patch.object(tuning.OmegaConf, "load", return_value=OmegaConf.create(legacy)):
                    config, selected_path, output = tuning.resolve_tuning_inputs(
                        initial, {}, REPO_ROOT / "unused" / output_name
                    )
                self.assertEqual(OmegaConf.to_container(config, resolve=True), legacy)
                self.assertEqual(selected_path, initial)
                self.assertEqual(output, REPO_ROOT / "unused" / expected_name)

    def test_invalid_predictors_and_obsolete_tuning_field_are_rejected(self):
        initial = CONFIG_DIR / "iqn_rnn_chronos_air_config.yaml"
        for predictor in (None, "unknown", "", ["lr"], {}, 2):
            with self.subTest(predictor=predictor):
                config = OmegaConf.load(initial)
                config.base_predictor = predictor
                with (
                    patch.object(tuning.OmegaConf, "load", return_value=config),
                    self.assertRaisesRegex(ValueError, "base_predictor"),
                ):
                    tuning.resolve_tuning_inputs(initial, {}, REPO_ROOT / "unused")
        with self.assertRaisesRegex(ValueError, "Move tuning.base_predictor.*base.*config"):
            tuning.resolve_tuning_inputs(
                initial, {"base_predictor": "lr"}, REPO_ROOT / "unused"
            )

    def test_ordinary_array_dry_run_keeps_predictor_metadata_and_paths_consistent(self):
        for dataset, (directory, artifact_name) in array_job.DATASET_ARTIFACTS.items():
            for task_id, (encoder, predictor) in enumerate(array_job.TASKS):
                with self.subTest(dataset=dataset, encoder=encoder, predictor=predictor):
                    stdout = io.StringIO()
                    with (
                        contextlib.redirect_stdout(stdout),
                        patch.object(Path, "mkdir") as mkdir,
                        patch.object(OmegaConf, "save") as save,
                    ):
                        array_job.main([dataset, "--task-id", str(task_id), "--dry-run"])
                    mkdir.assert_not_called()
                    save.assert_not_called()
                    config = OmegaConf.create("\n".join(stdout.getvalue().splitlines()[2:]))
                    self.assertEqual(config.base_predictor, predictor)
                    self.assertEqual(
                        Path(config.data.data_path),
                        REPO_ROOT / "data" / directory / predictor
                        / f"{predictor}_{artifact_name}_data.pkl",
                    )
                    self.assertEqual(
                        Path(config.saving_dir),
                        REPO_ROOT / "results/iqn_cp" / dataset / predictor / encoder
                        / "partially_monotonic",
                    )


if __name__ == "__main__":
    unittest.main()
