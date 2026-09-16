import contextlib
import importlib
import io
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "qr_cp_configs"
with patch.object(sys, "path", [str(REPO_ROOT / "sbatch"), *sys.path]):
    tuning = importlib.import_module("sbatch_run_tuning.run_qr_cp_tuning")
    array_job = importlib.import_module("sbatch_run_qr_cp.run_qr_cp_job")


class QRTuningPredictorTests(unittest.TestCase):
    def test_all_base_configs_preserve_default_prediction_and_result_paths(self):
        cases = [
            (encoder, predictor, "air", "air-10_prediction", "air-10", "air")
            for encoder in ("rnn", "transformer")
            for predictor in ("lr", "lstm", "chronos")
        ] + [
            (encoder, "lstm", "sapflux", "sapflux-solo3-large",
             "sapflux-solo3-large", "sapflux_solo3_large")
            for encoder in ("rnn", "transformer")
        ]
        for encoder, predictor, dataset, directory, artifact, result_dataset in cases:
            with self.subTest(encoder=encoder, predictor=predictor, dataset=dataset):
                initial = CONFIG_DIR / f"qr_{encoder}_{predictor}_{dataset}_config.yaml"
                config, selected_path, save_dir = tuning.resolve_tuning_inputs(
                    initial, {"num_sequences": "all"},
                    REPO_ROOT / "unused" / f"qr_{encoder}_{{base_predictor}}_{dataset}",
                )
                self.assertEqual(config.base_predictor, predictor)
                self.assertEqual(selected_path, initial)
                self.assertEqual(
                    Path(config.data.data_path),
                    Path("data") / directory / predictor / f"{predictor}_{artifact}_data.pkl",
                )
                self.assertEqual(
                    Path(config.saving_dir),
                    Path("results") / f"qr_{encoder}_{predictor}_{result_dataset}",
                )
                self.assertEqual(
                    save_dir, REPO_ROOT / "unused" / f"qr_{encoder}_{predictor}_{dataset}"
                )

    def test_changing_base_predictor_preserves_custom_settings_and_selected_file(self):
        for encoder in ("rnn", "transformer"):
            for predictor in ("lr", "lstm", "chronos"):
                with self.subTest(encoder=encoder, predictor=predictor):
                    initial = CONFIG_DIR / f"qr_{encoder}_chronos_air_config.yaml"
                    edited_config = OmegaConf.load(initial)
                    edited_config.base_predictor = f" {predictor.upper()} "
                    edited_config.model.dim_model = 128
                    edited_config.training.learning_rate = 0.0123
                    with patch.object(tuning.OmegaConf, "load", return_value=edited_config) as load:
                        config, selected_path, save_dir = tuning.resolve_tuning_inputs(
                            initial, {}, REPO_ROOT / "unused" / "qr_{base_predictor}"
                        )
                    load.assert_called_once_with(initial)
                    self.assertEqual(selected_path, initial)
                    self.assertEqual(config.base_predictor, predictor)
                    self.assertEqual(config.model.dim_model, 128)
                    self.assertEqual(config.training.learning_rate, 0.0123)
                    self.assertEqual(
                        Path(config.data.data_path),
                        Path("data") / "air-10_prediction" / predictor
                        / f"{predictor}_air-10_data.pkl",
                    )
                    self.assertEqual(
                        Path(config.saving_dir), Path("results") / f"qr_{encoder}_{predictor}_air"
                    )
                    self.assertEqual(save_dir, REPO_ROOT / "unused" / f"qr_{predictor}")

    def test_non_air_config_can_select_predictor_without_changing_dataset(self):
        initial = CONFIG_DIR / "qr_rnn_lstm_sapflux_config.yaml"
        edited_config = OmegaConf.load(initial)
        edited_config.base_predictor = "chronos"
        with patch.object(tuning.OmegaConf, "load", return_value=edited_config):
            config, selected_path, _ = tuning.resolve_tuning_inputs(
                initial, {}, REPO_ROOT / "unused"
            )
        self.assertEqual(selected_path, initial)
        self.assertEqual(
            Path(config.data.data_path),
            Path("data/sapflux-solo3-large/chronos/chronos_sapflux-solo3-large_data.pkl"),
        )
        self.assertEqual(
            Path(config.saving_dir), Path("results/qr_rnn_chronos_sapflux_solo3_large")
        )

    def test_legacy_config_preserves_explicit_paths_and_infers_output_predictor(self):
        initial = CONFIG_DIR / "qr_rnn_lstm_sapflux_config.yaml"
        legacy = OmegaConf.to_container(OmegaConf.load(initial), resolve=True)
        del legacy["base_predictor"]
        legacy["data"]["data_path"] = "custom/predictions/lstm_custom_data.pkl"
        for output_name, expected_name in (
            ("custom_result_name", "custom_result_name"),
            ("qr_{base_predictor}", "qr_lstm"),
        ):
            with self.subTest(output_name=output_name):
                with patch.object(tuning.OmegaConf, "load", return_value=OmegaConf.create(legacy)):
                    config, selected_path, save_dir = tuning.resolve_tuning_inputs(
                        initial, {"num_sequences": "all"}, REPO_ROOT / "unused" / output_name
                    )
                self.assertEqual(OmegaConf.to_container(config, resolve=True), legacy)
                self.assertEqual(selected_path, initial)
                self.assertEqual(save_dir, REPO_ROOT / "unused" / expected_name)

    def test_output_without_placeholder_is_unchanged(self):
        output = REPO_ROOT / "unused" / "explicit_output"
        _, _, save_dir = tuning.resolve_tuning_inputs(
            CONFIG_DIR / "qr_rnn_lr_air_config.yaml", {}, output
        )
        self.assertEqual(save_dir, output)

    def test_invalid_base_config_predictors_are_rejected(self):
        initial = CONFIG_DIR / "qr_rnn_chronos_air_config.yaml"
        for predictor in (None, "unknown", "", ["lr"], [], {}, 2):
            with self.subTest(predictor=predictor):
                edited_config = OmegaConf.load(initial)
                edited_config.base_predictor = predictor
                with (
                    patch.object(tuning.OmegaConf, "load", return_value=edited_config),
                    self.assertRaisesRegex(ValueError, "base_predictor"),
                ):
                    tuning.resolve_tuning_inputs(initial, {}, REPO_ROOT / "unused")

    def test_obsolete_tuning_predictor_is_rejected_with_migration_instruction(self):
        for predictor in ("lr", "chronos", None):
            with self.subTest(predictor=predictor):
                with self.assertRaisesRegex(ValueError, "Move tuning.base_predictor.*base.*config"):
                    tuning.resolve_tuning_inputs(
                        CONFIG_DIR / "qr_rnn_chronos_air_config.yaml",
                        {"base_predictor": predictor}, REPO_ROOT / "unused",
                    )

    def test_main_uses_base_predictor_for_artifact_output_and_metadata(self):
        for encoder in ("rnn", "transformer"):
            with self.subTest(encoder=encoder):
                initial = CONFIG_DIR / f"qr_{encoder}_chronos_air_config.yaml"
                edited_config = OmegaConf.load(initial)
                edited_config.base_predictor = "lr"
                edited_config.model.dim_model = 128
                expected_artifact = edited_config.data.data_path
                expected_output = REPO_ROOT / "unused" / f"qr_{encoder}_lr_air"
                args = SimpleNamespace(
                    base_config=initial,
                    grid_config=Path("unused_grid.yaml"),
                    save_dir=REPO_ROOT / "unused" / f"qr_{encoder}_{{base_predictor}}_air",
                    sequence_key=None, sequence_index=0, top_k=3, seed=2026,
                )
                prepared = SimpleNamespace(
                    data={"series": {"train_residuals_mu": 0.0, "train_residuals_std": 1.0}},
                    dataset={"series": {}},
                    prepare_quantile_regression_datasets=Mock(),
                )

                def sequence_result(config, sequence_item, normalization):
                    return {
                        "best_valid_loss": 0.1,
                        "best_epoch": 1,
                        "selection_score": 1.0,
                        "positive_delta_coverage": True,
                        "pair_metrics": {
                            str(tuple(pair)): {
                                "avg_coverage": 0.95,
                                "target_coverage": max(pair) - min(pair),
                                "avg_delta_coverage": 0.05,
                                "avg_interval_width": 1.0,
                                "avg_winkler_score": 1.0,
                            }
                            for pair in config.model.target_quantiles
                        },
                    }

                with (
                    patch.dict(os.environ, {"QR_BASE_PREDICTOR": "chronos"}),
                    patch.object(tuning, "parse_args", return_value=args),
                    patch.object(tuning.OmegaConf, "load", return_value=edited_config),
                    patch.object(
                        tuning, "load_grid",
                        return_value=(
                            {"training.learning_rate": [0.001]}, {"num_sequences": "all"}
                        ),
                    ),
                    patch.object(tuning, "load_data", return_value={"series": {}}) as load_data,
                    patch.object(tuning, "ConformalPredictionData", return_value=prepared),
                    patch.object(tuning, "set_global_seed"),
                    patch.object(tuning, "_run_single_trial", side_effect=sequence_result) as train,
                    patch.object(Path, "mkdir"),
                    patch.object(tuning, "write_trial_artifacts") as write_trial,
                    patch.object(tuning, "finalize_and_save_results") as finalize,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    tuning.main()

                load_data.assert_called_once_with(expected_artifact)
                train.assert_called_once()
                self.assertEqual(train.call_args.args[0].model.dim_model, 128)
                write_trial.assert_called_once()
                self.assertEqual(write_trial.call_args.args[0], expected_output)
                finalize.assert_called_once()
                self.assertEqual(finalize.call_args.args[0], expected_output)
                payload = finalize.call_args.args[1]
                self.assertEqual(Path(payload["base_config_path"]), initial)
                resolved = payload["all_trials"][0]["resolved_config"]
                self.assertEqual(resolved["base_predictor"], "lr")
                self.assertEqual(resolved["data"]["data_path"], expected_artifact)
                self.assertEqual(resolved["model"]["dim_model"], 128)

    def test_array_dry_run_updates_predictor_metadata_for_sapflux_templates(self):
        for task_id, (encoder, predictor) in enumerate(array_job.TASKS):
            with self.subTest(task_id=task_id):
                stdout = io.StringIO()
                with (
                    contextlib.redirect_stdout(stdout),
                    patch.object(Path, "mkdir") as mkdir,
                    patch.object(OmegaConf, "save") as save,
                ):
                    array_job.main(["sapflux", "--task-id", str(task_id), "--dry-run"])
                mkdir.assert_not_called()
                save.assert_not_called()
                config = OmegaConf.create("\n".join(stdout.getvalue().splitlines()[2:]))
                self.assertEqual(config.base_predictor, predictor)
                self.assertEqual(
                    Path(config.data.data_path),
                    REPO_ROOT / "data" / "sapflux-solo3-large" / predictor
                    / f"{predictor}_sapflux-solo3-large_data.pkl",
                )
                self.assertEqual(
                    Path(config.saving_dir),
                    REPO_ROOT / "results" / "qr_cp" / "sapflux" / predictor / encoder
                    / "nondecreasing",
                )


if __name__ == "__main__":
    unittest.main()
