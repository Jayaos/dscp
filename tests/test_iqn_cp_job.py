import contextlib
import importlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_job_module():
    sbatch_path = str(REPO_ROOT / "sbatch")
    with patch.object(sys, "path", [sbatch_path, *sys.path]):
        return importlib.import_module("sbatch_run_iqn_cp.run_iqn_cp_job")


class IQNCPJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.job = _load_job_module()

    def test_task_and_artifact_mappings_match_qr_cp_arrays(self):
        self.assertEqual(
            self.job.TASKS,
            (
                ("rnn", "lr"),
                ("rnn", "lstm"),
                ("rnn", "chronos"),
                ("transformer", "lr"),
                ("transformer", "lstm"),
                ("transformer", "chronos"),
            ),
        )
        self.assertEqual(
            self.job.DATASET_ARTIFACTS,
            {
                "air": ("air-10_prediction", "air-10"),
                "solar": ("solar_prediction", "nsdb-60m"),
                "sapflux": (
                    "sapflux-solo3-large",
                    "sapflux-solo3-large",
                ),
            },
        )

    def test_every_dry_run_resolves_without_artifacts_or_writes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "outputs"
            for dataset in self.job.DATASET_ARTIFACTS:
                for task_id, (encoder, predictor) in enumerate(self.job.TASKS):
                    for prediction_head in self.job.PREDICTION_HEADS:
                        with self.subTest(
                            dataset=dataset,
                            task_id=task_id,
                            prediction_head=prediction_head,
                        ):
                            stream = io.StringIO()
                            with contextlib.redirect_stdout(stream):
                                self.job.main(
                                    [
                                        dataset,
                                        "--task-id",
                                        str(task_id),
                                        "--prediction-head",
                                        prediction_head,
                                        "--output-root",
                                        str(output_root),
                                        "--seed",
                                        "17",
                                        "--dry-run",
                                    ]
                                )

                            output = stream.getvalue()
                            self.assertIn(f"dataset={dataset}", output)
                            self.assertIn(f"base_predictor={predictor}", output)
                            self.assertIn(f"encoder={encoder}", output)
                            self.assertIn(
                                f"prediction_head={prediction_head}",
                                output,
                            )
                            self.assertIn("seed=17", output)
                            expected_template = (
                                f"iqn_{encoder}_{predictor}_{dataset}_config.yaml"
                            )
                            self.assertIn(expected_template, output)
                            _, artifact_name = self.job.DATASET_ARTIFACTS[dataset]
                            self.assertIn(
                                f"{predictor}_{artifact_name}_data.pkl",
                                output,
                            )
                            expected_output = (
                                output_root.resolve()
                                / dataset
                                / predictor
                                / encoder
                                / prediction_head
                            )
                            self.assertIn(str(expected_output), output)

            self.assertFalse(output_root.exists())

    def test_each_dataset_uses_predictor_specific_configs_for_both_heads(self):
        from dscp import run_iqn_cp

        with tempfile.TemporaryDirectory() as temp_dir:
            fixture_root = Path(temp_dir)
            config_dir = fixture_root / "configs" / "iqn_cp_configs"
            config_dir.mkdir(parents=True)
            sentinels = {}

            for dataset_index, dataset in enumerate(
                ("air", "solar", "sapflux")
            ):
                for task_id, (encoder, predictor) in enumerate(self.job.TASKS):
                    sentinel = {
                        "template_id": f"{dataset}-{encoder}-{predictor}",
                        "dim_model": 20 + task_id + 100 * dataset_index,
                        "window_size": 100 + task_id + 1000 * dataset_index,
                        "learning_rate": (
                            0.001 * (task_id + 1 + 10 * dataset_index)
                        ),
                    }
                    sentinels[(dataset, encoder, predictor)] = sentinel
                    fixture_config = OmegaConf.create(
                        {
                            "device": "cpu",
                            "template_id": sentinel["template_id"],
                            "data": {"data_path": "template-artifact.pkl"},
                            "model": {
                                "prediction_head": "template-head",
                                "prediction_step": 9,
                                "dim_model": sentinel["dim_model"],
                                "window_size": sentinel["window_size"],
                            },
                            "training": {
                                "learning_rate": sentinel["learning_rate"],
                            },
                            "saving_dir": "template-output",
                        }
                    )
                    OmegaConf.save(
                        config=fixture_config,
                        f=(
                            config_dir
                            / (
                                f"iqn_{encoder}_{predictor}_"
                                f"{dataset}_config.yaml"
                            )
                        ),
                    )

            # A successful dispatch cannot accidentally rely on a shared
            # dataset template because this fixture deliberately omits them.
            for dataset in ("air", "solar", "sapflux"):
                self.assertFalse(
                    (config_dir / f"iqn_rnn_{dataset}_config.yaml").exists()
                )
                self.assertFalse(
                    (
                        config_dir
                        / f"iqn_transformer_{dataset}_config.yaml"
                    ).exists()
                )

            for dataset in ("air", "solar", "sapflux"):
                artifact_dir, artifact_name = self.job.DATASET_ARTIFACTS[dataset]
                for task_id, (encoder, predictor) in enumerate(self.job.TASKS):
                    for prediction_head in self.job.PREDICTION_HEADS:
                        with self.subTest(
                            dataset=dataset,
                            task_id=task_id,
                            encoder=encoder,
                            predictor=predictor,
                            prediction_head=prediction_head,
                        ):
                            rnn_runner = unittest.mock.Mock()
                            transformer_runner = unittest.mock.Mock()
                            with (
                                patch.object(self.job, "REPO_ROOT", fixture_root),
                                patch.object(Path, "is_file", return_value=True),
                                patch.object(Path, "mkdir"),
                                patch.object(OmegaConf, "save") as save_config,
                                patch.object(
                                    run_iqn_cp,
                                    "run_rnn_iqn_cp",
                                    rnn_runner,
                                ),
                                patch.object(
                                    run_iqn_cp,
                                    "run_transformer_iqn_cp",
                                    transformer_runner,
                                ),
                                patch("random.seed"),
                                patch("numpy.random.seed"),
                                patch("torch.manual_seed"),
                                patch(
                                    "torch.cuda.is_available",
                                    return_value=False,
                                ),
                                contextlib.redirect_stdout(io.StringIO()),
                            ):
                                self.job.main(
                                    [
                                        dataset,
                                        "--task-id",
                                        str(task_id),
                                        "--prediction-head",
                                        prediction_head,
                                        "--output-root",
                                        str(fixture_root / "outputs"),
                                        "--seed",
                                        "31",
                                    ]
                                )

                            save_config.assert_called_once()
                            saved_config = save_config.call_args.kwargs["config"]
                            sentinel = sentinels[(dataset, encoder, predictor)]
                            self.assertEqual(
                                saved_config.template_id,
                                sentinel["template_id"],
                            )
                            self.assertEqual(
                                saved_config.model.dim_model,
                                sentinel["dim_model"],
                            )
                            self.assertEqual(
                                saved_config.model.window_size,
                                sentinel["window_size"],
                            )
                            self.assertAlmostEqual(
                                saved_config.training.learning_rate,
                                sentinel["learning_rate"],
                            )
                            self.assertEqual(
                                saved_config.model.prediction_head,
                                prediction_head,
                            )
                            self.assertEqual(saved_config.model.prediction_step, 1)
                            self.assertEqual(
                                Path(saved_config.data.data_path),
                                fixture_root
                                / "data"
                                / artifact_dir
                                / predictor
                                / f"{predictor}_{artifact_name}_data.pkl",
                            )
                            self.assertEqual(
                                Path(saved_config.saving_dir),
                                (
                                    fixture_root
                                    / "outputs"
                                    / dataset
                                    / predictor
                                    / encoder
                                    / prediction_head
                                ).resolve(),
                            )

                            selected_runner = (
                                rnn_runner
                                if encoder == "rnn"
                                else transformer_runner
                            )
                            other_runner = (
                                transformer_runner
                                if encoder == "rnn"
                                else rnn_runner
                            )
                            selected_runner.assert_called_once()
                            other_runner.assert_not_called()

    def test_real_solar_configs_match_air_hyperparameters_and_paths(self):
        config_dir = REPO_ROOT / "configs" / "iqn_cp_configs"

        for encoder, predictor in self.job.TASKS:
            with self.subTest(encoder=encoder, predictor=predictor):
                air = OmegaConf.load(
                    config_dir
                    / f"iqn_{encoder}_{predictor}_air_config.yaml"
                )
                solar = OmegaConf.load(
                    config_dir
                    / f"iqn_{encoder}_{predictor}_solar_config.yaml"
                )

                air_values = OmegaConf.to_container(air, resolve=True)
                solar_values = OmegaConf.to_container(solar, resolve=True)
                air_values["data"].pop("data_path")
                solar_values["data"].pop("data_path")
                air_values.pop("saving_dir")
                solar_values.pop("saving_dir")
                self.assertEqual(solar_values, air_values)

                self.assertEqual(
                    solar.data.data_path,
                    (
                        f"./data/solar_prediction/{predictor}/"
                        f"{predictor}_nsdb-60m_data.pkl"
                    ),
                )
                output_prefix = (
                    "iqn_rnn"
                    if encoder == "rnn"
                    else "run_iqn_transformer"
                )
                self.assertEqual(
                    solar.saving_dir,
                    (
                        f"./results/{output_prefix}_{predictor}_solar_"
                        "partially_monotonic/"
                    ),
                )

    def test_predictor_specific_sapflux_configs_match_lstm_hyperparameters(self):
        config_dir = REPO_ROOT / "configs" / "iqn_cp_configs"

        for encoder in ("rnn", "transformer"):
            reference = OmegaConf.load(
                config_dir / f"iqn_{encoder}_lstm_sapflux_config.yaml"
            )
            reference_values = OmegaConf.to_container(reference, resolve=True)
            reference_values["data"].pop("data_path")
            reference_values.pop("saving_dir")

            for predictor in ("lr", "chronos"):
                with self.subTest(encoder=encoder, predictor=predictor):
                    config = OmegaConf.load(
                        config_dir
                        / f"iqn_{encoder}_{predictor}_sapflux_config.yaml"
                    )
                    values = OmegaConf.to_container(config, resolve=True)
                    values["data"].pop("data_path")
                    values.pop("saving_dir")
                    self.assertEqual(values, reference_values)

                    self.assertEqual(
                        config.data.data_path,
                        (
                            f"./data/sapflux-solo3-large/{predictor}/"
                            f"{predictor}_sapflux-solo3-large_data.pkl"
                        ),
                    )
                    self.assertEqual(
                        config.saving_dir,
                        (
                            f"./results/iqn_{encoder}_{predictor}_"
                            "sapflux_solo3_large_partially_monotonic/"
                        ),
                    )

    def test_non_dry_run_saves_config_seeds_and_dispatches_encoder(self):
        from dscp import run_iqn_cp

        cases = (
            (0, "partially_monotonic", "run_rnn_iqn_cp"),
            (5, "cosine_embedding", "run_transformer_iqn_cp"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for task_id, prediction_head, expected_runner in cases:
                with self.subTest(
                    task_id=task_id,
                    prediction_head=prediction_head,
                ):
                    rnn_runner = unittest.mock.Mock()
                    transformer_runner = unittest.mock.Mock()
                    with (
                        patch.object(Path, "is_file", return_value=True),
                        patch.object(Path, "mkdir") as mkdir,
                        patch.object(OmegaConf, "save") as save_config,
                        patch.object(run_iqn_cp, "run_rnn_iqn_cp", rnn_runner),
                        patch.object(
                            run_iqn_cp,
                            "run_transformer_iqn_cp",
                            transformer_runner,
                        ),
                        patch("random.seed") as random_seed,
                        patch("numpy.random.seed") as numpy_seed,
                        patch("torch.manual_seed") as torch_seed,
                        patch("torch.cuda.is_available", return_value=False),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        self.job.main(
                            [
                                "air",
                                "--task-id",
                                str(task_id),
                                "--prediction-head",
                                prediction_head,
                                "--output-root",
                                temp_dir,
                                "--seed",
                                "29",
                            ]
                        )

                    mkdir.assert_called_once_with(parents=True, exist_ok=True)
                    save_config.assert_called_once()
                    saved_config = save_config.call_args.kwargs["config"]
                    self.assertEqual(
                        saved_config.model.prediction_head,
                        prediction_head,
                    )
                    self.assertEqual(saved_config.model.prediction_step, 1)
                    self.assertEqual(saved_config.seed, 29)
                    random_seed.assert_called_once_with(29)
                    numpy_seed.assert_called_once_with(29)
                    torch_seed.assert_called_once_with(29)

                    selected_runner = (
                        rnn_runner
                        if expected_runner == "run_rnn_iqn_cp"
                        else transformer_runner
                    )
                    other_runner = (
                        transformer_runner
                        if selected_runner is rnn_runner
                        else rnn_runner
                    )
                    selected_runner.assert_called_once()
                    other_runner.assert_not_called()
                    self.assertEqual(
                        Path(selected_runner.call_args.args[0]).name,
                        "resolved_config.yaml",
                    )

    def test_sbatch_files_use_iqn_names_and_environment(self):
        directory = REPO_ROOT / "sbatch" / "sbatch_run_iqn_cp"
        for dataset in self.job.DATASET_ARTIFACTS:
            with self.subTest(dataset=dataset):
                content = (directory / f"run_iqn_cp_{dataset}.sbatch").read_text(
                    encoding="utf-8"
                )
                self.assertIn("#SBATCH --array=0-5", content)
                self.assertIn("#SBATCH --time=04:00:00", content)
                self.assertIn(
                    f"sbatch_run_iqn_cp.run_iqn_cp_job {dataset}",
                    content,
                )
                for variable in (
                    "IQN_PREDICTION_HEAD",
                    "IQN_OUTPUT_ROOT",
                    "IQN_SEED",
                ):
                    self.assertIn(variable, content)
                self.assertNotIn("QR_HEAD_TYPE", content)
                self.assertNotIn("sbatch_run_qr_cp", content)


if __name__ == "__main__":
    unittest.main()
