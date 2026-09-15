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
                            self.assertIn(
                                f"iqn_{encoder}_lstm_sapflux_config.yaml"
                                if dataset == "sapflux"
                                else f"iqn_{encoder}_air_config.yaml",
                                output,
                            )
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
