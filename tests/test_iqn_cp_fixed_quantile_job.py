import contextlib
import importlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from utils.experiment_config import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXED_CONFIG_DIR = REPO_ROOT / "configs" / "iqn_cp_fixed_quantile_configs"


class IQNCPFixedQuantileJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "path", [str(REPO_ROOT / "sbatch"), *sys.path]):
            cls.job = importlib.import_module("sbatch_run_iqn_cp.run_iqn_cp_job")

    def _dry_run(self, arguments):
        stream = io.StringIO()
        with (
            patch.object(Path, "mkdir") as mkdir,
            patch.object(OmegaConf, "save") as save,
            contextlib.redirect_stdout(stream),
        ):
            self.job.main([*arguments, "--dry-run"])
        mkdir.assert_not_called()
        save.assert_not_called()
        output = stream.getvalue()
        return output, OmegaConf.create(output.split("\n", 2)[2])

    def test_all_18_fixed_templates_preserve_settings_without_writes(self):
        checked = 0
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "outputs"
            for dataset in self.job.DATASET_ARTIFACTS:
                for task_id, (encoder, predictor) in enumerate(self.job.TASKS):
                    with self.subTest(dataset=dataset, encoder=encoder, predictor=predictor):
                        filename = (
                            f"iqn_{encoder}_{predictor}_{dataset}_fixed_quantile_config.yaml"
                        )
                        template = load_experiment_config(FIXED_CONFIG_DIR / filename)
                        output, config = self._dry_run(
                            [
                                dataset, "--task-id", str(task_id),
                                "--fixed-quantiles", "--seed", "17",
                                "--output-root", str(output_root),
                            ]
                        )
                        self.assertIn(str(FIXED_CONFIG_DIR / filename), output)
                        for section in ("model", "training", "plotting"):
                            self.assertEqual(
                                OmegaConf.to_container(config[section], resolve=True),
                                OmegaConf.to_container(template[section], resolve=True),
                            )
                        self.assertEqual(config.training.tau_mode, "target_quantiles")
                        self.assertEqual(config.training.validation_loss, "target_quantiles")
                        self.assertEqual(config.model.interval_mode, "direct")
                        self.assertEqual(
                            OmegaConf.to_container(config.model.target_quantiles),
                            [[0.95, 0.05]],
                        )
                        self.assertNotIn("num_taus", config.model)
                        expected_data = OmegaConf.to_container(template.data, resolve=True)
                        artifact_dir, artifact_name = self.job.DATASET_ARTIFACTS[dataset]
                        expected_data["data_path"] = str(
                            REPO_ROOT / "data" / artifact_dir / predictor
                            / f"{predictor}_{artifact_name}_data.pkl"
                        )
                        self.assertEqual(
                            OmegaConf.to_container(config.data, resolve=True), expected_data
                        )
                        self.assertEqual(config.base_predictor, predictor)
                        self.assertEqual(config.device, template.device)
                        self.assertEqual(config.seed, 17)
                        relative_output = Path(config.saving_dir).relative_to(
                            output_root.resolve()
                        )
                        self.assertEqual(
                            relative_output.parts[:4],
                            (dataset, predictor, encoder, config.model.prediction_head),
                        )
                        self.assertEqual(len(relative_output.parts), 5)
                        self.assertTrue(relative_output.name)
                        checked += 1
            self.assertFalse(output_root.exists())
        self.assertEqual(checked, 18)

    def test_default_roots_separate_modes_and_explicit_root_is_honored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            explicit_root = Path(temp_dir) / "chosen-output"
            for fixed in (False, True):
                for explicit in (False, True):
                    with self.subTest(fixed=fixed, explicit=explicit):
                        arguments = ["air", "--task-id", "0"]
                        if fixed:
                            arguments.append("--fixed-quantiles")
                        if explicit:
                            arguments.extend(["--output-root", str(explicit_root)])
                        output, config = self._dry_run(arguments)
                        expected_root = (
                            explicit_root if explicit else REPO_ROOT / "results"
                            / ("iqn_cp_fixed_quantile" if fixed else "iqn_cp")
                        ).resolve()
                        relative_output = Path(config.saving_dir).relative_to(expected_root)
                        expected_head = "partially_monotonic" if fixed else "cosine_embedding"
                        self.assertEqual(
                            relative_output.parts[:4],
                            ("air", "lr", "rnn", expected_head),
                        )
                        self.assertEqual(
                            config.training.tau_mode,
                            "target_quantiles" if fixed else "sampled_quantiles",
                        )
                        expected_name = (
                            "iqn_rnn_lr_air_fixed_quantile_config.yaml" if fixed
                            else "iqn_rnn_lr_air_config.yaml"
                        )
                        self.assertIn(expected_name, output)
            self.assertFalse(explicit_root.exists())

    def test_fixed_dispatch_saves_resolved_config_and_seeds_selected_encoder(self):
        from dscp import run_iqn_cp

        cases = (
            ("air", 0, None, False),
            ("solar", 4, "cosine_embedding", True),
            ("sapflux", 5, None, True),
        )
        for dataset, task_id, head_override, explicit_root in cases:
            with self.subTest(dataset=dataset, task_id=task_id, head=head_override):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    encoder, predictor = self.job.TASKS[task_id]
                    filename = (
                        f"iqn_{encoder}_{predictor}_{dataset}_fixed_quantile_config.yaml"
                    )
                    template = load_experiment_config(FIXED_CONFIG_DIR / filename)
                    template.saving_dir = (
                        f"./results/iqn_cp/{dataset}/${{base_predictor}}/{encoder}/"
                        "${model.prediction_head}/fixture-run/"
                    )
                    template_dir = root / "configs" / "iqn_cp_fixed_quantile_configs"
                    template_dir.mkdir(parents=True)
                    OmegaConf.save(template, template_dir / filename)
                    artifact_dir, artifact_name = self.job.DATASET_ARTIFACTS[dataset]
                    artifact = (
                        root / "data" / artifact_dir / predictor
                        / f"{predictor}_{artifact_name}_data.pkl"
                    )
                    artifact.parent.mkdir(parents=True)
                    artifact.touch()
                    arguments = [
                        dataset, "--task-id", str(task_id),
                        "--fixed-quantiles", "--seed", "29",
                    ]
                    output_root = root / "results" / "iqn_cp_fixed_quantile"
                    if explicit_root:
                        output_root = root / "custom-output"
                        arguments.extend(["--output-root", str(output_root)])
                    selected_head = head_override or template.model.prediction_head
                    if head_override:
                        arguments.extend(["--prediction-head", head_override])
                    with (
                        patch.object(self.job, "REPO_ROOT", root),
                        patch.object(run_iqn_cp, "run_rnn_iqn_cp") as rnn_runner,
                        patch.object(run_iqn_cp, "run_transformer_iqn_cp") as transformer_runner,
                        patch("random.seed") as random_seed,
                        patch("numpy.random.seed") as numpy_seed,
                        patch("torch.manual_seed") as torch_seed,
                        patch("torch.cuda.is_available", return_value=True),
                        patch("torch.cuda.manual_seed_all") as cuda_seed,
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        self.job.main(arguments)
                    expected_dir = (
                        output_root / dataset / predictor / encoder
                        / selected_head / "fixture-run"
                    ).resolve()
                    saved_path = expected_dir / "resolved_config.yaml"
                    self.assertTrue(saved_path.is_file())
                    saved = OmegaConf.load(saved_path)
                    self.assertEqual(Path(saved.saving_dir), expected_dir)
                    self.assertEqual(Path(saved.data.data_path), artifact)
                    self.assertEqual(saved.seed, 29)
                    self.assertEqual(saved.model.prediction_head, selected_head)
                    self.assertEqual(saved.training.tau_mode, "target_quantiles")
                    self.assertEqual(saved.training.validation_loss, "target_quantiles")
                    self.assertNotIn("num_taus", saved.model)
                    expected_model = OmegaConf.to_container(template.model, resolve=True)
                    expected_model["prediction_head"] = selected_head
                    self.assertEqual(
                        OmegaConf.to_container(saved.model, resolve=True), expected_model
                    )
                    for seed_mock in (random_seed, numpy_seed, torch_seed, cuda_seed):
                        seed_mock.assert_called_once_with(29)
                    selected_runner = rnn_runner if encoder == "rnn" else transformer_runner
                    other_runner = transformer_runner if encoder == "rnn" else rnn_runner
                    selected_runner.assert_called_once_with(str(saved_path))
                    other_runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
