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
MISSING_SELECTOR = object()


class IQNCPHeadSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "path", [str(REPO_ROOT / "sbatch"), *sys.path]):
            cls.job = importlib.import_module("sbatch_run_iqn_cp.run_iqn_cp_job")

    def _write_config(self, root, selector=MISSING_SELECTOR, task_id=0):
        encoder, predictor = self.job.TASKS[task_id]
        config = OmegaConf.create(
            {
                "device": "cpu",
                "base_predictor": predictor,
                "data": {"data_path": "unused.pkl"},
                "model": {"prediction_step": 1, "interval_mode": "direct"},
                "saving_dir": (
                    f"./results/iqn_cp/air/${{base_predictor}}/{encoder}/"
                    "${model.prediction_head}/fixture-run/"
                ),
            }
        )
        if selector is not MISSING_SELECTOR:
            config.model.prediction_head = selector
        config_dir = root / "configs" / "iqn_cp_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            config,
            config_dir / f"iqn_{encoder}_{predictor}_air_config.yaml",
        )

    def _dry_run(self, root, *extra_args, task_id=0):
        output_root = root / "outputs"
        stream = io.StringIO()
        with (
            patch.object(self.job, "REPO_ROOT", root),
            contextlib.redirect_stdout(stream),
        ):
            self.job.main(
                [
                    "air",
                    "--task-id",
                    str(task_id),
                    "--output-root",
                    str(output_root),
                    "--dry-run",
                    *extra_args,
                ]
            )
        self.assertFalse(output_root.exists())
        output = stream.getvalue()
        config = OmegaConf.create(output.split("\n", 2)[2])
        return output, config

    def test_yaml_selector_is_used_and_normalized_without_override(self):
        for selected in self.job.PREDICTION_HEADS:
            for raw_selector in (selected, f"  {selected.upper()}  "):
                with self.subTest(selector=raw_selector):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        root = Path(temp_dir)
                        self._write_config(root, raw_selector)
                        output, config = self._dry_run(root)
                        self.assertEqual(config.model.prediction_head, selected)
                        self.assertIn(f"prediction_head={selected},", output)
                        self.assertEqual(
                            Path(config.saving_dir),
                            (
                                root / "outputs" / "air" / "lr" / "rnn"
                                / selected / "fixture-run"
                            ).resolve(),
                        )
                        self.assertEqual(config.model.interval_mode, "direct")

    def test_missing_yaml_selector_uses_legacy_cosine_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_config(root)
            output, config = self._dry_run(root)
            self.assertEqual(config.model.prediction_head, "cosine_embedding")
            self.assertIn("prediction_head=cosine_embedding,", output)
            self.assertIn("cosine_embedding", Path(config.saving_dir).parts)

    def test_explicit_cli_selector_overrides_yaml(self):
        for selected in self.job.PREDICTION_HEADS:
            configured = next(head for head in self.job.PREDICTION_HEADS if head != selected)
            with self.subTest(selected=selected):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    self._write_config(root, configured)
                    output, config = self._dry_run(root, "--prediction-head", selected)
                    self.assertEqual(config.model.prediction_head, selected)
                    self.assertIn(f"prediction_head={selected},", output)
                    self.assertIn(selected, Path(config.saving_dir).parts)
                    self.assertNotIn(configured, Path(config.saving_dir).parts)

    def test_invalid_yaml_selector_fails_before_writing_outputs(self):
        for selector in ("unknown-head", "", None):
            with self.subTest(selector=selector):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    self._write_config(root, selector)
                    stderr = io.StringIO()
                    with (
                        patch.object(self.job, "REPO_ROOT", root),
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(stderr),
                        self.assertRaises(SystemExit) as raised,
                    ):
                        self.job.main(
                            ["air", "--task-id", "0", "--output-root", str(root / "outputs")]
                        )
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn("prediction_head", stderr.getvalue())
                    self.assertFalse((root / "outputs").exists())

    def test_explicit_override_can_replace_invalid_yaml_selector(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_config(root, "outdated-head")
            _, config = self._dry_run(root, "--prediction-head", "cosine_embedding")
            self.assertEqual(config.model.prediction_head, "cosine_embedding")

    def test_saved_config_and_dispatched_path_use_yaml_selector(self):
        from dscp import run_iqn_cp

        for task_id, selected, runner_name in (
            (0, "cosine_embedding", "run_rnn_iqn_cp"),
            (5, "partially_monotonic", "run_transformer_iqn_cp"),
        ):
            with self.subTest(task_id=task_id, selected=selected):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    self._write_config(root, selected, task_id=task_id)
                    encoder, predictor = self.job.TASKS[task_id]
                    artifact_dir, artifact_name = self.job.DATASET_ARTIFACTS["air"]
                    artifact = (
                        root / "data" / artifact_dir / predictor
                        / f"{predictor}_{artifact_name}_data.pkl"
                    )
                    artifact.parent.mkdir(parents=True)
                    artifact.touch()
                    output_root = root / "outputs"
                    stream = io.StringIO()
                    with (
                        patch.object(self.job, "REPO_ROOT", root),
                        patch.object(run_iqn_cp, "run_rnn_iqn_cp") as rnn_runner,
                        patch.object(run_iqn_cp, "run_transformer_iqn_cp") as transformer_runner,
                        patch("torch.cuda.is_available", return_value=False),
                        contextlib.redirect_stdout(stream),
                    ):
                        self.job.main(
                            ["air", "--task-id", str(task_id), "--output-root", str(output_root)]
                        )
                    output_dir = (
                        output_root / "air" / predictor / encoder
                        / selected / "fixture-run"
                    ).resolve()
                    saved_path = output_dir / "resolved_config.yaml"
                    self.assertTrue(saved_path.is_file())
                    saved = OmegaConf.load(saved_path)
                    self.assertEqual(saved.model.prediction_head, selected)
                    self.assertEqual(Path(saved.saving_dir), output_dir)
                    self.assertIn(f"prediction_head={selected},", stream.getvalue())
                    chosen_runner = (
                        rnn_runner if runner_name == "run_rnn_iqn_cp"
                        else transformer_runner
                    )
                    other_runner = (
                        transformer_runner if runner_name == "run_rnn_iqn_cp"
                        else rnn_runner
                    )
                    chosen_runner.assert_called_once_with(str(saved_path))
                    other_runner.assert_not_called()

    def test_all_actual_templates_are_inherited_without_output_writes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "outputs"
            for dataset in self.job.DATASET_ARTIFACTS:
                for task_id, (encoder, predictor) in enumerate(self.job.TASKS):
                    with self.subTest(dataset=dataset, encoder=encoder, predictor=predictor):
                        template = load_experiment_config(
                            REPO_ROOT / "configs" / "iqn_cp_configs"
                            / f"iqn_{encoder}_{predictor}_{dataset}_config.yaml"
                        )
                        selected = str(
                            template.model.get("prediction_head", "cosine_embedding")
                        ).strip().lower()
                        stream = io.StringIO()
                        with contextlib.redirect_stdout(stream):
                            self.job.main(
                                [
                                    dataset, "--task-id", str(task_id),
                                    "--output-root", str(output_root), "--dry-run",
                                ]
                            )
                        output = stream.getvalue()
                        config = OmegaConf.create(output.split("\n", 2)[2])
                        self.assertEqual(config.model.prediction_head, selected)
                        self.assertIn(f"prediction_head={selected},", output)
                        relative_path = Path(config.saving_dir).relative_to(output_root.resolve())
                        self.assertEqual(
                            relative_path.parts[:4],
                            (dataset, predictor, encoder, selected),
                        )
                        self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
