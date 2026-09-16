import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from sbatch.sbatch_run_split_cp import run_split_cp as cli


REPO_ROOT = Path(__file__).resolve().parents[1]


class SplitCPCLITests(unittest.TestCase):
    def test_all_nine_presets_use_saved_forecasts_and_calibration_test_only(self):
        paths = {
            "air": ("air-10_prediction", "air-10"),
            "solar": ("solar_prediction", "nsdb-60m"),
            "sapflux": ("sapflux-solo3-large", "sapflux-solo3-large"),
        }
        outputs = set()
        for dataset, (folder, artifact) in paths.items():
            for predictor in cli.BASE_PREDICTORS:
                with self.subTest(dataset=dataset, predictor=predictor):
                    args = cli.build_parser().parse_args([
                        "--dataset", dataset, "--base-predictor", predictor,
                    ])
                    config_path, config = cli.resolve_config(args)
                    self.assertTrue(config_path.is_file())
                    self.assertEqual(Path(config.data.data_path), (
                        REPO_ROOT / "data" / folder / predictor
                        / "{}_{}_data.pkl".format(predictor, artifact)
                    ))
                    self.assertEqual(config.data.calibration_ratio, 0.66)
                    self.assertEqual(config.data.test_ratio, 0.34)
                    self.assertEqual(config.model.prediction_step, 1)
                    self.assertEqual(config.num_cores, 1)
                    for name in ("train_ratio", "validation_ratio", "normalize"):
                        self.assertNotIn(name, config.data)
                    self.assertNotIn("seed", config)
                    self.assertEqual(Path(config.saving_dir), (
                        REPO_ROOT / "results" / "split_cp_{}_{}".format(predictor, dataset)
                    ))
                    outputs.add(config.saving_dir)
        self.assertEqual(len(outputs), 9)

    def test_dry_run_accepts_missing_artifact_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must_not_be_created"
            missing = Path(directory) / "missing.pkl"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), \
                    patch("baselines.split_cp.run_split_cp.run_split_cp") as run:
                config = cli.main([
                    "--dataset", "sapflux", "--base-predictor", "chronos", "--dry-run",
                    "--data-path", str(missing), "--output-dir", str(output),
                ])
            run.assert_not_called()
            self.assertIn("Artifact available: False", stdout.getvalue())
            self.assertEqual(Path(config.data.data_path), missing)
            self.assertFalse(output.exists())

    def test_worker_count_uses_yaml_unless_explicitly_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.yaml"
            config = OmegaConf.load(
                REPO_ROOT / "configs/split_cp_configs/split_cp_lr_air_config.yaml"
            )
            config.num_cores = 3
            OmegaConf.save(config, path)
            for override, expected in (([], 3), (["--num-cores", "2"], 2),
                                       (["--num_cores", "4"], 4)):
                with self.subTest(override=override):
                    args = cli.build_parser().parse_args([str(path)] + override)
                    _, resolved = cli.resolve_config(args)
                    self.assertEqual(resolved.num_cores, expected)

    def test_dispatch_passes_resolved_overrides_without_writing_launch_config(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "forecasts.pkl"
            artifact.touch()
            output = Path(directory) / "results"
            expected = {"series": {"result": "sentinel"}}
            with patch("baselines.split_cp.run_split_cp.run_split_cp", return_value=expected) as run:
                result = cli.main([
                    "--dataset", "air", "--base-predictor", "lr",
                    "--data-path", str(artifact), "--output-dir", str(output),
                    "--num-cores", "2",
                ])
            self.assertIs(result, expected)
            run.assert_called_once()
            resolved = run.call_args.args[0]
            self.assertEqual(Path(resolved.data.data_path), artifact)
            self.assertEqual(Path(resolved.saving_dir), output)
            self.assertEqual(resolved.num_cores, 2)
            self.assertFalse(output.exists())

    def test_dry_run_rejects_invalid_config_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            output = Path(directory) / "output"
            config = OmegaConf.load(
                REPO_ROOT / "configs/split_cp_configs/split_cp_lr_air_config.yaml"
            )
            config.data.calibration_ratio = 1.0
            config.saving_dir = str(output)
            OmegaConf.save(config, path)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main([str(path), "--dry-run"])
            self.assertFalse(output.exists())

    def test_preset_and_config_arguments_are_unambiguous(self):
        for arguments in ([], ["--dataset", "air"], ["--base-predictor", "lr"],
                          ["custom.yaml", "--dataset", "air", "--base-predictor", "lr"]):
            with self.subTest(arguments=arguments):
                args = cli.build_parser().parse_args(arguments)
                with self.assertRaises(ValueError):
                    cli.resolve_config(args)

    def test_direct_and_module_help_and_dry_run_work(self):
        entrypoints = [
            ["sbatch/sbatch_run_split_cp/run_split_cp.py"],
            ["-m", "sbatch.sbatch_run_split_cp.run_split_cp"],
        ]
        for entrypoint in entrypoints:
            for flags in (["--help"], ["--dataset", "air", "--base-predictor", "lr", "--dry-run"]):
                with self.subTest(entrypoint=entrypoint, flags=flags):
                    completed = subprocess.run(
                        [sys.executable] + entrypoint + flags, cwd=REPO_ROOT,
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertIn("SplitCP" if "--help" in flags else "Sequence workers: 1",
                                  completed.stdout)


if __name__ == "__main__":
    unittest.main()
