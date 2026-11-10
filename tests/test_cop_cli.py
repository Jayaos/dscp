"""CLI integration and config routing for the original COP baseline."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from sbatch.sbatch_run_cop import run_cop as cli


REPO_ROOT = Path(__file__).resolve().parents[1]


class COPCLITests(unittest.TestCase):
    def test_all_nine_presets_route_to_matching_forecasts_and_independent_outputs(self):
        artifacts = {
            "air": ("air-10_prediction", "air-10"),
            "solar": ("solar_prediction", "nsdb-60m"),
            "sapflux": ("sapflux-solo3-large", "sapflux-solo3-large"),
        }
        output_parents = set()
        for dataset, (folder, artifact) in artifacts.items():
            for predictor in cli.BASE_PREDICTORS:
                with self.subTest(dataset=dataset, predictor=predictor):
                    args = cli.build_parser().parse_args([
                        "--dataset", dataset, "--base-predictor", predictor,
                    ])
                    path, config = cli.resolve_config(args)
                    self.assertEqual(path.name, f"cop_{predictor}_{dataset}_config.yaml")
                    self.assertEqual(Path(config.data.data_path), (
                        REPO_ROOT / "data" / folder / predictor
                        / f"{predictor}_{artifact}_data.pkl"
                    ))
                    self.assertEqual(config.data.calibration_ratio, 0.66)
                    self.assertEqual(config.data.test_ratio, 0.34)
                    self.assertEqual(config.model.learning_rate, 0.1)
                    self.assertEqual(config.model.scale, 0.5)
                    self.assertTrue(config.model.proportional_lr)
                    self.assertEqual(config.model.score_type, "signed")
                    self.assertEqual(config.model.cdf, "ecdf")
                    self.assertEqual(config.model.initialization, "calibration_tail")
                    self.assertEqual(config.tuning.learning_rate_grid, [1.0, .5, .1, .05])
                    self.assertIsNone(config.tuning.warmup_steps)
                    self.assertEqual(config.model.window_size, 100)
                    self.assertEqual(config.model.target_quantiles, [[0.05, 0.95]])
                    self.assertTrue(config.tuning.enabled)
                    parent = Path(config.saving_dir).parent
                    self.assertEqual(parent, REPO_ROOT / "results" / "cop" / dataset / predictor)
                    output_parents.add(parent)
        self.assertEqual(len(output_parents), 9)

    def test_custom_hyperparameters_and_tuning_survive_path_and_worker_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.yaml"
            output = Path(directory) / "results"
            artifact = Path(directory) / "forecasts.pkl"
            source = OmegaConf.load(
                REPO_ROOT / "configs/cop_configs/cop_lr_air_config.yaml"
            )
            source.model.learning_rate = 0.02
            source.model.window_size = 87
            source.model.scale = 0.2
            source.model.proportional_lr = False
            source.model.score_type = "absolute"
            source.model.initialization = "cold_start"
            source.tuning.enabled = True
            source.tuning.learning_rate_grid = [0.01, 0.02]
            source.tuning.scale_grid = [.2, .5]
            source.tuning.warmup_steps = 20
            source.tuning.window_size_grid = [40, 87]
            source.num_cores = 3
            OmegaConf.save(source, path)

            args = cli.build_parser().parse_args([
                str(path), "--data-path", str(artifact),
                "--output-dir", str(output),
            ])
            _, config = cli.resolve_config(args)
            self.assertEqual(config.model.learning_rate, 0.02)
            self.assertEqual(config.model.window_size, 87)
            self.assertEqual(config.model.scale, 0.2)
            self.assertFalse(config.model.proportional_lr)
            self.assertEqual(config.model.score_type, "absolute")
            self.assertEqual(config.model.initialization, "cold_start")
            self.assertTrue(config.tuning.enabled)
            self.assertEqual(config.tuning.learning_rate_grid, [0.01, 0.02])
            self.assertEqual(config.tuning.scale_grid, [.2, .5])
            self.assertEqual(config.tuning.warmup_steps, 20)
            self.assertEqual(config.tuning.window_size_grid, [40, 87])
            self.assertEqual(config.num_cores, 3)
            self.assertEqual(Path(config.data.data_path), artifact)
            self.assertEqual(Path(config.saving_dir), output)

            args = cli.build_parser().parse_args([str(path), "--num_cores", "2"])
            _, config = cli.resolve_config(args)
            self.assertEqual(config.num_cores, 2)
            self.assertFalse(output.exists())

    def test_dry_run_validates_missing_artifact_without_writes_or_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pkl"
            output = Path(directory) / "must_not_exist"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), \
                    patch("baselines.cop.run_cop.run_cop") as run:
                config = cli.main([
                    "--dataset", "sapflux", "--base-predictor", "chronos",
                    "--data-path", str(missing), "--output-dir", str(output),
                    "--dry-run",
                ])
            run.assert_not_called()
            self.assertIn("Artifact available: False", stream.getvalue())
            self.assertIn("score_type: signed", stream.getvalue())
            self.assertEqual(Path(config.data.data_path), missing)
            self.assertFalse(output.exists())

    def test_dispatch_forwards_resolved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "forecasts.pkl"
            artifact.touch()
            output = Path(directory) / "output"
            expected = {"series": {"result": "sentinel"}}
            with patch("baselines.cop.run_cop.run_cop", return_value=expected) as run:
                result = cli.main([
                    "--dataset", "air", "--base-predictor", "lstm",
                    "--data-path", str(artifact), "--output-dir", str(output),
                    "--num-cores", "2",
                ])
            self.assertIs(result, expected)
            run.assert_called_once()
            config = run.call_args.args[0]
            self.assertEqual(Path(config.data.data_path), artifact)
            self.assertEqual(Path(config.saving_dir), output)
            self.assertEqual(config.num_cores, 2)
            self.assertFalse(output.exists())

    def test_invalid_hyperparameter_is_rejected_even_in_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            output = Path(directory) / "output"
            source = OmegaConf.load(
                REPO_ROOT / "configs/cop_configs/cop_lr_air_config.yaml"
            )
            source.model.learning_rate = -0.1
            source.saving_dir = str(output)
            OmegaConf.save(source, path)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main([str(path), "--dry-run"])
            self.assertFalse(output.exists())

    def test_missing_artifact_fails_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pkl"
            with patch("baselines.cop.run_cop.run_cop") as run, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main([
                    "--dataset", "air", "--base-predictor", "lr",
                    "--data-path", str(missing),
                ])
            run.assert_not_called()

    def test_preset_arguments_require_an_unambiguous_route(self):
        for arguments in ([], ["--dataset", "air"], ["--base-predictor", "lr"],
                          ["custom.yaml", "--dataset", "air", "--base-predictor", "lr"]):
            with self.subTest(arguments=arguments):
                args = cli.build_parser().parse_args(arguments)
                with self.assertRaises(ValueError):
                    cli.resolve_config(args)

    def test_selected_hyperparameter_replay_path_resolves_from_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.yaml"
            source = OmegaConf.load(REPO_ROOT / "configs/cop_configs/cop_lr_air_config.yaml")
            source.tuning.enabled = False
            source.selected_hyperparameters_path = "results/cop/selected_hyperparameters.pkl"
            OmegaConf.save(source, path)
            args = cli.build_parser().parse_args([str(path)])
            _, config = cli.resolve_config(args)
            self.assertFalse(config.tuning.enabled)
            self.assertEqual(Path(config.selected_hyperparameters_path),
                             REPO_ROOT / "results/cop/selected_hyperparameters.pkl")

    def test_direct_and_module_entrypoints(self):
        entrypoints = [
            ["sbatch/sbatch_run_cop/run_cop.py"],
            ["-m", "sbatch.sbatch_run_cop.run_cop"],
        ]
        for entrypoint in entrypoints:
            for flags in (["--help"], [
                "--dataset", "air", "--base-predictor", "lr", "--dry-run",
            ]):
                with self.subTest(entrypoint=entrypoint, flags=flags):
                    result = subprocess.run(
                        [sys.executable] + entrypoint + flags,
                        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    expected = "COP" if "--help" in flags else "Sequence workers: 1"
                    self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
