"""CLI integration for the Areces et al. LQT and SQT+AR baselines."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from sbatch.sbatch_run_conformalopt import run_conformalopt as cli


REPO_ROOT = Path(__file__).resolve().parents[1]


class ConformalOptCLITests(unittest.TestCase):
    def test_all_presets_route_to_matching_artifacts_and_independent_outputs(self):
        artifacts = {
            "air": ("air-10_prediction", "air-10"),
            "solar": ("solar_prediction", "nsdb-60m"),
            "sapflux": ("sapflux-solo3-large", "sapflux-solo3-large"),
        }
        output_parents = set()
        for method in cli.METHODS:
            for dataset, (folder, artifact) in artifacts.items():
                for predictor in cli.BASE_PREDICTORS:
                    with self.subTest(method=method, dataset=dataset, predictor=predictor):
                        args = cli.build_parser().parse_args([
                            "--method", method, "--dataset", dataset,
                            "--base-predictor", predictor,
                        ])
                        path, config = cli.resolve_config(args)
                        self.assertEqual(path.name, f"{method}_{predictor}_{dataset}_config.yaml")
                        self.assertEqual(Path(config.data.data_path), (
                            REPO_ROOT / "data" / folder / predictor
                            / f"{predictor}_{artifact}_data.pkl"
                        ))
                        self.assertEqual(config.data.calibration_ratio, 0.66)
                        self.assertEqual(config.data.test_ratio, 0.34)
                        self.assertEqual(config.model.method, method)
                        self.assertEqual(config.model.learning_rate_schedule, "fixed")
                        self.assertEqual(config.model.target_quantiles, [[0.05, 0.95]])
                        self.assertEqual(config.model.ar_window_size, 200)
                        self.assertEqual(config.model.solver, "CLARABEL")
                        self.assertTrue(config.tuning.enabled)
                        self.assertEqual(config.tuning.p_order_grid, [0, 1, 2])
                        self.assertEqual(len(config.tuning.learning_rate_grid), 11)
                        self.assertEqual(config.tuning.coverage_tolerance, 0.01)
                        if method == "lqt":
                            self.assertEqual(config.tuning.bias_grid, [.1, 1, 5, 10, 200, 100, 1000])
                        parent = Path(config.saving_dir).parent
                        self.assertEqual(parent, REPO_ROOT / "results" / method / dataset / predictor)
                        output_parents.add(parent)
        self.assertEqual(len(output_parents), 18)

    def test_custom_hyperparameters_and_tuning_survive_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.yaml"
            output = Path(directory) / "results"
            artifact = Path(directory) / "forecasts.pkl"
            source = OmegaConf.load(
                REPO_ROOT / "configs/conformalopt_configs/sqt_ar_lr_air_config.yaml"
            )
            source.model.learning_rate = .02
            source.model.learning_rate_schedule = "decaying"
            source.model.decay_exponent = .7
            source.model.p_order = 2
            source.model.ar_window_size = 45
            source.model.solver_options = {"max_iter": 200}
            source.tuning.enabled = False
            source.selected_hyperparameters_path = "results/selected_hyperparameters.pkl"
            source.tuning.learning_rate_grid = [.01, .02]
            source.tuning.p_order_grid = [1, 2]
            source.num_cores = 3
            OmegaConf.save(source, path)
            args = cli.build_parser().parse_args([
                str(path), "--data-path", str(artifact), "--output-dir", str(output),
            ])
            _, config = cli.resolve_config(args)
            self.assertEqual(config.model.learning_rate, .02)
            self.assertEqual(config.model.learning_rate_schedule, "decaying")
            self.assertEqual(config.model.decay_exponent, .7)
            self.assertEqual(config.model.p_order, 2)
            self.assertEqual(config.model.ar_window_size, 45)
            self.assertEqual(config.model.solver_options.max_iter, 200)
            self.assertFalse(config.tuning.enabled)
            self.assertEqual(
                Path(config.selected_hyperparameters_path),
                REPO_ROOT / "results/selected_hyperparameters.pkl",
            )
            self.assertEqual(config.tuning.learning_rate_grid, [.01, .02])
            self.assertEqual(config.tuning.p_order_grid, [1, 2])
            self.assertEqual(config.num_cores, 3)
            self.assertEqual(Path(config.data.data_path), artifact)
            self.assertEqual(Path(config.saving_dir), output)
            args = cli.build_parser().parse_args([str(path), "--num_cores", "2"])
            _, config = cli.resolve_config(args)
            self.assertEqual(config.num_cores, 2)
            self.assertFalse(output.exists())

    def test_dry_run_does_not_evaluate_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pkl"
            output = Path(directory) / "must_not_exist"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), \
                    patch("baselines.conformalopt.run_conformalopt.run_conformalopt") as run:
                config = cli.main([
                    "--method", "sqt_ar", "--dataset", "sapflux",
                    "--base-predictor", "chronos", "--data-path", str(missing),
                    "--output-dir", str(output), "--dry-run",
                ])
            run.assert_not_called()
            self.assertIn("Artifact available: False", stream.getvalue())
            self.assertIn("method: sqt_ar", stream.getvalue())
            self.assertEqual(Path(config.data.data_path), missing)
            self.assertFalse(output.exists())

    def test_dispatch_forwards_resolved_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "forecasts.pkl"
            artifact.touch()
            output = Path(directory) / "output"
            expected = {"series": {"result": "sentinel"}}
            with patch("baselines.conformalopt.run_conformalopt.run_conformalopt", return_value=expected) as run:
                result = cli.main([
                    "--method", "lqt", "--dataset", "air", "--base-predictor", "lstm",
                    "--data-path", str(artifact), "--output-dir", str(output),
                    "--num-cores", "2",
                ])
            self.assertIs(result, expected)
            run.assert_called_once()
            config = run.call_args.args[0]
            self.assertEqual(config.model.method, "lqt")
            self.assertEqual(Path(config.data.data_path), artifact)
            self.assertEqual(Path(config.saving_dir), output)
            self.assertEqual(config.num_cores, 2)
            self.assertFalse(output.exists())

    def test_invalid_hyperparameters_rejected_during_dry_run(self):
        invalid = [
            ("learning_rate", -1),
            ("p_order", -1),
            ("ar_window_size", 0),
            ("learning_rate_schedule", "unknown"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            output = Path(directory) / "output"
            for key, value in invalid:
                with self.subTest(setting=key):
                    source = OmegaConf.load(
                        REPO_ROOT / "configs/conformalopt_configs/lqt_lr_air_config.yaml"
                    )
                    source.model[key] = value
                    source.saving_dir = str(output)
                    OmegaConf.save(source, path)
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        cli.main([str(path), "--dry-run"])
                    self.assertFalse(output.exists())

    def test_missing_artifact_fails_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pkl"
            with patch("baselines.conformalopt.run_conformalopt.run_conformalopt") as run, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main([
                    "--method", "lqt", "--dataset", "air", "--base-predictor", "lr",
                    "--data-path", str(missing),
                ])
            run.assert_not_called()

    def test_preset_arguments_require_an_unambiguous_route(self):
        for arguments in (
            [], ["--method", "lqt"], ["--dataset", "air"], ["--base-predictor", "lr"],
            ["--method", "lqt", "--dataset", "air"],
            ["--dataset", "air", "--base-predictor", "lr"],
            ["custom.yaml", "--method", "lqt"],
            ["custom.yaml", "--method", "lqt", "--dataset", "air", "--base-predictor", "lr"],
        ):
            with self.subTest(arguments=arguments):
                args = cli.build_parser().parse_args(arguments)
                with self.assertRaises(ValueError):
                    cli.resolve_config(args)

    def test_direct_and_module_entrypoints(self):
        entrypoints = [
            ["sbatch/sbatch_run_conformalopt/run_conformalopt.py"],
            ["-m", "sbatch.sbatch_run_conformalopt.run_conformalopt"],
        ]
        for entrypoint in entrypoints:
            for flags in (["--help"], [
                "--method", "lqt", "--dataset", "air", "--base-predictor", "lr", "--dry-run",
            ]):
                with self.subTest(entrypoint=entrypoint, flags=flags):
                    result = subprocess.run(
                        [sys.executable] + entrypoint + flags, cwd=REPO_ROOT,
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    expected = "SQT+AR" if "--help" in flags else "Sequence workers: 1"
                    self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
