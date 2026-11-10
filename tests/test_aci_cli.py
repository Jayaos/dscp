"""CLI integration and config routing for the original ACI baseline."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from sbatch.sbatch_run_aci import run_aci as cli


REPO_ROOT = Path(__file__).resolve().parents[1]


class ACICLITests(unittest.TestCase):
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
                    self.assertEqual(path.name, f"aci_{predictor}_{dataset}_config.yaml")
                    self.assertEqual(Path(config.data.data_path), (
                        REPO_ROOT / "data" / folder / predictor
                        / f"{predictor}_{artifact}_data.pkl"
                    ))
                    self.assertEqual(config.data.calibration_ratio, 0.66)
                    self.assertEqual(config.data.test_ratio, 0.34)
                    self.assertEqual(config.model.gamma, 0.005)
                    self.assertEqual(config.model.window_size, 1250)
                    self.assertEqual(config.model.target_quantiles, [[0.05, 0.95]])
                    self.assertNotIn("tuning", config)
                    parent = Path(config.saving_dir).parent
                    self.assertEqual(parent, REPO_ROOT / "results" / "aci" / dataset / predictor)
                    output_parents.add(parent)
        self.assertEqual(len(output_parents), 9)

    def test_fixed_hyperparameters_survive_path_and_worker_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.yaml"
            output = Path(directory) / "results"
            artifact = Path(directory) / "forecasts.pkl"
            source = OmegaConf.load(
                REPO_ROOT / "configs/aci_configs/aci_lr_air_config.yaml"
            )
            source.model.gamma = 0.02
            source.model.window_size = 87
            source.model.initial_alpha = 0.2
            source.num_cores = 3
            OmegaConf.save(source, path)

            args = cli.build_parser().parse_args([
                str(path), "--data-path", str(artifact),
                "--output-dir", str(output),
            ])
            _, config = cli.resolve_config(args)
            self.assertEqual(config.model.gamma, 0.02)
            self.assertEqual(config.model.window_size, 87)
            self.assertEqual(config.model.initial_alpha, 0.2)
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
                    patch("baselines.aci.run_aci.run_aci") as run:
                config = cli.main([
                    "--dataset", "sapflux", "--base-predictor", "chronos",
                    "--data-path", str(missing), "--output-dir", str(output),
                    "--dry-run",
                ])
            run.assert_not_called()
            self.assertIn("Artifact available: False", stream.getvalue())
            self.assertIn("gamma: 0.005", stream.getvalue())
            self.assertEqual(Path(config.data.data_path), missing)
            self.assertFalse(output.exists())

    def test_dispatch_forwards_resolved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "forecasts.pkl"
            artifact.touch()
            output = Path(directory) / "output"
            expected = {"series": {"result": "sentinel"}}
            with patch("baselines.aci.run_aci.run_aci", return_value=expected) as run:
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
                REPO_ROOT / "configs/aci_configs/aci_lr_air_config.yaml"
            )
            source.model.gamma = -0.1
            source.saving_dir = str(output)
            OmegaConf.save(source, path)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main([str(path), "--dry-run"])
            self.assertFalse(output.exists())

    def test_stale_tuning_config_is_rejected_before_normal_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.yaml"
            source = OmegaConf.load(REPO_ROOT / "configs/aci_configs/aci_lr_air_config.yaml")
            for tuning in ({"enabled": True}, {"enabled": False}, {}):
                source.tuning = tuning
                OmegaConf.save(source, path)
                error = io.StringIO()
                with self.subTest(tuning=tuning), patch("baselines.aci.run_aci.run_aci") as run, \
                        contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
                    cli.main([str(path), "--dry-run"])
                run.assert_not_called()
                self.assertIn("do not accept a tuning section", error.getvalue())
                self.assertIn("sbatch.sbatch_run_tuning.run_aci_tuning", error.getvalue())

    def test_missing_artifact_fails_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pkl"
            with patch("baselines.aci.run_aci.run_aci") as run, \
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

    def test_direct_and_module_entrypoints(self):
        entrypoints = [
            ["sbatch/sbatch_run_aci/run_aci.py"],
            ["-m", "sbatch.sbatch_run_aci.run_aci"],
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
                    expected = "ACI" if "--help" in flags else "Sequence workers: 1"
                    self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
