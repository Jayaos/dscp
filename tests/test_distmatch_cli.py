"""Verify launch configuration, CPU overrides, and import-light dry runs."""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from sbatch.sbatch_run_distmatch import run_distmatch as cli


REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET = REPO_ROOT / "configs/distmatch_configs/distmatch_lr_air_config.yaml"


class DistMatchCLITests(unittest.TestCase):
    def _config(self, directory, **overrides):
        config = OmegaConf.load(PRESET)
        artifact = Path(directory) / "forecast.pkl"
        artifact.write_bytes(b"Path validation does not load this artifact.")
        config.data.data_path = str(artifact)
        config.saving_dir = str(Path(directory) / "output")
        config.num_cores = 3
        for key, value in overrides.items():
            OmegaConf.update(config, key, value)
        config_path = Path(directory) / "experiment.yaml"
        OmegaConf.save(config, config_path)
        return config_path

    def test_omitted_worker_option_preserves_config_at_runner_boundary(self):
        for option, expected_override, expected_workers in (
            ([], None, 3), (["--num-cores", "2"], 2, 2),
        ):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                config_path = self._config(directory)
                calls = []
                runner = types.ModuleType("baselines.distmatch.run_distmatch")

                def run_distmatch(path, num_cores=None):
                    calls.append((OmegaConf.load(path), num_cores))
                    return "finished"

                runner.run_distmatch = run_distmatch
                with patch.dict(sys.modules, {runner.__name__: runner}):
                    result = cli.main([str(config_path), *option])
                self.assertEqual(result, "finished")
                self.assertEqual(len(calls), 1)
                launched, override = calls[0]
                self.assertEqual(override, expected_override)
                self.assertEqual(launched.num_cores, expected_workers)
                self.assertEqual(launched.threads_per_worker, 1)

    def test_dry_run_resolves_paths_and_creates_no_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._config(directory)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                config = cli.main([str(config_path), "--dry-run"])
            self.assertIn("Sequence workers: 3", stdout.getvalue())
            self.assertTrue(Path(config.data.data_path).is_absolute())
            self.assertFalse(Path(config.saving_dir).exists())

    def test_relative_paths_resolve_against_repository_root(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._config(directory, **{
                "data.data_path": "data/air-10_prediction/lr/lr_air-10_data.pkl",
                "saving_dir": "results/cli_distmatch/seed_${seed}",
                "matching.cache_dir": "misc/distmatch_cache",
            })
            args = cli.build_parser().parse_args([str(config_path), "--seed", "17"])
            _, config = cli.resolve_config(args)
            self.assertEqual(Path(config.data.data_path), REPO_ROOT / "data/air-10_prediction/lr/lr_air-10_data.pkl")
            self.assertEqual(Path(config.saving_dir), REPO_ROOT / "results/cli_distmatch/seed_17")
            self.assertEqual(Path(config.matching.cache_dir), REPO_ROOT / "misc/distmatch_cache")

    def test_slurm_allocation_preserves_configured_workers_and_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._config(directory, threads_per_worker=2)
            with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "8"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                config = cli.main([str(config_path), "--dry-run"])
            self.assertEqual(config.num_cores, 3)
            self.assertEqual(config.threads_per_worker, 2)
            self.assertFalse(Path(config.saving_dir).exists())

    def test_slurm_allocation_error_recommends_sufficient_request(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._config(directory, threads_per_worker=2)
            stderr = io.StringIO()
            with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "4"}), \
                    contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
                cli.main([str(config_path), "--dry-run"])
            self.assertEqual(exc.exception.code, 2)
            self.assertIn("3 workers * 2 threads", stderr.getvalue())
            self.assertIn("sbatch --cpus-per-task=6", stderr.getvalue())
            self.assertFalse((Path(directory) / "output").exists())

    def test_dry_run_rejects_missing_artifact_before_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._config(directory)
            missing = Path(directory) / "missing.pkl"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
                cli.main([str(config_path), "--dry-run", "--data-path", str(missing)])
            self.assertEqual(exc.exception.code, 2)
            self.assertIn("Saved forecast artifact does not exist", stderr.getvalue())
            self.assertFalse((Path(directory) / "output").exists())

    def test_invalid_splits_and_worker_values_fail_in_dry_run(self):
        cases = (
            {"data.test_ratio": 0.5}, {"data.train_ratio": -0.5},
            {"num_cores": 0}, {"num_cores": True}, {"num_cores": 1.5},
            {"threads_per_worker": 0}, {"model.prediction_step": 2},
        )
        for updates in cases:
            with self.subTest(updates=updates), tempfile.TemporaryDirectory() as directory:
                config_path = self._config(directory, **updates)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    cli.main([str(config_path), "--dry-run"])
                self.assertEqual(exc.exception.code, 2)
                self.assertFalse((Path(directory) / "output").exists())

    def test_help_and_dry_run_do_not_import_numerical_packages(self):
        script = """
import importlib.abc
import sys
class BlockNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch', 'scipy', 'sklearn', 'sklearn_quantile', 'matplotlib'}:
            raise AssertionError('Unexpected numerical import: ' + fullname)
sys.meta_path.insert(0, BlockNumericalImports())
from sbatch.sbatch_run_distmatch.run_distmatch import main
main(sys.argv[1:])
"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._config(directory)
            for arguments in (["--help"], [str(config_path), "--dry-run"]):
                with self.subTest(arguments=arguments):
                    result = subprocess.run(
                        [sys.executable, "-c", script, *arguments], cwd=REPO_ROOT,
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_checked_in_presets_and_grid_have_portable_paths(self):
        for name in ("lr_air", "lstm_air", "chronos_air", "lr_solar", "lstm_sapflux"):
            with self.subTest(name=name):
                path = REPO_ROOT / "configs/distmatch_configs" / f"distmatch_{name}_config.yaml"
                args = cli.build_parser().parse_args([str(path)])
                _, config = cli.resolve_config(args)
                self.assertEqual(config.num_cores, 4)
                self.assertEqual(config.threads_per_worker, 1)
                self.assertFalse(config.data.normalize)
                self.assertEqual(config.model.past_window_len, 100)
                self.assertTrue(Path(config.data.data_path).is_relative_to(REPO_ROOT))
        grid = OmegaConf.load(REPO_ROOT / "configs/distmatch_configs/distmatch_air_tuning_config.yaml")
        self.assertEqual(list(grid.grid["model.match_threshold"]), [0.025, 0.05, 0.1])
        self.assertEqual(list(grid.grid["model.past_window_len"]), [25, 50, 100])


if __name__ == "__main__":
    unittest.main()
