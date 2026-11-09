"""Entry-point checks for the log-only AMZN / Prophet reproduction."""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from sbatch.sbatch_reproduce_cop_table2 import run_amzn_prophet as cli


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_MODULE = "baselines.cop_table2.run_reproduction"


class COPTable2CLITests(unittest.TestCase):
    def test_default_inputs_are_anchored_to_repository(self):
        args = cli.build_parser().parse_args([])
        with patch.object(Path, "is_file", return_value=True):
            paths = cli.resolve_inputs(args)
        self.assertEqual(paths, (
            REPO_ROOT / "data" / "amzn_prophet.npz",
            REPO_ROOT / "data" / "amzn_prophet_targets.csv",
        ))
        self.assertEqual(args.method, "all")
        self.assertEqual(args.lqt_order, 1)

    def test_dry_run_validates_inputs_prints_metadata_and_never_runs_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            forecast = Path(directory) / "forecasts.npz"
            targets = Path(directory) / "targets.csv"
            forecast.touch()
            targets.touch()
            data = {
                "y": [2.0, 3.0], "forecasts": [2.5, 2.5],
                "timestamps": ["2020-01-01", "2020-01-02"],
                "metadata": {"alignment": "authors' dates", "count": 2},
            }
            stream = io.StringIO()
            before = set(Path(directory).rglob("*"))
            with contextlib.redirect_stdout(stream), \
                    patch(RUNNER_MODULE + ".load_amzn_data", return_value=data) as load, \
                    patch(RUNNER_MODULE + ".run_reproduction") as run:
                result = cli.main([
                    "--data-path", str(forecast), "--targets-path", str(targets),
                    "--method", "lqt", "--dry-run",
                ])
            self.assertIs(result, data)
            load.assert_called_once_with(forecast, targets)
            run.assert_not_called()
            self.assertIn("Validated observations: 2", stream.getvalue())
            self.assertIn("authors' dates", stream.getvalue())
            self.assertIn("Methods: LQT", stream.getvalue())
            self.assertEqual(before, set(Path(directory).rglob("*")))

    def test_dispatches_all_or_selected_method_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            forecast = Path(directory) / "forecasts.npz"
            targets = Path(directory) / "targets.csv"
            forecast.touch()
            targets.touch()
            expected = [{"method": "sentinel"}]
            before = set(Path(directory).rglob("*"))
            for method in ("all", "aci", "lqt", "cop"):
                with self.subTest(method=method), contextlib.redirect_stdout(io.StringIO()), \
                        patch(RUNNER_MODULE + ".run_reproduction", return_value=expected) as run:
                    result = cli.main([
                        "--data-path", str(forecast), "--targets-path", str(targets),
                        "--method", method,
                    ])
                    self.assertIs(result, expected)
                    run.assert_called_once_with(
                        forecast, targets, methods=None if method == "all" else (method,), lqt_order=1,
                    )
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch(RUNNER_MODULE + ".run_reproduction", return_value=expected) as run:
                cli.main(["--data-path", str(forecast), "--targets-path", str(targets),
                          "--method", "lqt", "--lqt-order", "2"])
                run.assert_called_once_with(forecast, targets, methods=("lqt",), lqt_order=2)
            self.assertEqual(before, set(Path(directory).rglob("*")))

    def test_missing_input_fails_before_algorithm_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            forecast = Path(directory) / "forecasts.npz"
            targets = Path(directory) / "targets.csv"
            for existing, missing in ((targets, forecast), (forecast, targets)):
                existing.touch()
                if missing.exists():
                    missing.unlink()
                stream = io.StringIO()
                with self.subTest(missing=missing), contextlib.redirect_stderr(stream), \
                        patch(RUNNER_MODULE + ".run_reproduction") as run, \
                        self.assertRaises(SystemExit) as error:
                    cli.main(["--data-path", str(forecast), "--targets-path", str(targets)])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("input file does not exist", stream.getvalue())
                run.assert_not_called()

    def test_invalid_input_gets_an_actionable_cli_error(self):
        with patch.object(Path, "is_file", return_value=True), \
                patch(RUNNER_MODULE + ".load_amzn_data", side_effect=ValueError("Target dates do not align.")), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as errors, \
                self.assertRaises(SystemExit) as error:
            cli.main(["--dry-run"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("Target dates do not align.", errors.getvalue())

    def test_output_and_configuration_options_are_not_exposed(self):
        for option in ("--output-dir", "--config", "--save-results", "--plot"):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                cli.build_parser().parse_args([option, "unused"])
            self.assertEqual(error.exception.code, 2)

    def test_direct_and_module_entrypoints_work_outside_the_repository(self):
        entrypoints = (
            [str(REPO_ROOT / "sbatch/sbatch_reproduce_cop_table2/run_amzn_prophet.py")],
            ["-m", "sbatch.sbatch_reproduce_cop_table2.run_amzn_prophet"],
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(REPO_ROOT), environment.get("PYTHONPATH"),
        )))
        with tempfile.TemporaryDirectory() as directory:
            for entrypoint in entrypoints:
                with self.subTest(entrypoint=entrypoint):
                    result = subprocess.run(
                        [sys.executable, "-B"] + entrypoint + ["--help"],
                        cwd=directory, env=environment, capture_output=True, text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("Table 2", result.stdout)
                    self.assertIn("--targets-path", result.stdout)
                    self.assertNotIn("--output-dir", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
