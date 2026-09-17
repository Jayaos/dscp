"""KOWCPI selects settings inside calibration and reserves the final test tail."""

import contextlib
import io
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from baselines.kowcpi import run_kowcpi as ordinary
from sbatch.sbatch_run_tuning import run_kowcpi_tuning as tuning


REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR = (0.05, 0.95)


def _config(*, normalize=False, kernel="epanechnikov", ratio=0.15):
    return OmegaConf.create({
        "data": {"calibration_ratio": 0.66, "normalize": normalize},
        "model": {
            "target_quantiles": [list(PAIR)],
            "prediction_step": 1,
            "past_window": 3,
            "kernel": kernel,
            "bandwidth": None,
            "bandwidth_range": [1.0, 4.0, 4],
            "use_beta_search": False,
            "update_with_test": True,
        },
        "tuning": {"model_selection_valid_ratio": ratio, "delta_threshold": -1.0},
    })


def _sequence():
    index = np.arange(100, dtype=float)
    predictions = 10.0 + index / 20.0
    return {
        "heldout_y": predictions + np.sin(index / 3.0) + 0.1 * np.cos(index),
        "heldout_predictions": predictions,
    }


class CapturingEstimator:
    """Return known intervals while preserving real split and reporting paths."""

    def __init__(self, **kwargs):
        self.bandwidth = 2.0
        self.settings = kwargs
        self.call = None

    def predict_residual_intervals(self, **kwargs):
        self.call = dict(kwargs)
        self.call["residuals"] = np.asarray(kwargs["residuals"]).copy()
        size = kwargs["test_size"]
        return np.full(size, -0.25), np.full(size, 0.75)


class KOWCPITuningTests(unittest.TestCase):
    @contextlib.contextmanager
    def _capture(self, module):
        estimators = []

        def factory(**kwargs):
            estimator = CapturingEstimator(**kwargs)
            estimators.append(estimator)
            return estimator

        with patch.object(module, "KOWCPIResidualIntervalEstimator", side_effect=factory):
            yield estimators

    def test_validation_indices_reach_estimator_and_real_metrics(self):
        item = _sequence()
        with self._capture(tuning) as estimators:
            result = tuning._run_single_trial(_config(), item)
        call = estimators[0].call
        self.assertEqual((call["calibration_size"], call["test_size"]), (56, 10))
        np.testing.assert_allclose(
            call["residuals"], (item["heldout_y"] - item["heldout_predictions"])[:66]
        )
        expected_metadata = {
            "evaluation_split": "validation", "final_test_evaluated": False,
            "evaluation_start": 56, "evaluation_end": 66,
            "calibration_size": 56, "valid_size": 10,
            "nominal_calibration_size": 66, "test_size": 34,
        }
        for name, expected in expected_metadata.items():
            self.assertEqual(result[name], expected, name)
        errors = item["heldout_y"][56:66] - item["heldout_predictions"][56:66]
        expected_scores = 1.0 + 20.0 * (
            np.maximum(-0.25 - errors, 0.0) + np.maximum(errors - 0.75, 0.0)
        )
        metrics = result["pair_metrics"][str(PAIR)]
        self.assertAlmostEqual(metrics["avg_coverage"], np.mean((errors >= -0.25) & (errors <= 0.75)))
        self.assertAlmostEqual(metrics["avg_interval_width"], 1.0)
        self.assertAlmostEqual(result["selection_score"], expected_scores.mean(), places=4)

    def test_tuning_normalization_uses_only_initial_history(self):
        config = _config(normalize=True)
        original = _sequence()
        changed = _sequence()
        changed["heldout_y"][56:66] += 1000.0
        calls = []
        for item in (original, changed):
            with self._capture(tuning) as estimators:
                tuning._run_single_trial(config, item)
            calls.append(estimators[0].call)
        expected_std = original["heldout_y"][:56].std() + 1e-8
        np.testing.assert_allclose(
            calls[0]["residuals"],
            (original["heldout_y"][:66] - original["heldout_predictions"][:66]) / expected_std,
        )
        np.testing.assert_array_equal(calls[0]["residuals"][:56], calls[1]["residuals"][:56])
        np.testing.assert_allclose(
            calls[1]["residuals"][56:] - calls[0]["residuals"][56:], 1000.0 / expected_std
        )

    def test_normal_run_refits_on_all_calibration_and_evaluates_final_test(self):
        config = _config(normalize=True)
        item = _sequence()
        with self._capture(ordinary) as estimators, contextlib.redirect_stdout(io.StringIO()):
            result = ordinary._run_kowcpi_sequence("example", item, config, [list(PAIR)])
        call = estimators[0].call
        self.assertEqual((call["calibration_size"], call["test_size"]), (66, 34))
        self.assertEqual(len(call["residuals"]), 100)
        expected_std = item["heldout_y"][:66].std() + 1e-8
        np.testing.assert_allclose(
            call["residuals"], (item["heldout_y"] - item["heldout_predictions"]) / expected_std
        )
        metadata = result["metadata"]
        self.assertEqual(metadata["evaluation_split"], "test")
        self.assertEqual((metadata["evaluation_start"], metadata["evaluation_end"]), (66, 100))
        self.assertEqual(metadata["boundaries"]["calibration_size"], 66)
        self.assertEqual(metadata["boundaries"]["test_size"], 34)
        interval_results = result["evaluation_results"][PAIR]
        np.testing.assert_allclose(interval_results["target_y"], item["heldout_y"][66:], rtol=1e-6)
        self.assertEqual(len(interval_results["coverage"]), 34)
        self.assertAlmostEqual(interval_results["train_residuals_std"], expected_std)

    def test_real_tuning_bandwidth_and_metrics_ignore_poisoned_final_test(self):
        for kernel in ("epanechnikov", "gaussian"):
            for normalize in (False, True):
                with self.subTest(kernel=kernel, normalize=normalize):
                    config = _config(kernel=kernel, normalize=normalize)
                    with contextlib.redirect_stdout(io.StringIO()):
                        reference = tuning._run_single_trial(config, _sequence())
                    self.assertIsNotNone(reference["selected_bandwidths"][str(PAIR)])
                    self.assertTrue(np.isfinite(reference["selection_score"]))
                    for poison in (1e100, np.nan):
                        item = _sequence()
                        item["heldout_y"][66:] = poison
                        item["heldout_predictions"][66:] = -poison
                        with contextlib.redirect_stdout(io.StringIO()):
                            result = tuning._run_single_trial(config, item)
                        self.assertEqual(result, reference)

    def test_validation_outcomes_change_scores_but_not_first_fit_bandwidth(self):
        config = _config(normalize=True)
        config.model.update_with_test = False
        baseline = _sequence()
        changed = _sequence()
        changed["heldout_y"][56:66] += 100.0
        with contextlib.redirect_stdout(io.StringIO()):
            first = tuning._run_single_trial(config, baseline)
            second = tuning._run_single_trial(config, changed)
        self.assertEqual(first["selected_bandwidths"], second["selected_bandwidths"])
        self.assertGreater(second["selection_score"], first["selection_score"])
        self.assertEqual(second["pair_metrics"][str(PAIR)]["avg_coverage"], 0.0)

    def _run_main(self, directory, ratio, *, item=None, grid=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / "lr_toy_data.pkl"
        with artifact.open("wb") as stream:
            pickle.dump({"example": _sequence() if item is None else item}, stream)
        config = _config(normalize=True)
        del config["tuning"]
        config.data.data_path = str(artifact)
        config.saving_dir = str(directory / "normal")
        config.plotting = {"plotting": False}
        base_path, grid_path = directory / "base.yaml", directory / "grid.yaml"
        OmegaConf.save(config, base_path)
        OmegaConf.save(OmegaConf.create({
            "grid": grid or {"model.past_window": [2, 3]},
            "tuning": {
                "num_sequences": 1, "delta_threshold": -1.0,
                "model_selection_valid_ratio": ratio,
            },
        }), grid_path)
        args = SimpleNamespace(
            base_config=base_path, grid_config=grid_path, save_dir=directory / "tuning",
            num_cores=1, sequence_key=None, sequence_index=0, seed=12, top_k=2,
        )
        with patch.object(tuning, "parse_args", return_value=args), contextlib.redirect_stdout(io.StringIO()):
            tuning.main()
        with (args.save_dir / "tuning_results.pkl").open("rb") as stream:
            return pickle.load(stream)

    def test_main_loads_validation_fraction_from_tuning_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            for ratio, history, validation in ((0.15, 56, 10), (0.30, 46, 20)):
                with self.subTest(ratio=ratio), self._capture(tuning) as estimators:
                    payload = self._run_main(Path(directory) / str(ratio), ratio)
                    self.assertEqual(len(estimators), 2)
                    for estimator, record in zip(estimators, payload["all_trials"]):
                        self.assertEqual(estimator.call["calibration_size"], history)
                        self.assertEqual(estimator.call["test_size"], validation)
                        self.assertEqual(len(estimator.call["residuals"]), 66)
                        self.assertEqual(record["resolved_config"]["tuning"]["model_selection_valid_ratio"], ratio)
                        result = record["result"]["sequence_results"]["example"]
                        self.assertEqual(result["evaluation_start"], history)
                        self.assertEqual(result["evaluation_end"], 66)
                        self.assertFalse(result["final_test_evaluated"])

    def test_real_main_ranking_ignores_final_test_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            clean = self._run_main(Path(directory) / "clean", 0.15)
            poisoned_item = _sequence()
            poisoned_item["heldout_y"][66:] = np.nan
            poisoned_item["heldout_predictions"][66:] = np.inf
            poisoned = self._run_main(Path(directory) / "poisoned", 0.15, item=poisoned_item)
            self.assertEqual(
                [record["result"] for record in clean["all_trials"]],
                [record["result"] for record in poisoned["all_trials"]],
            )
            self.assertTrue(clean["top_trials"])
            self.assertEqual(
                [record["trial_index"] for record in clean["top_trials"]],
                [record["trial_index"] for record in poisoned["top_trials"]],
            )

    def test_checked_in_configs_keep_validation_only_in_tuning_yaml(self):
        paths = sorted((REPO_ROOT / "configs" / "kowcpi_configs").glob("*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(config=path.name):
                config = OmegaConf.load(path)
                if "grid" in config:
                    grid, settings = tuning.load_grid(path)
                    self.assertTrue(grid)
                    self.assertGreater(settings["model_selection_valid_ratio"], 0.0)
                    self.assertLess(settings["model_selection_valid_ratio"], 1.0)
                else:
                    self.assertNotIn("train_ratio", config.data)
                    self.assertNotIn("valid_ratio", config.data)
                    self.assertNotIn("validation_ratio", config.data)
                    self.assertNotIn("test_ratio", config.data)
                    self.assertGreater(config.data.calibration_ratio, 0.0)
                    self.assertLess(config.data.calibration_ratio, 1.0)
                    self.assertNotIn("tuning", config)

    def test_tuning_cli_help_works_as_module_and_direct_script(self):
        for command in (
            [sys.executable, "-m", "sbatch.sbatch_run_tuning.run_kowcpi_tuning", "--help"],
            [sys.executable, str(REPO_ROOT / "sbatch" / "sbatch_run_tuning" / "run_kowcpi_tuning.py"), "--help"],
        ):
            with self.subTest(command=command):
                completed = subprocess.run(
                    command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertIn("--grid-config", completed.stdout)
                self.assertIn("--num-cores", completed.stdout)


if __name__ == "__main__":
    unittest.main()
