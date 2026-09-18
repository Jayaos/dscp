"""The SPCI tuner and final runner must use the same optimized interval rule."""

import contextlib
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from baselines.spci import run_spci as ordinary_runner
from sbatch.sbatch_run_tuning import run_spci_tuning as tuning
from utils.reporting import compute_winkler_score


PAIR = (0.625, 0.125)  # Asymmetric fixed tails; nominal coverage is exactly one half.
CURVE_QUANTILES = np.linspace(0.0, 1.0, 9)
CURVES = np.array([
    [0, 1, 2, 3, 4, 10, 20, 30, 40],
    [0, 5, 10, 11, 12, 13, 14, 19, 24],
    [0, 10, 20, 30, 36, 37, 38, 39, 40],
], dtype=float).T


class KnownQuantileForest:
    def __init__(self, quantiles, expected_fit_count):
        self.quantiles = np.asarray(quantiles)
        self.expected_fit_count = expected_fit_count
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self, features, targets):
        self.fit_calls += 1
        self.fit_features = np.asarray(features).copy()
        self.fit_targets = np.asarray(targets).copy()
        if len(targets) != self.expected_fit_count:
            raise AssertionError("Factory sample count and fitting data disagree.")
        return self

    def predict(self, features):
        self.predict_calls += 1
        self.predict_features = np.asarray(features).copy()
        columns = np.column_stack([
            np.interp(self.quantiles, CURVE_QUANTILES, curve) for curve in CURVES.T
        ])
        return columns[:, np.arange(len(features)) % 3].astype(np.float32)


class SPCIBetaIntegrationTests(unittest.TestCase):
    def _configs(self, directory, optimize_beta=True):
        directory = Path(directory)
        index = np.arange(40, dtype=np.float32)
        predictions = 10 + 0.1 * index
        residuals = np.resize(np.array([2, 12, 38], dtype=np.float32), len(index))
        artifact = directory / "lr_toy_data.pkl"
        with artifact.open("wb") as stream:
            pickle.dump({"series": {
                "heldout_x": np.column_stack((index, index * 2)),
                "heldout_y": predictions + residuals,
                "heldout_predictions": predictions,
            }}, stream)
        base = OmegaConf.create({
            "seed": 17,
            "data": {"data_path": str(artifact), "train_ratio": 0.6, "normalize": False},
            "model": {
                "target_quantiles": [PAIR], "optimize_beta": optimize_beta, "beta_bins": 5,
                "window_size": 3, "prediction_step": 1,
                "n_estimators": 10, "max_depth": 2, "criterion": "squared_error",
            },
            "plotting": {"plotting": False},
            "saving_dir": str(directory / "ordinary"),
        })
        base_path, grid_path = directory / "base.yaml", directory / "grid.yaml"
        OmegaConf.save(base, base_path)
        OmegaConf.save(OmegaConf.create({
            "grid": {"model.window_size": [3]},
            "tuning": {"model_selection_valid_ratio": 0.25, "delta_threshold": -1.0},
        }), grid_path)
        return base_path, grid_path

    @contextlib.contextmanager
    def _capture_runner(self, runner):
        forests = []

        def factory(config, n_train_samples, quantiles):
            forest = KnownQuantileForest(quantiles, n_train_samples)
            forests.append(forest)
            return forest

        with patch.object(runner, "build_quantile_forest", side_effect=factory), \
                patch.object(runner, "compute_winkler_score", wraps=compute_winkler_score) as score, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield forests, score

    def _check_selection(self, metadata, count):
        np.testing.assert_array_equal(metadata["selected_beta"], np.resize([0, 0.25, 0.5], count))
        np.testing.assert_array_equal(metadata["lower_quantile_levels"], metadata["selected_beta"])
        np.testing.assert_array_equal(metadata["upper_quantile_levels"], np.resize([0.5, 0.75, 1], count))
        self.assertEqual(metadata["nominal_alpha"], 0.5)
        self.assertTrue(metadata["optimize_beta"])

    def _check_forest(self, forest, fitting, predicting):
        np.testing.assert_array_equal(forest.quantiles, CURVE_QUANTILES)
        self.assertEqual(forest.fit_calls, 1)
        self.assertEqual(forest.predict_calls, 1)
        self.assertEqual(len(forest.fit_targets), fitting)
        self.assertEqual(len(forest.predict_features), predicting)

    def _check_scoring(self, score, count):
        score.assert_called_once()
        upper, lower, _, _, alpha = score.call_args.args[:5]
        self.assertEqual(alpha, 0.5)
        np.testing.assert_array_equal(lower, np.resize([0, 10, 36], count))
        np.testing.assert_array_equal(upper, np.resize([4, 14, 40], count))

    def test_optimized_tuning_exports_rule_and_final_run_reuses_it_without_refitting_online(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory)
            output = Path(directory) / "tuning"
            with self._capture_runner(tuning) as (forests, score):
                payload = tuning.run_tuning(base_path, grid_path, output, seed=41)
            self.assertEqual(len(forests), 1)
            self._check_forest(forests[0], fitting=15, predicting=6)
            self._check_scoring(score, count=6)
            result = payload["all_trials"][0]["result"]["sequence_results"]["series"]
            self._check_selection(result["interval_selection"][str(PAIR)], count=6)
            self.assertTrue(np.isfinite(result["selection_score"]))
            self.assertFalse(result["final_test_evaluated"])

            saved_path = output / "trial_0001" / "resolved_config.yaml"
            saved = OmegaConf.load(saved_path)
            self.assertTrue(saved.model.optimize_beta)
            self.assertEqual(saved.model.beta_bins, 5)
            self.assertEqual(saved.seed, 42)
            with self._capture_runner(ordinary_runner) as (forests, score):
                ordinary_runner.run_spci_experiment(saved_path)
            self.assertEqual(len(forests), 1)
            self._check_forest(forests[0], fitting=21, predicting=16)
            self._check_scoring(score, count=16)
            with (Path(saved.saving_dir) / "log.pkl").open("rb") as stream:
                log = pickle.load(stream)
            result = log["series"]["evaluation_results"][PAIR]
            self._check_selection(result, count=16)
            np.testing.assert_array_equal(result["lower_interval"], np.resize([0, 10, 36], 16))
            np.testing.assert_array_equal(result["upper_interval"], np.resize([4, 14, 40], 16))
            self.assertTrue(np.isfinite(result["winkler_score"]).all())

    def test_disabled_mode_keeps_original_asymmetric_quantiles_and_winkler_penalties(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, grid_path = self._configs(directory, optimize_beta=False)
            with self._capture_runner(tuning) as (forests, score):
                tuning.run_tuning(base_path, grid_path, Path(directory) / "tuning")
            self.assertEqual(len(forests), 1)
            np.testing.assert_array_equal(forests[0].quantiles, sorted(PAIR))
            score.assert_called_once()
            self.assertEqual(score.call_args.args[4], PAIR)
            with self._capture_runner(ordinary_runner) as (forests, score):
                ordinary_runner.run_spci_experiment(base_path)
            self.assertEqual(len(forests), 1)
            np.testing.assert_array_equal(forests[0].quantiles, sorted(PAIR))
            score.assert_called_once()
            self.assertEqual(score.call_args.args[4], PAIR)


if __name__ == "__main__":
    unittest.main()
