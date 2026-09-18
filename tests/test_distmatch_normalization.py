"""Check upstream target scaling, causal fit boundaries, and raw-unit reporting."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from baselines.distmatch.config import validate_config
from baselines.distmatch.data import prepare_sequence
from baselines.distmatch.model import DistMatchResidualIntervalEstimator
from baselines.distmatch.run_distmatch import evaluate_sequence, sequence_seed


REPO_ROOT = Path(__file__).resolve().parents[1]
PAIRS = ((0.05, 0.95), (0.1, 0.9))


def _config(normalize=True, mode="upstream_target"):
    return {
        "seed": 37,
        "num_cores": 1,
        "threads_per_worker": 1,
        "show_progress": False,
        "data": {
            "train_ratio": 0.5, "valid_ratio": 0.25, "test_ratio": 0.25,
            "normalize": normalize, "normalization_mode": mode,
        },
        "model": {
            "past_window_len": 3, "match_threshold": 0.5,
            "n_trees": 2, "qrf_n_estimators": 2, "qrf_max_depth": 2,
            "beta_bins": 3, "target_quantiles": [list(pair) for pair in PAIRS],
        },
    }


def _artifact():
    index = np.arange(24, dtype=np.float64)
    predictions = 40 + 0.7 * index
    return {
        "train_y": np.linspace(-30, 20, 8)[:, None],
        "heldout_y": predictions + 2 + 1.7 * np.sin(0.8 * index),
        "heldout_predictions": predictions[:, None],
        # Target normalization never needs contemporaneous covariates.
        "heldout_x": np.full((24, 2), np.nan),
    }


def _target_stats(item, start):
    history = torch.cat([
        torch.as_tensor(item["train_y"], dtype=torch.float64).reshape(-1),
        torch.as_tensor(item["heldout_y"][:start], dtype=torch.float64),
    ])
    mean = history.mean().item()
    std = history.std().item()  # Upstream torch.std uses sample correction=1.
    return mean, std if std != 0 else 1.0


class DistMatchNormalizationTests(unittest.TestCase):
    def test_config_preserves_legacy_default_and_validates_mode(self):
        config = _config()
        del config["data"]["normalization_mode"]
        self.assertEqual(validate_config(config)["data"]["normalization_mode"], "residual_inputs")
        for mode in ("upstream_target", "residual_inputs"):
            with self.subTest(mode=mode):
                self.assertEqual(validate_config(_config(mode=mode))["data"]["normalization_mode"], mode)
        for mode in ("target", "", True, None, 1):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "normalization_mode"):
                validate_config(_config(mode=mode))

    def test_preparation_matches_independent_torch_sample_statistics(self):
        item = _artifact()
        residuals = item["heldout_y"] - item["heldout_predictions"][:, 0]
        for split, start, end in (("validation", 12, 18), ("test", 18, 24)):
            with self.subTest(split=split):
                prepared = prepare_sequence(item, _config(), split=split)
                mean, std = _target_stats(item, start)
                metadata = prepared["normalization"]
                self.assertEqual(metadata["mode"], "upstream_target")
                self.assertAlmostEqual(metadata["target_mean"], mean)
                self.assertAlmostEqual(metadata["target_std"], std)
                self.assertEqual(metadata["fit_size"], len(item["train_y"]) + start)
                self.assertEqual(metadata["heldout_fit_end"], start)
                self.assertEqual(metadata["ddof"], 1)
                self.assertIsInstance(metadata["source"], str)
                self.assertTrue(metadata["source"])
                np.testing.assert_allclose(prepared["train_residuals"], residuals[:12] / std)
                np.testing.assert_allclose(prepared["warmup_residuals"], residuals[12:start] / std)
                np.testing.assert_allclose(prepared["residuals"], residuals[start:end] / std)
                np.testing.assert_array_equal(prepared["y"], item["heldout_y"][start:end])
                np.testing.assert_array_equal(prepared["predictions"], item["heldout_predictions"][start:end, 0])
                np.testing.assert_array_equal(prepared["target_indices"], np.arange(start, end))

    def test_scaled_residuals_match_executed_upstream_normalize_method(self):
        source_path = REPO_ROOT / "dist_match_conformal" / "code" / "loader" / "dataset.py"
        if not source_path.is_file():
            self.skipTest("Optional upstream checkout is not present")
        # Execute the actual upstream method without loading its unrelated
        # dataset classes or optional dependencies.
        source = ast.parse(source_path.read_text(encoding="utf-8"))
        dataset = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "TsDataset")
        method = next(node for node in dataset.body if isinstance(node, ast.FunctionDef) and node.name == "_normalize")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
        normalize = namespace["_normalize"]
        item = _artifact()
        raw_targets = torch.cat([
            torch.as_tensor(item["train_y"], dtype=torch.float64).reshape(-1),
            torch.as_tensor(item["heldout_y"], dtype=torch.float64),
        ])
        raw_predictions = torch.as_tensor(item["heldout_predictions"], dtype=torch.float64).reshape(-1)
        for split, start, end in (("validation", 12, 18), ("test", 18, 24)):
            with self.subTest(split=split):
                normalized_y, mean, std = normalize(SimpleNamespace(test_step=8 + start), raw_targets)
                normalized_predictions = (raw_predictions - mean) / std
                expected = (normalized_y[8:] - normalized_predictions).numpy()
                prepared = prepare_sequence(item, _config(), split=split)
                actual = np.concatenate([
                    prepared["train_residuals"], prepared["warmup_residuals"], prepared["residuals"],
                ])
                np.testing.assert_allclose(actual, expected[:end], rtol=1e-12, atol=1e-14)

    def test_runner_restores_raw_units_against_manually_scaled_estimator(self):
        item, config = _artifact(), _config()
        _, std = _target_stats(item, 18)
        residuals = (item["heldout_y"] - item["heldout_predictions"][:, 0]) / std
        options = {name: value for name, value in config["model"].items() if name != "target_quantiles"}
        reference = DistMatchResidualIntervalEstimator(seed=sequence_seed(config["seed"], "station"), **options)
        reference.fit(residuals[:12], normalize=False)
        for residual in residuals[12:18]:
            reference.observe(float(residual))
        expected = {pair: [] for pair in PAIRS}
        for residual in residuals[18:]:
            intervals = reference.predict_intervals(PAIRS)
            for pair in PAIRS:
                lower, upper, _ = intervals[pair]
                expected[pair].append((lower * std, upper * std))
            reference.observe(float(residual))

        actual = evaluate_sequence("station", item, config)
        self.assertEqual(actual["metadata"]["normalization"], prepare_sequence(item, config)["normalization"])
        self.assertEqual(actual["metadata"]["input_mean"], 0)
        self.assertEqual(actual["metadata"]["input_std"], 1)
        for pair, quantiles in expected.items():
            with self.subTest(pair=pair):
                result = actual["evaluation_results"][pair]
                lower_residual, upper_residual = np.asarray(quantiles).T
                np.testing.assert_allclose(result["lower_residual_quantile"], lower_residual, rtol=1e-6)
                np.testing.assert_allclose(result["upper_residual_quantile"], upper_residual, rtol=1e-6)
                predictions = item["heldout_predictions"][18:, 0]
                targets = item["heldout_y"][18:]
                lower, upper = predictions + lower_residual, predictions + upper_residual
                np.testing.assert_allclose(result["lower_interval"], lower, rtol=1e-7)
                np.testing.assert_allclose(result["upper_interval"], upper, rtol=1e-7)
                np.testing.assert_array_equal(result["target_y"], targets)
                np.testing.assert_array_equal(result["target_predictions"], predictions)
                np.testing.assert_array_equal(result["coverage"], (lower <= targets) & (targets <= upper))
                np.testing.assert_allclose(result["interval_width"], upper - lower, rtol=1e-6)
                alpha = 1 - (pair[1] - pair[0])
                score = upper - lower + (2 / alpha) * (np.maximum(lower - targets, 0) + np.maximum(targets - upper, 0))
                np.testing.assert_allclose(result["winkler_score"], score, rtol=1e-6)

    def test_current_and_future_targets_do_not_change_scaler_or_first_interval(self):
        for split, start in (("validation", 12), ("test", 18)):
            with self.subTest(split=split):
                item = _artifact()
                baseline = evaluate_sequence("station", item, _config(), split=split)
                changed = copy.deepcopy(item)
                changed["heldout_y"][start:] += 1000 + np.arange(24 - start) * 100
                actual = evaluate_sequence("station", changed, _config(), split=split)
                self.assertEqual(actual["metadata"]["normalization"], baseline["metadata"]["normalization"])
                for pair in PAIRS:
                    for field in ("lower_interval", "upper_interval"):
                        self.assertEqual(actual["evaluation_results"][pair][field][0], baseline["evaluation_results"][pair][field][0])

    def test_reserved_test_suffix_may_be_nonfinite_during_validation(self):
        item = _artifact()
        baseline = evaluate_sequence("station", item, _config(), split="validation")
        item["heldout_y"][18:] = np.nan
        item["heldout_predictions"][18:] = np.inf
        actual = evaluate_sequence("station", item, _config(), split="validation")
        self.assertEqual(actual["metadata"]["normalization"], baseline["metadata"]["normalization"])
        self.assertEqual(actual["evaluation_results"], baseline["evaluation_results"])

    def test_upstream_scaling_requires_valid_training_targets_without_fallback(self):
        item = _artifact()
        del item["train_y"]
        with self.assertRaisesRegex(ValueError, "train_y"):
            prepare_sequence(item, _config())
        for train_y in ([], [np.nan], [np.inf], [[1, 2], [3, 4]], ["invalid"]):
            with self.subTest(train_y=train_y), self.assertRaisesRegex(ValueError, "train_y"):
                item = _artifact()
                item["train_y"] = train_y
                prepare_sequence(item, _config())

    def test_constant_target_history_uses_unit_scale(self):
        item = _artifact()
        item["train_y"][:] = 7
        item["heldout_y"][:] = 7
        prepared = prepare_sequence(item, _config())
        self.assertEqual(prepared["normalization"]["target_mean"], 7)
        self.assertEqual(prepared["normalization"]["target_std"], 1)
        residuals = item["heldout_y"] - item["heldout_predictions"][:, 0]
        np.testing.assert_array_equal(prepared["train_residuals"], residuals[:12])
        actual = evaluate_sequence("station", item, _config())
        for result in actual["evaluation_results"].values():
            self.assertTrue(np.isfinite(result["lower_interval"]).all())
            self.assertTrue(np.isfinite(result["upper_interval"]).all())

    def test_legacy_input_scaling_ignores_training_targets(self):
        config = _config()
        del config["data"]["normalization_mode"]
        item = _artifact()
        item["train_y"][:] = np.nan
        baseline = evaluate_sequence("station", item, config)
        del item["train_y"]
        actual = evaluate_sequence("station", item, config)
        self.assertEqual(actual["evaluation_results"], baseline["evaluation_results"])
        residuals = item["heldout_y"] - item["heldout_predictions"][:, 0]
        self.assertAlmostEqual(actual["metadata"]["input_mean"], residuals[:12].mean())
        self.assertAlmostEqual(actual["metadata"]["input_std"], residuals[:12].std())
        np.testing.assert_array_equal(prepare_sequence(item, config)["train_residuals"], residuals[:12])

    def test_normalize_false_disables_target_scaling_and_training_history_requirement(self):
        item = _artifact()
        del item["train_y"]
        config = _config(normalize=False)
        prepared = prepare_sequence(item, config)
        residuals = item["heldout_y"] - item["heldout_predictions"][:, 0]
        np.testing.assert_array_equal(prepared["train_residuals"], residuals[:12])
        np.testing.assert_array_equal(prepared["warmup_residuals"], residuals[12:18])
        np.testing.assert_array_equal(prepared["residuals"], residuals[18:])
        actual = evaluate_sequence("station", item, config)
        expected = evaluate_sequence("station", item, _config(normalize=False, mode="residual_inputs"))
        self.assertEqual(actual["evaluation_results"], expected["evaluation_results"])
        self.assertEqual(actual["metadata"]["input_mean"], 0)
        self.assertEqual(actual["metadata"]["input_std"], 1)


if __name__ == "__main__":
    unittest.main()
