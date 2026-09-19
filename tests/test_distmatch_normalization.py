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


def _config(normalize_residual=True):
    return {
        "seed": 37,
        "num_cores": 1,
        "threads_per_worker": 1,
        "show_progress": False,
        "data": {
            "train_ratio": 0.5, "valid_ratio": 0.25, "test_ratio": 0.25,
            "normalize_residual": normalize_residual,
        },
        "model": {
            "past_window_len": 3, "match_threshold": 0.5,
            "n_trees": 2, "qrf_n_estimators": 2, "qrf_max_depth": 2,
            "beta_bins": 3, "target_quantiles": [list(pair) for pair in PAIRS],
        },
    }


def _artifact(with_train_y=True):
    index = np.arange(24, dtype=np.float64)
    predictions = 40 + 0.7 * index
    item = {
        "heldout_y": predictions + 2 + 1.7 * np.sin(0.8 * index),
        "heldout_predictions": predictions[:, None],
        # Target normalization never needs contemporaneous covariates.
        "heldout_x": np.full((24, 2), np.nan),
    }
    if with_train_y:
        item["train_y"] = np.linspace(-30, 20, 8)[:, None]
    return item


def _target_stats(item, start):
    history = torch.as_tensor(item["heldout_y"][:start], dtype=torch.float64).reshape(-1)
    if "train_y" in item:
        history = torch.cat([
            torch.as_tensor(item["train_y"], dtype=torch.float64).reshape(-1), history,
        ])
    mean = history.mean().item()
    std = history.std().item()  # Upstream torch.std uses sample correction=1.
    return mean, std if std != 0 else 1.0


class DistMatchNormalizationTests(unittest.TestCase):
    def test_normalization_defaults_to_true_and_requires_a_boolean(self):
        config = _config()
        del config["data"]["normalize_residual"]
        self.assertIs(validate_config(config)["data"]["normalize_residual"], True)
        default = prepare_sequence(_artifact(), config)
        explicit = prepare_sequence(_artifact(), _config(True))
        self.assertEqual(default["normalization"], explicit["normalization"])
        np.testing.assert_array_equal(default["train_residuals"], explicit["train_residuals"])
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                self.assertIs(validate_config(_config(enabled))["data"]["normalize_residual"], enabled)
        for invalid in ("true", "false", "", None, 0, 1, [], {}):
            for path in ("validate_config", "prepare_sequence"):
                with self.subTest(invalid=invalid, path=path), self.assertRaisesRegex(ValueError, "normalize_residual"):
                    if path == "validate_config":
                        validate_config(_config(invalid))
                    else:
                        prepare_sequence(_artifact(), _config(invalid))

    def test_obsolete_normalization_settings_require_explicit_migration(self):
        for old_settings in (
            {"normalize": True}, {"normalize": False},
            {"normalization_mode": "upstream_target"},
            {"normalization_mode": "residual_inputs"},
            {"normalize": False, "normalization_mode": "residual_inputs"},
        ):
            for include_new_key in (True, False):
                config = _config()
                if not include_new_key:
                    del config["data"]["normalize_residual"]
                config["data"].update(old_settings)
                for path in ("validate_config", "prepare_sequence"):
                    with self.subTest(old_settings=old_settings, include_new_key=include_new_key, path=path), \
                            self.assertRaisesRegex(ValueError, "normalize_residual"):
                        if path == "validate_config":
                            validate_config(config)
                        else:
                            prepare_sequence(_artifact(), config)

    def test_preparation_matches_independent_torch_sample_statistics(self):
        for with_train_y in (True, False):
            item = _artifact(with_train_y)
            residuals = item["heldout_y"] - item["heldout_predictions"][:, 0]
            for split, start, end in (("validation", 12, 18), ("test", 18, 24)):
                with self.subTest(with_train_y=with_train_y, split=split):
                    prepared = prepare_sequence(item, _config(), split=split)
                    mean, std = _target_stats(item, start)
                    metadata = prepared["normalization"]
                    self.assertIs(metadata["enabled"], True)
                    self.assertAlmostEqual(metadata["target_mean"], mean)
                    self.assertAlmostEqual(metadata["target_std"], std)
                    self.assertEqual(metadata["fit_size"], len(item.get("train_y", [])) + start)
                    self.assertEqual(metadata["heldout_fit_end"], start)
                    self.assertEqual(metadata["ddof"], 1)
                    source = "heldout_y[:evaluation_start]"
                    self.assertEqual(metadata["source"], "train_y + " + source if with_train_y else source)
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
        for with_train_y in (True, False):
            for normalize_residual in (True, False):
                with self.subTest(with_train_y=with_train_y, normalize_residual=normalize_residual):
                    self._assert_runner_restores_raw_units(_artifact(with_train_y), normalize_residual)

    def _assert_runner_restores_raw_units(self, item, normalize_residual):
        config = _config(normalize_residual)
        std = _target_stats(item, 18)[1] if normalize_residual else 1.0
        residuals = (item["heldout_y"] - item["heldout_predictions"][:, 0]) / std
        options = {name: value for name, value in config["model"].items() if name != "target_quantiles"}
        reference = DistMatchResidualIntervalEstimator(seed=sequence_seed(config["seed"], "station"), **options)
        reference.fit(residuals[:12])
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
        self.assertIs(actual["metadata"]["normalize_residual"], normalize_residual)
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
        for with_train_y in (True, False):
            for split, start in (("validation", 12), ("test", 18)):
                with self.subTest(with_train_y=with_train_y, split=split):
                    item = _artifact(with_train_y)
                    baseline = evaluate_sequence("station", item, _config(), split=split)
                    changed = copy.deepcopy(item)
                    changed["heldout_y"][start:] += 1000 + np.arange(24 - start) * 100
                    actual = evaluate_sequence("station", changed, _config(), split=split)
                    self.assertEqual(actual["metadata"]["normalization"], baseline["metadata"]["normalization"])
                    for pair in PAIRS:
                        for field in ("lower_interval", "upper_interval"):
                            self.assertEqual(actual["evaluation_results"][pair][field][0], baseline["evaluation_results"][pair][field][0])

    def test_reserved_test_suffix_may_be_nonfinite_during_validation(self):
        for with_train_y in (True, False):
            with self.subTest(with_train_y=with_train_y):
                item = _artifact(with_train_y)
                baseline = evaluate_sequence("station", item, _config(), split="validation")
                item["heldout_y"][18:] = np.nan
                item["heldout_predictions"][18:] = np.inf
                actual = evaluate_sequence("station", item, _config(), split="validation")
                self.assertEqual(actual["metadata"]["normalization"], baseline["metadata"]["normalization"])
                self.assertEqual(actual["evaluation_results"], baseline["evaluation_results"])

    def test_upstream_scaling_rejects_invalid_provided_training_targets(self):
        for train_y in (None, [], [np.nan], [np.inf], [[1, 2], [3, 4]], ["invalid"]):
            with self.subTest(train_y=train_y), self.assertRaisesRegex(ValueError, "train_y"):
                item = _artifact()
                item["train_y"] = train_y
                prepare_sequence(item, _config())

    def test_constant_target_history_uses_unit_scale(self):
        for with_train_y in (True, False):
            with self.subTest(with_train_y=with_train_y):
                item = _artifact(with_train_y)
                if with_train_y:
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

    def test_fallback_requires_two_pre_evaluation_targets(self):
        item = {name: values[:3] for name, values in _artifact(False).items()}
        # With three observations validation starts at index 1, test at index 2.
        with self.assertRaisesRegex(ValueError, "at least two outcomes"):
            prepare_sequence(item, _config(), split="validation")
        prepared = prepare_sequence(item, _config(), split="test")
        mean, std = _target_stats(item, 2)
        self.assertEqual(prepared["normalization"]["fit_size"], 2)
        self.assertAlmostEqual(prepared["normalization"]["target_mean"], mean)
        self.assertAlmostEqual(prepared["normalization"]["target_std"], std)

    def test_disabled_normalization_ignores_training_targets(self):
        config = _config(False)
        item = _artifact()
        item["train_y"][:] = np.nan
        baseline = evaluate_sequence("station", item, config)
        del item["train_y"]
        actual = evaluate_sequence("station", item, config)
        self.assertEqual(actual["evaluation_results"], baseline["evaluation_results"])
        residuals = item["heldout_y"] - item["heldout_predictions"][:, 0]
        self.assertIs(actual["metadata"]["normalize_residual"], False)
        self.assertIs(actual["metadata"]["normalization"]["enabled"], False)
        np.testing.assert_array_equal(prepare_sequence(item, config)["train_residuals"], residuals[:12])

    def test_disabled_normalization_preserves_raw_residuals_and_unit_scale(self):
        item = _artifact(False)
        config = _config(False)
        prepared = prepare_sequence(item, config)
        residuals = item["heldout_y"] - item["heldout_predictions"][:, 0]
        np.testing.assert_array_equal(prepared["train_residuals"], residuals[:12])
        np.testing.assert_array_equal(prepared["warmup_residuals"], residuals[12:18])
        np.testing.assert_array_equal(prepared["residuals"], residuals[18:])
        self.assertIs(prepared["normalization"]["enabled"], False)
        self.assertEqual(prepared["normalization"]["target_mean"], 0)
        self.assertEqual(prepared["normalization"]["target_std"], 1)


if __name__ == "__main__":
    unittest.main()
