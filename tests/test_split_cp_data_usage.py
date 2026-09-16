import copy
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf

from baselines.split_cp.data import prepare_sequence, split_boundaries
from baselines.split_cp.run_split_cp import (
    evaluate_sequence,
    evaluate_sequences,
    run_split_cp,
    validate_config,
)
from utils.plotting import _resolve_logged_interval_endpoints


def _config(calibration_ratio=0.66, test_ratio=0.34):
    return OmegaConf.create({
        "data": {"calibration_ratio": calibration_ratio, "test_ratio": test_ratio},
        "model": {"target_quantiles": [[0.1, 0.9], [0.05, 0.95]], "prediction_step": 1},
        "plotting": {"plotting": False},
    })


def _artifact(length=100):
    index = np.arange(length, dtype=np.float64)
    predictions = 1000.0 + 0.25 * index
    residuals = (-1.0) ** index * (1.0 + index % 19)
    return {"heldout_y": predictions + residuals, "heldout_predictions": predictions[:, None]}


class SplitCPDataTests(unittest.TestCase):
    def test_calibration_test_boundaries_use_floor_and_exhaust_the_series(self):
        for length, ratio, expected in ((100, 0.66, 66), (101, 0.66, 66), (100, 0.29, 29), (101, 0.8, 80)):
            with self.subTest(length=length, ratio=ratio):
                result = split_boundaries(length, ratio, 1.0 - ratio)
                self.assertEqual(result["calibration_start"], 0)
                self.assertEqual(result["calibration_end"], expected)
                self.assertEqual(result["calibration_size"], expected)
                self.assertEqual(result["test_start"], expected)
                self.assertEqual(result["test_size"], length - expected)

    def test_explicit_boundary_preserves_exact_comparison_indices(self):
        result = split_boundaries(101, test_start=67)
        self.assertEqual(result["calibration_end"], 67)
        self.assertEqual(result["test_start"], 67)
        self.assertEqual(result["test_size"], 34)

    def test_invalid_splits_are_rejected(self):
        for length, calibration, test in ((0, .66, .34), (1, .66, .34), (100, 0, 1), (100, 1, 0), (100, -.1, 1.1), (100, .5, .4), (100, np.nan, .34)):
            with self.subTest(length=length, calibration=calibration, test=test), self.assertRaises(ValueError):
                split_boundaries(length, calibration, test)
        for start in (0, 101, -1, 2.5, True):
            with self.subTest(start=start), self.assertRaises(ValueError):
                split_boundaries(101, test_start=start)

    def test_mixed_shapes_have_aligned_scalar_residuals_and_need_no_covariates(self):
        item = _artifact(101)
        prepared = prepare_sequence(item, _config())
        expected = item["heldout_y"] - item["heldout_predictions"][:, 0]
        np.testing.assert_array_equal(prepared["calibration_residuals"], expected[:66])
        np.testing.assert_array_equal(prepared["target_indices"], np.arange(66, 101))
        np.testing.assert_array_equal(prepared["predictions"], item["heldout_predictions"][66:, 0])
        np.testing.assert_array_equal(prepared["y"], item["heldout_y"][66:])

    def test_explicit_global_and_per_series_start_overrides_ratio(self):
        config = _config()
        for start in (67, {"station": 67, "another": 80}):
            with self.subTest(start=start):
                config.data.test_start = start
                result = prepare_sequence(_artifact(101), config, key="station")
                np.testing.assert_array_equal(result["target_indices"], np.arange(67, 101))
                self.assertEqual(len(result["calibration_residuals"]), 67)

    def test_bad_lengths_multioutput_and_nonfinite_observations_are_rejected(self):
        for predictions in (np.zeros(99), np.zeros((100, 2)), np.full(100, np.inf)):
            item = _artifact()
            item["heldout_predictions"] = predictions
            with self.subTest(shape=predictions.shape), self.assertRaises(ValueError):
                prepare_sequence(item, _config())
        for index in (0, 66, 99):
            item = _artifact()
            item["heldout_y"][index] = np.nan
            with self.subTest(index=index), self.assertRaises(ValueError):
                prepare_sequence(item, _config())


class SplitCPRunnerTests(unittest.TestCase):
    def test_ratio_defaults_and_invalid_values_are_validated_before_decimal_arithmetic(self):
        config = validate_config({"data": {"calibration_ratio": 0.8}})
        self.assertEqual(config["data"]["test_ratio"], 0.2)
        for value in (True, None, "invalid", float("nan"), float("inf"), 0, 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config({"data": {"calibration_ratio": value}})

    def assert_same_intervals(self, expected, actual):
        self.assertEqual(expected["evaluation_results"].keys(), actual["evaluation_results"].keys())
        self.assertEqual(expected["metadata"]["quantile_radii"], actual["metadata"]["quantile_radii"])
        for pair, reference in expected["evaluation_results"].items():
            for field in ("lower_interval", "upper_interval", "lower_residual_quantile", "upper_residual_quantile"):
                with self.subTest(pair=pair, field=field):
                    np.testing.assert_array_equal(actual["evaluation_results"][pair][field], reference[field])

    def test_all_test_labels_are_excluded_from_calibration_and_interval_updates(self):
        item = _artifact()
        baseline = evaluate_sequence("station", item, _config())
        changed = copy.deepcopy(item)
        changed["heldout_y"][66:] += 100000.0
        actual = evaluate_sequence("station", changed, _config())
        self.assert_same_intervals(baseline, actual)
        self.assertNotEqual(baseline["evaluation_results"][(.1, .9)]["avg_coverage"], actual["evaluation_results"][(.1, .9)]["avg_coverage"])

    def test_known_quantiles_metrics_metadata_and_plotting_use_original_units(self):
        predictions = 1000.0 + np.arange(25, dtype=float)
        residuals = np.concatenate([np.arange(1.0, 20.0), [-30.0, -18.0, 0.0, 10.0, 18.0, 40.0]])
        item = {"heldout_y": predictions + residuals, "heldout_predictions": predictions[:, None]}
        config = _config()
        config.data.test_start = 19
        result = evaluate_sequence("station", item, config)
        metadata = result["metadata"]
        self.assertEqual(metadata["calibration_size"], 19)
        self.assertEqual(metadata["interval_scale"], "original_response")
        self.assertEqual(metadata["residual_quantile_scale"], "original_residual")
        self.assertEqual(metadata["calibration_update"], "fixed")
        np.testing.assert_array_equal(metadata["target_indices"], np.arange(19, 25))
        for pair, radius in (((.1, .9), 16.0), ((.05, .95), 18.0)):
            metrics = result["evaluation_results"][pair]
            self.assertEqual(metadata["quantile_ranks"][pair], int(radius))
            self.assertEqual(metadata["quantile_radii"][pair], radius)
            lower = predictions[19:] - radius
            upper = predictions[19:] + radius
            target = item["heldout_y"][19:]
            alpha = 1.0 - (pair[1] - pair[0])
            score = upper - lower + 2.0 / alpha * (np.maximum(lower - target, 0) + np.maximum(target - upper, 0))
            np.testing.assert_array_equal(metrics["lower_interval"], lower)
            np.testing.assert_array_equal(metrics["upper_interval"], upper)
            np.testing.assert_array_equal(metrics["lower_residual_quantile"], np.full(6, -radius))
            np.testing.assert_array_equal(metrics["upper_residual_quantile"], np.full(6, radius))
            np.testing.assert_allclose(metrics["winkler_score"], score)
            self.assertAlmostEqual(metrics["avg_coverage"], np.mean((lower <= target) & (target <= upper)))
            self.assertAlmostEqual(metrics["avg_interval_width"], 2.0 * radius)
            self.assertAlmostEqual(metrics["avg_winkler_score"], score.mean())
            plotted_lower, plotted_upper = _resolve_logged_interval_endpoints(metrics)
            np.testing.assert_array_equal(plotted_lower, lower)
            np.testing.assert_array_equal(plotted_upper, upper)

    def test_unbounded_intervals_have_full_coverage_and_infinite_scores_without_nans(self):
        result = evaluate_sequence("tiny", _artifact(3), _config())
        for metrics in result["evaluation_results"].values():
            self.assertEqual(metrics["avg_coverage"], 1.0)
            self.assertTrue(np.isposinf(metrics["avg_interval_width"]))
            self.assertTrue(np.isposinf(metrics["avg_winkler_score"]))
            self.assertTrue(np.isposinf(metrics["winkler_score"]).all())
            self.assertTrue(np.isposinf(metrics["interval_width"]).all())

    def test_each_series_uses_only_its_own_calibration_data(self):
        first, second = _artifact(50), _artifact(61)
        second["heldout_y"] += 300.0
        config = _config()
        combined = evaluate_sequences({"first": first, "second": second}, config, num_cores=1)
        for key, item in (("first", first), ("second", second)):
            with self.subTest(key=key):
                alone = evaluate_sequence(key, item, config)
                self.assert_same_intervals(alone, combined[key])
        self.assertNotEqual(combined["first"]["metadata"]["quantile_radii"], combined["second"]["metadata"]["quantile_radii"])

    def test_reversed_pairs_preserve_result_keys_and_asymmetric_pairs_are_rejected(self):
        config = _config()
        expected = evaluate_sequence("station", _artifact(), config)
        config.model.target_quantiles = [[.9, .1]]
        actual = evaluate_sequence("station", _artifact(), config)
        for field in ("lower_interval", "upper_interval"):
            np.testing.assert_array_equal(actual["evaluation_results"][(.9, .1)][field], expected["evaluation_results"][(.1, .9)][field])
        config.model.target_quantiles = [[.1, .8]]
        with self.assertRaises(ValueError):
            evaluate_sequence("station", _artifact(), config)

    def test_invalid_prediction_steps_are_rejected(self):
        for step in (0, 2, True, 1.5):
            config = _config()
            config.model.prediction_step = step
            with self.subTest(step=step), self.assertRaises(ValueError):
                evaluate_sequence("station", _artifact(), config)

    def test_parallel_and_serial_evaluation_agree(self):
        data = {"first": _artifact(50), "second": _artifact(61)}
        serial = evaluate_sequences(data, _config(), num_cores=1)
        parallel = evaluate_sequences(data, _config(), num_cores=2)
        self.assertEqual(list(serial), list(parallel))
        for key in data:
            self.assert_same_intervals(serial[key], parallel[key])

    def test_temporary_artifact_writes_standard_pipeline_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "forecasts.pkl"
            with artifact_path.open("wb") as stream:
                pickle.dump({"station": _artifact(50)}, stream)
            config = _config()
            config.data.data_path = str(artifact_path)
            config.saving_dir = str(root / "results")
            config_path = root / "config.yaml"
            OmegaConf.save(config, config_path)
            result = run_split_cp(config_path, num_cores=1)
            output = Path(config.saving_dir)
            self.assertTrue((output / "resolved_config.yaml").is_file())
            with (output / "log.pkl").open("rb") as stream:
                saved = pickle.load(stream)
            self.assert_same_intervals(result["station"], saved["station"])
            with (output / "summary_results.pkl").open("rb") as stream:
                summary = pickle.load(stream)
            for pair, metrics in result["station"]["evaluation_results"].items():
                self.assertAlmostEqual(summary[pair]["avg_coverage_mean"], metrics["avg_coverage"])
                self.assertAlmostEqual(summary[pair]["avg_interval_width_mean"], metrics["avg_interval_width"])
            # Programmatic experiment callers can pass a resolved mapping, too.
            config.saving_dir = str(root / "mapping_results")
            mapped = run_split_cp(OmegaConf.to_container(config, resolve=True), num_cores=1)
            self.assert_same_intervals(result["station"], mapped["station"])

    def test_unbounded_run_preserves_infinite_mean_and_marks_undefined_dispersion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "forecasts.pkl"
            with artifact_path.open("wb") as stream:
                pickle.dump({"tiny": _artifact(3)}, stream)
            config = _config()
            config.data.data_path = str(artifact_path)
            config.saving_dir = str(root / "results")
            config.plotting.plotting = True
            run_split_cp(config)
            with (Path(config.saving_dir) / "summary_results.pkl").open("rb") as stream:
                summary = pickle.load(stream)
            for result in summary.values():
                self.assertEqual(result["avg_coverage_mean"], 1.0)
                for metric in ("avg_interval_width", "avg_winkler_score"):
                    self.assertTrue(np.isposinf(result[metric + "_mean"]))
                    self.assertIsNone(result[metric + "_std"])
            self.assertEqual(len(list((Path(config.saving_dir) / "plots").glob("*.pdf"))), 2)


if __name__ == "__main__":
    unittest.main()
