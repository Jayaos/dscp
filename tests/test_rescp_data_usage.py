import copy
import unittest

import numpy as np
from omegaconf import OmegaConf

from baselines.rescp.data import prepare_sequence, scalar_sequence, split_boundaries
from baselines.rescp.run_rescp import evaluate_sequence, evaluate_sequences
from utils.plotting import _resolve_logged_interval_endpoints


def _config(calibration_ratio=0.5, validation_ratio=0.2, test_ratio=0.3):
    return OmegaConf.create(
        {
            "seed": 17,
            "device": "cpu",
            "data": {
                "calibration_ratio": calibration_ratio,
                "validation_ratio": validation_ratio,
                "test_ratio": test_ratio,
                "normalize": True,
            },
            "model": {
                "reservoir_size": 8,
                "spectral_radius": 0.8,
                "leak_rate": 0.9,
                "input_scaling": 0.25,
                "connectivity": 1.0,
                "temperature": 0.1,
                "calibration_size": 12,
                "sampling_num": 32,
                "use_beta_search": True,
                "beta_bins": 4,
                "decay": "linear",
                "decay_rate": 0.99,
                "recurrence": "upstream",
                "target_quantiles": [[0.1, 0.9], [0.05, 0.95]],
            },
        }
    )


def _artifact(length=100):
    index = np.arange(length, dtype=np.float64)
    predictions = 50.0 + index * 0.2
    residuals = 2.0 + np.sin(index * 0.7) + index * 0.03
    return {
        "heldout_x": np.column_stack([index, index * 100.0]),
        "heldout_y": predictions + residuals,
        "heldout_predictions": predictions[:, None],
    }


class ResCPScalarSequenceTests(unittest.TestCase):
    def test_accepts_scalar_lists_vectors_and_single_columns(self):
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        for value in (expected.tolist(), expected.astype(np.float32), expected[:, None]):
            with self.subTest(shape=np.shape(value)):
                actual = scalar_sequence(value, "heldout_y")
                self.assertEqual(actual.dtype, np.dtype(np.float64))
                self.assertEqual(actual.shape, (3,))
                np.testing.assert_array_equal(actual, expected)

    def test_rejects_non_scalar_targets_and_nonfinite_values(self):
        for value in (
            np.zeros((4, 2)),
            np.zeros((1, 4)),
            np.zeros((4, 1, 1)),
            1.0,
            [1.0, np.nan],
            [1.0, np.inf],
            [1.0, -np.inf],
        ):
            with self.subTest(shape=np.shape(value)), self.assertRaises(ValueError):
                scalar_sequence(value, "heldout_y")


class ResCPSplitTests(unittest.TestCase):
    def test_split_rounding_matches_existing_cp_target_boundaries(self):
        for length, ratios, expected in (
            (100, (0.5, 0.16, 0.34), (50, 66, 34)),
            (101, (0.5, 0.16, 0.34), (50, 67, 34)),
            (100, (0.6, 0.2, 0.2), (60, 80, 20)),
            (100, (0.29, 0.14, 0.57), (29, 43, 57)),
        ):
            with self.subTest(length=length, ratios=ratios):
                result = split_boundaries(length, *ratios)
                calibration_end, validation_end, test_size = expected
                self.assertEqual(result["calibration_end"], calibration_end)
                self.assertEqual(result["validation_end"], validation_end)
                self.assertEqual(result["test_size"], test_size)
                self.assertEqual(result["calibration_size"], calibration_end)
                self.assertEqual(
                    result["validation_size"], validation_end - calibration_end
                )
                self.assertEqual(
                    sum(result[key] for key in (
                        "calibration_size", "validation_size", "test_size"
                    )),
                    length,
                )

    def test_zero_validation_is_available_for_fixed_hyperparameters(self):
        result = split_boundaries(101, 0.66, 0.0, 0.34)
        self.assertEqual(result["calibration_end"], 66)
        self.assertEqual(result["validation_end"], 66)
        self.assertEqual(result["validation_size"], 0)
        self.assertEqual(result["test_size"], 35)

    def test_invalid_ratios_and_empty_partitions_are_rejected(self):
        for length, ratios in (
            (100, (0.0, 0.2, 0.8)),
            (100, (-0.1, 0.2, 0.9)),
            (100, (0.5, -0.1, 0.6)),
            (100, (0.5, 0.5, 0.0)),
            (100, (0.5, 0.2, 0.4)),
            (100, (np.nan, 0.2, 0.3)),
            (100, (0.5, np.inf, 0.3)),
            (0, (0.5, 0.2, 0.3)),
            (2, (0.1, 0.2, 0.7)),
            (2, (0.5, 0.4, 0.1)),
        ):
            with self.subTest(length=length, ratios=ratios), self.assertRaises(ValueError):
                split_boundaries(length, *ratios)


class ResCPPreparationTests(unittest.TestCase):
    def test_mixed_lstm_shapes_produce_aligned_scalar_residuals(self):
        item = _artifact()
        result = prepare_sequence(item, _config(), split="validation")
        expected_residuals = item["heldout_y"] - item["heldout_predictions"][:, 0]
        np.testing.assert_array_equal(
            result["calibration_residuals"], expected_residuals[:50]
        )
        np.testing.assert_array_equal(result["residuals"], expected_residuals[50:70])
        np.testing.assert_array_equal(result["y"], item["heldout_y"][50:70])
        np.testing.assert_array_equal(
            result["predictions"], item["heldout_predictions"][50:70, 0]
        )
        np.testing.assert_array_equal(result["target_indices"], np.arange(50, 70))
        self.assertEqual(np.asarray(result["warmup_residuals"]).size, 0)

    def test_test_prefix_replays_validation_without_resplitting(self):
        item = _artifact(101)
        config = _config(0.5, 0.16, 0.34)
        result = prepare_sequence(item, config, split="test")
        residuals = item["heldout_y"] - item["heldout_predictions"][:, 0]
        np.testing.assert_array_equal(result["calibration_residuals"], residuals[:50])
        np.testing.assert_array_equal(result["warmup_residuals"], residuals[50:67])
        np.testing.assert_array_equal(result["residuals"], residuals[67:])
        np.testing.assert_array_equal(result["target_indices"], np.arange(67, 101))

    def test_reserved_test_values_and_nans_cannot_change_validation_data(self):
        item = _artifact()
        baseline = prepare_sequence(item, _config(), split="validation")
        altered = copy.deepcopy(item)
        altered["heldout_y"][70:] = np.nan
        altered["heldout_predictions"][70:] = np.inf
        altered["heldout_x"][:] = np.nan
        actual = prepare_sequence(altered, _config(), split="validation")
        for key in (
            "calibration_residuals", "warmup_residuals", "residuals",
            "y", "predictions", "target_indices",
        ):
            with self.subTest(field=key):
                np.testing.assert_array_equal(actual[key], baseline[key])
        self.assertEqual(actual["boundaries"], baseline["boundaries"])
        with self.assertRaises(ValueError):
            prepare_sequence(altered, _config(), split="test")

    def test_nonfinite_consumed_values_are_rejected(self):
        for field, index in (("heldout_y", 5), ("heldout_predictions", 55)):
            with self.subTest(field=field, index=index):
                item = _artifact()
                item[field][index] = np.nan
                with self.assertRaises(ValueError):
                    prepare_sequence(item, _config(), split="validation")

    def test_mismatched_lengths_and_multioutput_are_rejected(self):
        for predictions in (np.zeros(99), np.zeros((100, 2))):
            with self.subTest(shape=predictions.shape):
                item = _artifact()
                item["heldout_predictions"] = predictions
                with self.assertRaises(ValueError):
                    prepare_sequence(item, _config(), split="validation")

    def test_zero_validation_evaluates_test_and_rejects_validation(self):
        config = _config(0.7, 0.0, 0.3)
        result = prepare_sequence(_artifact(), config, split="test")
        np.testing.assert_array_equal(result["target_indices"], np.arange(70, 100))
        self.assertEqual(np.asarray(result["warmup_residuals"]).size, 0)
        with self.assertRaises(ValueError):
            prepare_sequence(_artifact(), config, split="validation")

    def test_preparation_uses_only_required_base_artifact_fields(self):
        item = _artifact()
        item.pop("heldout_x")
        result = prepare_sequence(item, _config(), split="test")
        self.assertEqual(len(result["residuals"]), 30)

    def test_unknown_evaluation_partition_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_sequence(_artifact(), _config(), split="training")


class ResCPRunnerDataUsageTests(unittest.TestCase):
    def _assert_same_endpoints(self, baseline, actual, count=None):
        self.assertEqual(
            baseline["evaluation_results"].keys(), actual["evaluation_results"].keys()
        )
        for pair, expected in baseline["evaluation_results"].items():
            for field in (
                "lower_interval", "upper_interval",
                "lower_residual_quantile", "upper_residual_quantile",
            ):
                with self.subTest(pair=pair, field=field):
                    np.testing.assert_array_equal(
                        np.asarray(actual["evaluation_results"][pair][field])[:count],
                        np.asarray(expected[field])[:count],
                    )

    def test_validation_predictions_do_not_depend_on_reserved_test(self):
        item = _artifact(50)
        baseline = evaluate_sequence("station", item, _config(), split="validation")
        changed = copy.deepcopy(item)
        changed["heldout_y"][35:] = np.nan
        changed["heldout_predictions"][35:] = np.inf
        actual = evaluate_sequence("station", changed, _config(), split="validation")
        self._assert_same_endpoints(baseline, actual)

    def test_prediction_is_issued_before_observing_its_target(self):
        item = _artifact(50)
        baseline = evaluate_sequence("station", item, _config(), split="test")
        changed = copy.deepcopy(item)
        changed["heldout_y"][39:] += 100_000.0
        actual = evaluate_sequence("station", changed, _config(), split="test")
        # Test starts at 35. Intervals at indices 35 through 39 are unchanged,
        # including the interval for the first altered observation itself.
        self._assert_same_endpoints(baseline, actual, count=5)

    def test_encoder_normalization_uses_only_calibration_prefix(self):
        item = _artifact(50)
        baseline = evaluate_sequence("station", item, _config(), split="test")
        changed = copy.deepcopy(item)
        changed["heldout_y"][25:] += 20_000.0
        actual = evaluate_sequence("station", changed, _config(), split="test")
        residuals = item["heldout_y"][:25] - item["heldout_predictions"][:25, 0]
        self.assertAlmostEqual(baseline["metadata"]["input_mean"], residuals.mean())
        self.assertAlmostEqual(baseline["metadata"]["input_std"], residuals.std())
        self.assertEqual(
            actual["metadata"]["input_mean"], baseline["metadata"]["input_mean"]
        )
        self.assertEqual(
            actual["metadata"]["input_std"], baseline["metadata"]["input_std"]
        )

    def test_normalized_reservoir_reports_raw_quantiles_and_endpoints(self):
        index = np.arange(50, dtype=np.float64)
        predictions = 1_000.0 + 5.0 * index
        residuals = 100.0 + 10.0 * (index % 7)
        item = {
            "heldout_y": predictions + residuals,
            "heldout_predictions": predictions[:, None],
        }
        result = evaluate_sequence("station", item, _config(), split="test")
        target_y = item["heldout_y"][35:]
        for pair, metrics in result["evaluation_results"].items():
            with self.subTest(pair=pair):
                self.assertIsInstance(pair, tuple)
                low_r = np.asarray(metrics["lower_residual_quantile"])
                high_r = np.asarray(metrics["upper_residual_quantile"])
                low = np.asarray(metrics["lower_interval"])
                high = np.asarray(metrics["upper_interval"])
                self.assertEqual(low.shape, (15,))
                self.assertTrue(np.all(low_r >= 100.0))
                self.assertTrue(np.all(high_r <= 160.0))
                self.assertTrue(np.all(low <= high))
                np.testing.assert_allclose(low, predictions[35:] + low_r)
                np.testing.assert_allclose(high, predictions[35:] + high_r)
                self.assertAlmostEqual(
                    metrics["avg_coverage"],
                    np.mean((low <= target_y) & (target_y <= high)),
                )
                self.assertAlmostEqual(metrics["avg_interval_width"], np.mean(high - low))
                alpha = 1.0 - (max(pair) - min(pair))
                expected_score = (
                    high - low
                    + np.maximum(low - target_y, 0.0) * (2.0 / alpha)
                    + np.maximum(target_y - high, 0.0) * (2.0 / alpha)
                )
                self.assertAlmostEqual(metrics["avg_winkler_score"], np.mean(expected_score))
                plotted_low, plotted_high = _resolve_logged_interval_endpoints(metrics)
                np.testing.assert_array_equal(plotted_low, low)
                np.testing.assert_array_equal(plotted_high, high)

    def test_unequal_series_have_independent_states_and_normalization(self):
        config = _config()
        first = _artifact(50)
        second = _artifact(61)
        second["heldout_y"] += 300.0
        data = {"first": first, "second": second}
        together = evaluate_sequences(data, config, split="test", num_cores=1)
        for key, item in data.items():
            with self.subTest(series=key):
                alone = evaluate_sequences({key: item}, config, split="test", num_cores=1)
                self._assert_same_endpoints(together[key], alone[key])
        lengths = [
            len(next(iter(together[key]["evaluation_results"].values()))["lower_interval"])
            for key in data
        ]
        self.assertEqual(lengths, [15, 18])

    def test_reversed_quantile_pairs_keep_their_result_key(self):
        config = _config()
        config.model.target_quantiles = [[0.1, 0.9]]
        expected = evaluate_sequence("station", _artifact(50), config)
        config.model.target_quantiles = [[0.9, 0.1]]
        actual = evaluate_sequence("station", _artifact(50), config)
        for field in ("lower_interval", "upper_interval", "selected_beta"):
            np.testing.assert_array_equal(
                actual["evaluation_results"][(0.9, 0.1)][field],
                expected["evaluation_results"][(0.1, 0.9)][field],
            )

    def test_fixed_asymmetric_quantiles_use_actual_tail_penalties(self):
        config = _config()
        config.model.target_quantiles = [[0.1, 0.7]]
        config.model.use_beta_search = False
        result = evaluate_sequence("station", _artifact(50), config)
        metrics = result["evaluation_results"][(0.1, 0.7)]
        lower = np.array(metrics["lower_interval"])
        upper = np.array(metrics["upper_interval"])
        target = np.array(metrics["target_y"])
        expected = (upper - lower + np.maximum(lower - target, 0) / 0.1
                    + np.maximum(target - upper, 0) / 0.3)
        np.testing.assert_allclose(metrics["winkler_score"], expected, atol=1e-10)
        np.testing.assert_array_equal(metrics["selected_beta"], np.full(len(target), 0.1))

    def test_single_and_multiple_workers_produce_identical_results(self):
        data = {"first": _artifact(50), "second": _artifact(61)}
        sequential = evaluate_sequences(data, _config(), split="test", num_cores=1)
        parallel = evaluate_sequences(data, _config(), split="test", num_cores=2)
        for key in data:
            with self.subTest(series=key):
                self._assert_same_endpoints(sequential[key], parallel[key])


if __name__ == "__main__":
    unittest.main()
