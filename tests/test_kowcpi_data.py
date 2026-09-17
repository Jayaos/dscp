import copy
import unittest

import numpy as np

from baselines.kowcpi.data import prepare_sequence, split_boundaries


def config(normalize=False):
    return {"data": {"calibration_ratio": 0.66, "normalize": normalize}}


def artifact(length=100):
    index = np.arange(length, dtype=float)
    predictions = 10.0 + index * 0.2
    return {"heldout_y": predictions + np.sin(index), "heldout_predictions": predictions[:, None]}


class KOWCPIDataTests(unittest.TestCase):
    def test_outer_and_nested_boundaries_preserve_test_for_100_and_101(self):
        for length in (100, 101):
            with self.subTest(length=length):
                outer = split_boundaries(length, 0.66)
                inner = split_boundaries(length, 0.66, model_selection_valid_ratio=0.15)
                self.assertEqual(outer, split_boundaries(length, 0.66, 0.34))
                self.assertEqual(outer["calibration_size"], 66)
                self.assertEqual(outer["validation_size"], 0)
                self.assertEqual(inner["calibration_size"], 56)
                self.assertEqual(inner["validation_size"], 10)
                for bounds in (outer, inner):
                    self.assertEqual(bounds["nominal_calibration_size"], 66)
                    self.assertEqual(bounds["test_start"], 66)
                    self.assertEqual(bounds["test_size"], length - 66)
                    self.assertEqual(sum(bounds[key] for key in (
                        "calibration_size", "validation_size", "test_size",
                    )), length)

    def test_integer_rounding_matches_other_methods(self):
        self.assertEqual(split_boundaries(100, 0.29)["calibration_size"], 29)
        inner = split_boundaries(100, 0.7, model_selection_valid_ratio=0.1)
        self.assertEqual(inner["calibration_size"], 63)
        self.assertEqual(inner["validation_size"], 7)

    def test_preparation_returns_history_and_evaluation_as_flat_prefix(self):
        item = artifact()
        for split, history_size, evaluation_size, end in (
            ("validation", 56, 10, 66), ("test", 66, 34, 100),
        ):
            with self.subTest(split=split):
                result = prepare_sequence(item, config(), split=split)
                self.assertEqual(result["calibration_size"], history_size)
                self.assertEqual(result["evaluation_start"], history_size)
                self.assertEqual(result["evaluation_end"], end)
                self.assertEqual(result["evaluation_size"], evaluation_size)
                self.assertEqual(result["raw_predictions"].shape, (end,))
                np.testing.assert_allclose(result["residuals"], np.sin(np.arange(end)))
                self.assertIsNone(result["residual_normalized_std"])
                self.assertIsNone(result["residual_normalization_params"])

    def test_validation_ignores_nonfinite_and_overflowing_test_values(self):
        expected = prepare_sequence(artifact(), config(True), split="validation")
        for future_y, future_prediction in ((np.nan, np.inf), (1e308, -1e308)):
            with self.subTest(future_y=future_y):
                item = artifact()
                item["heldout_y"][66:] = future_y
                item["heldout_predictions"][66:] = future_prediction
                with np.errstate(all="raise"):
                    actual = prepare_sequence(item, config(True), split="validation")
                np.testing.assert_array_equal(actual["residuals"], expected["residuals"])
                self.assertEqual(actual["residual_normalized_std"], expected["residual_normalized_std"])
                with self.assertRaises(ValueError):
                    prepare_sequence(item, config(False), split="test")

    def test_validation_slices_before_converting_test_values_to_float(self):
        class UnavailableTestValue:
            def __float__(self):
                raise AssertionError("Final-test value was converted during tuning.")

        item = {key: value.astype(object) for key, value in artifact().items()}
        item["heldout_y"][66:] = UnavailableTestValue()
        item["heldout_predictions"][66:] = UnavailableTestValue()
        actual = prepare_sequence(item, config(), split="validation")
        np.testing.assert_allclose(actual["residuals"], np.sin(np.arange(66)))

    def test_normalization_uses_initial_history_and_excludes_validation(self):
        item = artifact()
        item["heldout_y"][56:66] += 10000.0
        actual = prepare_sequence(item, config(True), split="validation")
        mean = item["heldout_y"][:56].mean()
        std = item["heldout_y"][:56].std() + 1e-8
        expected = (item["heldout_y"][:66] - mean) / std - (
            item["heldout_predictions"][:66, 0] - mean
        ) / std
        self.assertEqual(actual["residual_normalized_std"], std)
        self.assertEqual(actual["residual_normalization_params"], (0.0, std))
        np.testing.assert_array_equal(actual["residuals"], expected)

    def test_inputs_are_not_mutated_or_aliased(self):
        item = artifact()
        settings = config(True)
        original_item, original_settings = copy.deepcopy(item), copy.deepcopy(settings)
        result = prepare_sequence(item, settings, split="validation")
        for key in item:
            np.testing.assert_array_equal(item[key], original_item[key])
        self.assertEqual(settings, original_settings)
        result["raw_y"][:] = 0.0
        result["raw_predictions"][:] = 0.0
        for key in item:
            np.testing.assert_array_equal(item[key], original_item[key])

    def test_invalid_ratios_lengths_and_empty_splits_are_rejected(self):
        for invalid in (True, np.bool_(False), np.nan, np.inf, "0.66", None, [0.66], 0.0, 1.0, -0.1):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    split_boundaries(100, invalid, 0.34)
                if invalid is not None:
                    with self.assertRaises(ValueError):
                        split_boundaries(100, 0.66, invalid)
                settings = config()
                settings["tuning"] = {"model_selection_valid_ratio": invalid}
                with self.assertRaises(ValueError):
                    prepare_sequence(artifact(), settings, split="validation")
        for length in (0, -1, 100.5, True):
            with self.subTest(length=length), self.assertRaises(ValueError):
                split_boundaries(length, 0.66, 0.34)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            split_boundaries(100, 0.6, 0.3)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            split_boundaries(1, 0.66, 0.34)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            split_boundaries(2, 0.5, 0.5, model_selection_valid_ratio=0.15)

    def test_legacy_and_missing_split_keys_have_migration_errors(self):
        for key in ("train_ratio", "valid_ratio", "validation_ratio"):
            with self.subTest(key=key):
                settings = config()
                settings["data"][key] = 0.1
                with self.assertRaisesRegex(ValueError, "tuning.model_selection_valid_ratio"):
                    prepare_sequence(artifact(), settings)
        settings = config()
        del settings["data"]["calibration_ratio"]
        with self.assertRaisesRegex(ValueError, "requires data.calibration_ratio"):
            prepare_sequence(artifact(), settings)

    def test_invalid_sequence_shapes_values_and_lengths_are_rejected(self):
        for values in ([], 1.0, np.zeros((100, 2)), np.zeros((1, 100)),
                       np.zeros((100, 1, 1)), [np.nan] * 100, [np.inf] * 100,
                       [1j] * 100, ["not-a-number"] * 100):
            with self.subTest(shape=np.shape(values)):
                item = artifact()
                item["heldout_y"] = values
                with self.assertRaises(ValueError):
                    prepare_sequence(item, config())
        item = artifact()
        item["heldout_predictions"] = item["heldout_predictions"][:-1]
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            prepare_sequence(item, config())
        with self.assertRaisesRegex(ValueError, "split must be"):
            prepare_sequence(artifact(), config(), split="training")


if __name__ == "__main__":
    unittest.main()
