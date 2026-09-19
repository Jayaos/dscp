"""Prevent scalar forecast shapes from broadcasting into pairwise residuals."""

import importlib.util
import itertools
import unittest

import numpy as np

from utils.forecast_data import canonicalize_forecast_data


def artifact(y_column=False, predictions_column=True, length=100):
    time = np.arange(length, dtype=np.float64)
    targets = 10.0 + 0.2 * time + np.sin(time / 3.0)
    predictions = targets - (0.5 * np.cos(time / 4.0) - 0.2)
    return {
        "series": {
            "heldout_x": np.column_stack((time, np.cos(time))),
            "heldout_y": targets[:, None] if y_column else targets,
            "heldout_predictions": (
                predictions[:, None] if predictions_column else predictions
            ),
        }
    }


class ForecastShapeTests(unittest.TestCase):
    def test_vector_and_column_combinations_produce_aligned_residuals(self):
        for y_column, predictions_column in itertools.product((False, True), repeat=2):
            with self.subTest(y_column=y_column, predictions_column=predictions_column):
                source = artifact(y_column, predictions_column)
                original = source["series"]
                expected = (
                    original["heldout_y"].reshape(-1)
                    - original["heldout_predictions"].reshape(-1)
                )
                actual = canonicalize_forecast_data(source)["series"]
                self.assertEqual(actual["heldout_y"].shape, (100, 1))
                self.assertEqual(actual["heldout_predictions"].shape, (100, 1))
                residuals = actual["heldout_y"] - actual["heldout_predictions"]
                self.assertEqual(residuals.shape, (100, 1))
                np.testing.assert_array_equal(residuals[:, 0], expected)

    def test_lists_and_single_timestamp_are_supported(self):
        for targets, predictions in (([3.5], [[2.0]]), ([[3.5]], [2.0])):
            with self.subTest(targets=targets):
                item = canonicalize_forecast_data({"one": {
                    "heldout_y": targets,
                    "heldout_predictions": predictions,
                }})["one"]
                self.assertEqual(item["heldout_y"].shape, (1, 1))
                self.assertEqual(item["heldout_predictions"].shape, (1, 1))
                np.testing.assert_array_equal(
                    item["heldout_y"] - item["heldout_predictions"], [[1.5]]
                )

    def test_canonicalization_does_not_mutate_input_or_copy_array_storage(self):
        for dtype in (np.float32, np.float64, np.int64, np.uint16):
            with self.subTest(dtype=dtype):
                # Noncontiguous vectors must also be handled without a full copy.
                targets = np.arange(20, dtype=dtype)[::2]
                predictions = np.arange(10, dtype=dtype)[:, None]
                features = np.ones((10, 2, 3))
                history = np.arange(7, dtype=dtype)
                metadata = {"description": "preserve unrelated fields"}
                source_item = {
                    "heldout_y": targets,
                    "heldout_predictions": predictions,
                    "heldout_x": features,
                    "train_y": history,
                    "metadata": metadata,
                }
                source = {"series": source_item}
                result = canonicalize_forecast_data(source)
                self.assertIsNot(result, source)
                self.assertIsNot(result["series"], source_item)
                self.assertEqual(set(result["series"]), set(source_item))
                self.assertEqual(targets.shape, (10,))
                for name, value in source_item.items():
                    self.assertIs(source_item[name], value)
                    if name not in ("heldout_y", "heldout_predictions"):
                        self.assertIs(result["series"][name], value)
                for field in ("heldout_y", "heldout_predictions"):
                    self.assertEqual(result["series"][field].dtype, np.dtype(dtype))
                    self.assertTrue(np.shares_memory(result["series"][field], source_item[field]))
                    np.testing.assert_array_equal(
                        result["series"][field].reshape(-1), source_item[field].reshape(-1)
                    )
                result["series"]["new_field"] = "prepared metadata"
                self.assertNotIn("new_field", source_item)

    def test_nonfinite_values_are_preserved_without_changing_missing_data_policy(self):
        values = np.array([1.0, np.nan, np.inf, -np.inf])
        result = canonicalize_forecast_data({"series": {
            "heldout_y": values,
            "heldout_predictions": values[:, None],
        }})["series"]
        np.testing.assert_array_equal(result["heldout_y"][:, 0], values)
        np.testing.assert_array_equal(result["heldout_predictions"][:, 0], values)

    def test_empty_container_is_valid(self):
        self.assertEqual(canonicalize_forecast_data({}), {})

    def test_nonmapping_containers_are_rejected(self):
        for values in (None, [], np.array([1.0])):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    canonicalize_forecast_data(values)

    def test_invalid_scalar_shapes_fail_with_sequence_and_field(self):
        invalid = (
            np.array(1.0),
            np.array([]),
            np.empty((0, 1)),
            np.empty((3, 0)),
            np.ones((3, 2)),
            np.ones((1, 3)),
            np.ones((3, 1, 1)),
            [[1.0], [2.0, 3.0]],
        )
        for field, values in itertools.product(("heldout_y", "heldout_predictions"), invalid):
            with self.subTest(field=field, shape=np.shape(values) if isinstance(values, np.ndarray) else "ragged"):
                source = artifact(length=3)
                source["series"][field] = values
                with self.assertRaises(ValueError) as error:
                    canonicalize_forecast_data(source)
                self.assertIn("series", str(error.exception))
                self.assertIn(field, str(error.exception))

    def test_non_real_numeric_fields_are_rejected(self):
        invalid = (
            np.array([True, False, True]),
            np.array([1, 2, 3], dtype=complex),
            np.array(["1", "2", "3"]),
            np.array([1, 2, 3], dtype=object),
        )
        for field, values in itertools.product(("heldout_y", "heldout_predictions"), invalid):
            with self.subTest(field=field, dtype=values.dtype):
                source = artifact(length=3)
                source["series"][field] = values
                with self.assertRaises(ValueError) as error:
                    canonicalize_forecast_data(source)
                self.assertIn("series", str(error.exception))
                self.assertIn(field, str(error.exception))

    def test_different_lengths_are_rejected_even_when_numpy_could_broadcast(self):
        for prediction_length in (1, 2, 4):
            with self.subTest(prediction_length=prediction_length):
                source = artifact(length=3)
                source["series"]["heldout_predictions"] = np.zeros((prediction_length, 1))
                with self.assertRaises(ValueError) as error:
                    canonicalize_forecast_data(source)
                self.assertIn("series", str(error.exception))

    def test_missing_fields_or_nonmapping_entries_fail_with_sequence(self):
        for item in (
            {"heldout_y": [1.0]},
            {"heldout_predictions": [1.0]},
            {},
            [1.0],
        ):
            with self.subTest(item=item):
                with self.assertRaises(ValueError) as error:
                    canonicalize_forecast_data({"broken_station": item})
                self.assertIn("broken_station", str(error.exception))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class ConformalForecastShapeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dscp.data import ConformalPredictionData

        cls.data_class = ConformalPredictionData

    def assert_prepared_equal(self, actual, expected):
        self.assertEqual(set(actual.data["series"]), set(expected.data["series"]))
        for field in expected.data["series"]:
            np.testing.assert_allclose(
                actual.data["series"][field], expected.data["series"][field],
                err_msg=field,
            )
        self.assertEqual(set(actual.dataset["series"]), set(expected.dataset["series"]))
        for split, expected_dataset in expected.dataset["series"].items():
            actual_dataset = actual.dataset["series"][split]
            self.assertEqual(len(actual_dataset), len(expected_dataset))
            for field in (
                "strided_x", "strided_y", "strided_residual", "target_x",
                "target_y", "target_predictions", "target_residual",
            ):
                np.testing.assert_array_equal(
                    getattr(actual_dataset, field).numpy(),
                    getattr(expected_dataset, field).numpy(),
                    err_msg=f"{split}.{field}",
                )

    def test_quantile_final_and_tuning_splits_preserve_residuals_scaling_and_order(self):
        modes = (
            ("qr_iqn_final", {"train_ratio": 0.5, "valid_ratio": 0.16}, 50, 100),
            ("lcp_final", {"train_ratio": 0.6, "valid_ratio": 0.1,
                           "calibration_ratio": 0.1, "test_ratio": 0.2}, 60, 100),
            ("qr_iqn_tuning", {"train_ratio": 0.5, "valid_ratio": 0.16,
                               "model_selection_valid_ratio": 0.2}, 40, 66),
            ("lcp_tuning", {"train_ratio": 0.6, "valid_ratio": 0.1,
                            "calibration_ratio": 0.1, "test_ratio": 0.2,
                            "model_selection_valid_ratio": 0.2}, 48, 80),
        )
        for (mode, split_options, fit_end, exposed_end), normalize in itertools.product(modes, (False, True)):
            expected = self.data_class(artifact(True, True))
            expected.prepare_quantile_regression_datasets(
                past_window=5, prediction_steps=1, normalize=normalize, **split_options
            )
            for y_column, predictions_column in ((False, True), (True, False), (False, False)):
                with self.subTest(mode=mode, normalize=normalize, y_column=y_column,
                                  predictions_column=predictions_column):
                    source = artifact(y_column, predictions_column)
                    raw_y = source["series"]["heldout_y"].reshape(-1).copy()
                    raw_predictions = source["series"]["heldout_predictions"].reshape(-1).copy()
                    original_shapes = {key: value.shape for key, value in source["series"].items()}
                    actual = self.data_class(source)
                    actual.prepare_quantile_regression_datasets(
                        past_window=5, prediction_steps=1, normalize=normalize, **split_options
                    )
                    self.assert_prepared_equal(actual, expected)
                    self.assertEqual(set(source["series"]), set(original_shapes))
                    for field, shape in original_shapes.items():
                        self.assertEqual(source["series"][field].shape, shape)
                    np.testing.assert_array_equal(
                        actual.data["series"]["raw_heldout_residuals"], raw_y - raw_predictions
                    )
                    scale = raw_y[:fit_end].std() + 1e-8 if normalize else 1.0
                    np.testing.assert_allclose(
                        actual.data["series"]["heldout_residuals"],
                        (raw_y - raw_predictions) / scale, atol=1e-14,
                    )
                    all_targets = np.concatenate([
                        dataset.target_y.numpy().reshape(-1)
                        for dataset in actual.dataset["series"].values()
                    ])
                    np.testing.assert_allclose(all_targets, raw_y[5:exposed_end], rtol=1e-6)

    def test_hopcpt_preserves_signed_residuals_scaling_and_targets(self):
        for normalize, absolute in itertools.product((False, True), repeat=2):
            options = dict(
                prediction_steps=1, y_lags=5, train_ratio=0.5, valid_ratio=0.16,
                normalize=normalize, predict_absolute_residual=absolute,
                conformal_absolute_residual=absolute,
            )
            expected = self.data_class(artifact(True, True))
            expected.prepare_hopcpt_datasets(**options)
            for y_column, predictions_column in ((False, True), (True, False), (False, False)):
                with self.subTest(normalize=normalize, absolute=absolute, y_column=y_column,
                                  predictions_column=predictions_column):
                    source = artifact(y_column, predictions_column)
                    source_predictions = source["series"]["heldout_predictions"].copy()
                    raw_y = source["series"]["heldout_y"].reshape(-1)
                    raw_predictions = source_predictions.reshape(-1)
                    actual = self.data_class(source)
                    actual.prepare_hopcpt_datasets(**options)
                    prepared = actual.data["series"]
                    self.assertEqual(set(prepared), set(expected.data["series"]))
                    for field, value in expected.data["series"].items():
                        np.testing.assert_allclose(prepared[field], value, err_msg=field)
                    self.assertEqual(prepared["heldout_signed_residuals"].shape, (100,))
                    scale = raw_y[:50].std() + 1e-8 if normalize else 1.0
                    signed = (raw_y - raw_predictions) / scale
                    np.testing.assert_allclose(prepared["heldout_signed_residuals"], signed, atol=1e-14)
                    np.testing.assert_allclose(
                        prepared["heldout_context_residuals"],
                        np.abs(signed[5:]) if absolute else signed[5:], atol=1e-14,
                    )
                    np.testing.assert_array_equal(prepared["raw_heldout_target_y"].reshape(-1), raw_y[5:])
                    np.testing.assert_array_equal(
                        prepared["raw_heldout_target_predictions"].reshape(-1), raw_predictions[5:]
                    )
                    np.testing.assert_array_equal(source["series"]["heldout_predictions"], source_predictions)
                    self.assertEqual(set(source["series"]), {"heldout_x", "heldout_y", "heldout_predictions"})

    def test_invalid_shapes_fail_at_constructor_before_preparation(self):
        source = artifact()
        source["series"]["heldout_predictions"] = np.zeros((100, 2))
        with self.assertRaisesRegex(ValueError, "series"):
            self.data_class(source)


if __name__ == "__main__":
    unittest.main()
