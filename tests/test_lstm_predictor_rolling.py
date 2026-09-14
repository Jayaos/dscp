import unittest
from types import SimpleNamespace

import numpy as np

from base_predictor.lstm_predictor import (
    LSTMPredictor,
    _inner_split_index,
    _make_heldout_sequence_prediction_data,
    _make_sequence_prediction_data,
    _mean_per_sequence_mse,
)


class LSTMSequenceConstructionTests(unittest.TestCase):
    def test_sequence_uses_only_prior_covariate_target_pairs(self):
        x = np.array(
            [
                [10.0, 100.0],
                [20.0, 200.0],
                [30.0, 300.0],
                [40.0, 400.0],
                [50.0, 500.0],
            ],
            dtype=np.float32,
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        inputs, targets = _make_sequence_prediction_data(x, y, k=2)

        expected_inputs = np.array(
            [
                [[10.0, 100.0, 1.0], [20.0, 200.0, 2.0]],
                [[20.0, 200.0, 2.0], [30.0, 300.0, 3.0]],
                [[30.0, 300.0, 3.0], [40.0, 400.0, 4.0]],
            ],
            dtype=np.float32,
        )
        expected_targets = np.array([[3.0], [4.0], [5.0]], dtype=np.float32)

        np.testing.assert_array_equal(inputs, expected_inputs)
        np.testing.assert_array_equal(targets, expected_targets)

    def test_heldout_windows_cross_split_and_use_true_observed_targets(self):
        train_x = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
        heldout_x = np.array([[40.0], [50.0], [60.0]], dtype=np.float32)
        train_y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        heldout_y = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        inputs, targets = _make_heldout_sequence_prediction_data(
            train_x,
            heldout_x,
            train_y,
            heldout_y,
            k=2,
        )

        expected_inputs = np.array(
            [
                [[20.0, 2.0], [30.0, 3.0]],
                [[30.0, 3.0], [40.0, 4.0]],
                [[40.0, 4.0], [50.0, 5.0]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(inputs, expected_inputs)
        np.testing.assert_array_equal(targets, heldout_y[:, None])

        changed_heldout_y = heldout_y.copy()
        changed_heldout_y[0] = 400.0
        changed_inputs, _ = _make_heldout_sequence_prediction_data(
            train_x,
            heldout_x,
            train_y,
            changed_heldout_y,
            k=2,
        )

        np.testing.assert_array_equal(changed_inputs[0], inputs[0])
        self.assertFalse(np.array_equal(changed_inputs[1], inputs[1]))

        changed_heldout_y = heldout_y.copy()
        changed_heldout_y[-1] = 600.0
        changed_inputs, _ = _make_heldout_sequence_prediction_data(
            train_x,
            heldout_x,
            train_y,
            changed_heldout_y,
            k=2,
        )
        np.testing.assert_array_equal(changed_inputs, inputs)

    def test_exactly_one_window_of_history_has_no_training_target(self):
        x = np.array([[10.0], [20.0]], dtype=np.float32)
        y = np.array([1.0, 2.0], dtype=np.float32)

        inputs, targets = _make_sequence_prediction_data(x, y, k=2)

        self.assertEqual(inputs.shape, (0, 2, 2))
        self.assertEqual(targets.shape, (0, 1))


class LSTMInnerValidationTests(unittest.TestCase):
    def test_each_sequence_is_split_before_global_normalization_and_pooling(self):
        series_a_x = np.concatenate(
            [np.zeros(9), [100.0], np.full(10, 10_000.0)]
        ).astype(np.float32)[:, None]
        series_a_y = np.concatenate(
            [np.full(9, 10.0), [1000.0], np.full(10, 100_000.0)]
        ).astype(np.float32)
        series_b_x = np.concatenate(
            [np.full(9, 2.0), [200.0], np.full(10, 20_000.0)]
        ).astype(np.float32)[:, None]
        series_b_y = np.concatenate(
            [np.full(9, 14.0), [2000.0], np.full(10, 200_000.0)]
        ).astype(np.float32)
        data = SimpleNamespace(
            data_type="toy",
            data={
                "a": {"x": series_a_x, "y": series_a_y},
                "b": {"x": series_b_x, "y": series_b_y},
            },
        )

        predictor = LSTMPredictor(
            data,
            embedding_dim=2,
            hidden_dim=2,
            num_layers=1,
            train_ratio=0.5,
            window_length=2,
        )
        predictor._prepare_fit_data(0.9)

        for item in predictor.data_processed.values():
            self.assertEqual(item["inner_train_end"], 9)
            self.assertEqual(len(item["train_y_seq"]), 7)
            self.assertEqual(len(item["valid_y_seq"]), 1)
            np.testing.assert_allclose(item["train_x_mu"], [1.0])
            np.testing.assert_allclose(item["train_x_std"], [1.0])
            np.testing.assert_allclose(item["train_y_mu"], 12.0)
            np.testing.assert_allclose(item["train_y_std"], 2.0)

        self.assertEqual(
            sum(len(item["train_y_seq"]) for item in predictor.data_processed.values()),
            14,
        )
        self.assertEqual(
            sum(len(item["valid_y_seq"]) for item in predictor.data_processed.values()),
            2,
        )

        item_a = predictor.data_processed["a"]
        item_b = predictor.data_processed["b"]
        np.testing.assert_allclose(
            item_a["valid_input_seq"][0],
            [[-1.0, -1.0], [-1.0, -1.0]],
        )
        np.testing.assert_allclose(item_a["valid_y_seq"], [[494.0]], rtol=1e-6)
        np.testing.assert_allclose(item_b["valid_y_seq"], [[994.0]], rtol=1e-6)
        np.testing.assert_allclose(item_a["normalized_heldout_x"][0], [9999.0])
        np.testing.assert_allclose(item_a["normalized_heldout_y"][0], 49994.0)

    def test_inner_split_rejects_empty_train_or_validation_windows(self):
        self.assertEqual(_inner_split_index(10, 2, 0.7), 7)

        for ratio in (0.0, 1.0, np.nan):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    _inner_split_index(10, 2, ratio)

        with self.assertRaisesRegex(ValueError, "more than window_length"):
            _inner_split_index(10, 2, 0.2)

    def test_validation_metric_weights_each_sequence_equally(self):
        loss = _mean_per_sequence_mse(
            squared_errors=np.array([4.0, 1.0, 1.0, 1.0]),
            sequence_lengths=[1, 3],
        )
        reversed_loss = _mean_per_sequence_mse(
            squared_errors=np.array([1.0, 1.0, 1.0, 4.0]),
            sequence_lengths=[3, 1],
        )

        self.assertEqual(loss, 2.5)
        self.assertEqual(reversed_loss, 2.5)


if __name__ == "__main__":
    unittest.main()
