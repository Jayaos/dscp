import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dscp.data import ConformalPredictionData
from dscp.models.rnn_predictor import RNNPredictor
from dscp.models.transformer_predictor import TransformerPredictor


class LocalCPSplitTests(unittest.TestCase):
    @staticmethod
    def _artifact(length=100):
        values = np.arange(length, dtype=np.float32)
        return {
            "series": {
                "heldout_x": values[:, None],
                "heldout_y": values,
                "heldout_predictions": np.zeros_like(values),
            }
        }

    def test_four_way_split_is_chronological_and_exhaustive(self):
        prepared = ConformalPredictionData(self._artifact())
        prepared.prepare_quantile_regression_datasets(
            past_window=5,
            prediction_steps=1,
            train_ratio=0.6,
            valid_ratio=0.1,
            calibration_ratio=0.1,
            test_ratio=0.2,
        )

        splits = prepared.dataset["series"]
        self.assertEqual(len(splits["train_dataset"]), 55)
        self.assertEqual(len(splits["valid_dataset"]), 10)
        self.assertEqual(len(splits["calibration_dataset"]), 10)
        self.assertEqual(len(splits["test_dataset"]), 20)

        expected_targets = {
            "train_dataset": np.arange(5, 60),
            "valid_dataset": np.arange(60, 70),
            "calibration_dataset": np.arange(70, 80),
            "test_dataset": np.arange(80, 100),
        }
        for split_name, expected in expected_targets.items():
            with self.subTest(split=split_name):
                actual = splits[split_name].target_residual.squeeze(-1).numpy()
                np.testing.assert_array_equal(actual, expected)

        self.assertEqual(prepared.data["series"]["train_size"], 60)
        self.assertEqual(prepared.data["series"]["valid_size"], 10)
        self.assertEqual(prepared.data["series"]["calibration_size"], 10)
        self.assertEqual(prepared.data["series"]["test_size"], 20)

    def test_legacy_three_way_split_remains_available(self):
        prepared = ConformalPredictionData(self._artifact())
        prepared.prepare_quantile_regression_datasets(
            past_window=5,
            prediction_steps=1,
            train_ratio=0.6,
            valid_ratio=0.2,
        )

        splits = prepared.dataset["series"]
        self.assertNotIn("calibration_dataset", splits)
        self.assertEqual(len(splits["train_dataset"]), 55)
        self.assertEqual(len(splits["valid_dataset"]), 20)
        self.assertEqual(len(splits["test_dataset"]), 20)

    def test_cumulative_boundaries_do_not_round_down_at_integer_cutoffs(self):
        prepared = ConformalPredictionData(self._artifact(length=20))
        prepared.prepare_quantile_regression_datasets(
            past_window=2,
            prediction_steps=1,
            train_ratio=0.7,
            valid_ratio=0.1,
            calibration_ratio=0.1,
            test_ratio=0.1,
        )

        splits = prepared.dataset["series"]
        self.assertEqual(len(splits["train_dataset"]), 12)
        self.assertEqual(len(splits["valid_dataset"]), 2)
        self.assertEqual(len(splits["calibration_dataset"]), 2)
        self.assertEqual(len(splits["test_dataset"]), 2)

    def test_normalization_statistics_use_only_training_prefix(self):
        artifact = self._artifact()
        artifact["series"]["heldout_x"][60:] += 10_000.0
        artifact["series"]["heldout_y"][60:] += 20_000.0
        prepared = ConformalPredictionData(artifact)
        prepared.prepare_quantile_regression_datasets(
            past_window=5,
            prediction_steps=1,
            train_ratio=0.6,
            valid_ratio=0.1,
            calibration_ratio=0.1,
            test_ratio=0.2,
            normalize=True,
        )

        training_values = np.arange(60, dtype=np.float32)
        series_data = prepared.data["series"]
        np.testing.assert_allclose(series_data["train_x_mu"], training_values.mean())
        np.testing.assert_allclose(series_data["train_x_std"], training_values.std())
        np.testing.assert_allclose(series_data["train_y_mu"], training_values.mean())
        np.testing.assert_allclose(series_data["train_y_std"], training_values.std())

    def test_four_way_ratios_are_validated(self):
        cases = (
            {"calibration_ratio": 0.1, "test_ratio": None},
            {"calibration_ratio": 0.1, "test_ratio": 0.3},
            {"calibration_ratio": 0.0, "test_ratio": 0.3},
        )
        for extra_ratios in cases:
            with self.subTest(ratios=extra_ratios):
                prepared = ConformalPredictionData(self._artifact())
                with self.assertRaises(ValueError):
                    prepared.prepare_quantile_regression_datasets(
                        past_window=5,
                        prediction_steps=1,
                        train_ratio=0.6,
                        valid_ratio=0.1,
                        **extra_ratios,
                    )

    def test_four_way_split_rejects_multi_step_targets(self):
        prepared = ConformalPredictionData(self._artifact())
        with self.assertRaisesRegex(ValueError, "prediction_steps=1"):
            prepared.prepare_quantile_regression_datasets(
                past_window=5,
                prediction_steps=2,
                train_ratio=0.6,
                valid_ratio=0.1,
                calibration_ratio=0.1,
                test_ratio=0.2,
            )


class LocalCPEncodingTests(unittest.TestCase):
    @staticmethod
    def _dataloader():
        strided_x = torch.zeros(5, 3, 1)
        strided_residual = torch.tensor(
            [
                [1.0, 2.0, 98.0],
                [3.0, 4.0, 99.0],
                [5.0, 6.0, 100.0],
                [7.0, 8.0, 101.0],
                [9.0, 10.0, 102.0],
            ]
        )
        strided_y = torch.zeros(5, 3)
        target_x = torch.zeros(5, 1, 1)
        target_residual = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]])
        target_y = torch.zeros(5, 1)
        target_predictions = torch.zeros(5, 1)
        dataset = TensorDataset(
            strided_x,
            strided_residual,
            strided_y,
            target_x,
            target_residual,
            target_y,
            target_predictions,
        )
        return DataLoader(dataset, batch_size=2, shuffle=False)

    def test_rnn_encoder_exports_target_residual(self):
        model = RNNPredictor(
            rnn_type="rnn",
            dim_feature=1,
            dim_model=2,
            num_layer=1,
            prediction_step=1,
            dropout=0.0,
            training_quantiles=[0.1, 0.5, 0.9],
        )
        _, residuals = model.encode_dataloader(
            model,
            self._dataloader(),
            strided_features="r",
            device="cpu",
        )
        torch.testing.assert_close(
            residuals,
            torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]]),
        )

    def test_transformer_encoder_exports_target_residual(self):
        model = TransformerPredictor(
            dim_feature=1,
            dim_model=2,
            num_head=1,
            dim_ff=4,
            num_layer=1,
            prediction_step=1,
            dropout=0.0,
            training_quantiles=[0.1, 0.5, 0.9],
        )
        _, residuals = model.encode_dataloader(
            model,
            self._dataloader(),
            strided_features="r",
            device="cpu",
        )
        torch.testing.assert_close(
            residuals,
            torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]]),
        )


if __name__ == "__main__":
    unittest.main()
