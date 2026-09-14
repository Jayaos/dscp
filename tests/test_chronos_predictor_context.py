import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


# Importing the predictor should not require the optional Chronos dependency or
# instantiate/download a pretrained model during unit tests.
chronos_stub = ModuleType("chronos")


class _UnusedChronos2Pipeline:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise AssertionError("This unit test must not load a pretrained model")


chronos_stub.Chronos2Pipeline = _UnusedChronos2Pipeline
with patch.dict(sys.modules, {"chronos": chronos_stub}):
    from base_predictor import chronos_predictor


class _RecordingPipeline:
    def __init__(self):
        self.calls = []

    def predict_df(self, context_df, *args, **kwargs):
        self.calls.append((context_df.copy(deep=True), args, dict(kwargs)))
        return pd.DataFrame(
            {
                "predictions": np.zeros(
                    kwargs["prediction_length"], dtype=np.float32
                )
            }
        )


class ChronosHistoricalContextTests(unittest.TestCase):
    def test_predict_df_receives_only_features_before_each_forecast_block(self):
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
        data = SimpleNamespace(data_type="toy", data={"series": {"x": x, "y": y}})

        pipeline = _RecordingPipeline()
        predictor = chronos_predictor.ChronosPredictor.__new__(
            chronos_predictor.ChronosPredictor
        )
        predictor.chronos2 = pipeline
        predictor.data = data.data
        predictor.data_type = data.data_type
        predictor.predictions = {}

        with patch.object(chronos_predictor, "tqdm", new=lambda values: values):
            predictor.predict(window_length=2, prediction_length=2)

        self.assertEqual(len(pipeline.calls), 2)
        expected_contexts = (
            (x[0:2], y[0:2], 2),
            (x[2:4], y[2:4], 1),
        )
        for (context_df, positional_args, keyword_args), (
            expected_x,
            expected_y,
            expected_prediction_length,
        ) in zip(pipeline.calls, expected_contexts):
            with self.subTest(prediction_length=expected_prediction_length):
                self.assertEqual(positional_args, ())
                self.assertNotIn("future_df", keyword_args)
                self.assertEqual(
                    keyword_args["prediction_length"], expected_prediction_length
                )
                self.assertEqual(keyword_args["quantile_levels"], [0.5])
                np.testing.assert_array_equal(
                    context_df[["feat_0", "feat_1"]].to_numpy(), expected_x
                )
                np.testing.assert_array_equal(
                    context_df["target"].to_numpy(), expected_y
                )


if __name__ == "__main__":
    unittest.main()
