import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from base_predictor.data import BasePredictorData
from sbatch_run_base_predictor.common import DATA_TYPE_CHOICES, default_data_dir


SAPFLUX_FEATURE_COLUMNS = (
    "ta",
    "rh",
    "sw_in",
    "ppfd_in",
    "ws",
    "precip",
    "swc_shallow",
    "swc_deep",
    "ext_rad",
    "vpd",
)


def _sapflux_frame(num_rows, cadence_minutes=10, target_name="tree_flux"):
    row = np.arange(num_rows, dtype=np.float32)
    data = {
        "solar_TIMESTAMP": pd.date_range(
            "2016-05-01", periods=num_rows, freq=f"{cadence_minutes}min"
        ),
        target_name: row + np.float32(0.25),
    }
    for offset, column in enumerate(SAPFLUX_FEATURE_COLUMNS, start=1):
        data[column] = row + np.float32(offset * 100_000)
    return pd.DataFrame(data)


class SapfluxDataTests(unittest.TestCase):
    def _load_mock_frames(self, frames):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        data_dir = Path(temporary_directory.name)
        for file_name in frames:
            (data_dir / file_name).touch()

        def fake_read_csv(path, *args, **kwargs):
            return frames[Path(path).name].copy(deep=True)

        data = BasePredictorData()
        with patch("base_predictor.data.pd.read_csv", side_effect=fake_read_csv):
            data.load_data("sapflux-solo3-large", data_dir)
        return data

    def test_loads_scalar_target_and_exact_ten_features_without_resampling(self):
        frame = _sapflux_frame(15_000, cadence_minutes=10, target_name="CZE_tree_13")

        loaded = self._load_mock_frames({"czech_tree.csv": frame})
        series = loaded.data["czech_tree"]

        self.assertEqual(loaded.data_type, "sapflux-solo3-large")
        self.assertEqual(series["x"].shape, (15_000, 10))
        self.assertEqual(series["y"].shape, (15_000,))
        self.assertEqual(series["x"].dtype, np.float32)
        self.assertEqual(series["y"].dtype, np.float32)

        # The first feature is a row marker. Keeping every marker in order proves
        # that the native observations were neither resampled nor downsampled.
        np.testing.assert_array_equal(
            series["x"][:, 0],
            np.arange(15_000, dtype=np.float32) + np.float32(100_000),
        )
        np.testing.assert_array_equal(
            series["y"],
            np.arange(15_000, dtype=np.float32) + np.float32(0.25),
        )

    def test_large_variant_uses_inclusive_15000_to_20000_row_filter(self):
        frames = {
            "too_short.csv": _sapflux_frame(14_999),
            "lower_boundary.csv": _sapflux_frame(15_000),
            "upper_boundary.csv": _sapflux_frame(20_000),
            "too_long.csv": _sapflux_frame(20_001),
        }

        loaded = self._load_mock_frames(frames)

        self.assertEqual(
            set(loaded.data),
            {"lower_boundary", "upper_boundary"},
        )

    def test_missing_environmental_feature_is_rejected(self):
        frame = _sapflux_frame(15_000).drop(columns="vpd")
        frame["not_vpd"] = np.arange(len(frame), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "(?i)(feature|vpd)"):
            self._load_mock_frames({"missing_vpd.csv": frame})

    def test_shared_base_predictor_cli_exposes_bundled_dataset(self):
        self.assertIn("sapflux-solo3-large", DATA_TYPE_CHOICES)
        self.assertEqual(
            default_data_dir("sapflux-solo3-large"),
            Path(__file__).resolve().parents[1]
            / "data"
            / "sapflux"
            / "0.1.5"
            / "prepared",
        )


if __name__ == "__main__":
    unittest.main()
