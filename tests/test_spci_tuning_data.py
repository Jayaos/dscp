"""Chronological isolation checks for SPCI's inner-training tuning split."""

import copy
import unittest

import numpy as np
from omegaconf import OmegaConf

from baselines.spci.tuning_data import prepare_spci_tuning_data


class SPCITuningDataTests(unittest.TestCase):
    @staticmethod
    def artifact(length=100):
        values = np.arange(length, dtype=np.float32)
        return {"series": {
            "heldout_x": np.column_stack((values, values * 2)),
            "heldout_y": values,
            "heldout_predictions": np.zeros_like(values),
        }}

    @staticmethod
    def config(window=5, ratio=0.2, normalize=False):
        return OmegaConf.create({
            "model": {"window_size": window, "prediction_step": 1},
            "data": {"train_ratio": 0.5, "valid_ratio": 0.16, "normalize": normalize},
            "tuning": {"model_selection_valid_ratio": ratio},
        })

    def test_tuning_uses_only_later_fraction_of_nominal_training_region(self):
        for length in (100, 101):
            for ratio, fit_end in ((0.2, 40), (0.3, 35)):
                with self.subTest(length=length, ratio=ratio):
                    prepared = prepare_spci_tuning_data(self.artifact(length), self.config(ratio=ratio))
                    datasets = prepared.dataset["series"]
                    self.assertEqual(set(datasets), {"train_dataset", "model_selection_valid_dataset"})
                    np.testing.assert_array_equal(
                        datasets["train_dataset"].target_y.flatten().numpy(), np.arange(5, fit_end)
                    )
                    np.testing.assert_array_equal(
                        datasets["model_selection_valid_dataset"].target_y.flatten().numpy(),
                        np.arange(fit_end, 50),
                    )

    def test_windows_change_fit_count_but_preserve_evaluation_timestamps(self):
        for window in (1, 5, 20, 39):
            with self.subTest(window=window):
                prepared = prepare_spci_tuning_data(self.artifact(), self.config(window=window))
                datasets = prepared.dataset["series"]
                self.assertEqual(len(datasets["train_dataset"]), 40 - window)
                evaluation = datasets["model_selection_valid_dataset"]
                np.testing.assert_array_equal(evaluation.target_y.flatten().numpy(), np.arange(40, 50))
                np.testing.assert_array_equal(
                    evaluation.strided_residual[0].numpy(), np.arange(40 - window, 40)
                )

    def test_outer_validation_and_test_values_cannot_change_tuning_examples(self):
        artifact = self.artifact()
        modified = copy.deepcopy(artifact)
        for name in ("heldout_x", "heldout_y", "heldout_predictions"):
            modified["series"][name][50:] += 100_000
        baseline = prepare_spci_tuning_data(artifact, self.config(normalize=True))
        changed = prepare_spci_tuning_data(modified, self.config(normalize=True))
        for split in ("train_dataset", "model_selection_valid_dataset"):
            for field in (
                "strided_x", "strided_y", "strided_residual", "target_x",
                "target_residual", "target_y", "target_predictions",
            ):
                with self.subTest(split=split, field=field):
                    np.testing.assert_array_equal(
                        getattr(baseline.dataset["series"][split], field).numpy(),
                        getattr(changed.dataset["series"][split], field).numpy(),
                    )

    def test_normalization_uses_fit_prefix_and_input_is_not_mutated(self):
        artifact = self.artifact()
        artifact["series"]["heldout_x"][40:] += 10_000
        artifact["series"]["heldout_y"][40:] += 20_000
        original = copy.deepcopy(artifact)
        prepared = prepare_spci_tuning_data(artifact, self.config(normalize=True))
        metadata = prepared.data["series"]
        expected_y = np.arange(40, dtype=np.float32)
        np.testing.assert_allclose(metadata["train_y_mu"], expected_y.mean())
        np.testing.assert_allclose(metadata["train_y_std"], expected_y.std())
        np.testing.assert_allclose(metadata["train_x_mu"], artifact["series"]["heldout_x"][:40].mean(axis=0))
        self.assertEqual(set(artifact["series"]), set(original["series"]))
        for key in original["series"]:
            np.testing.assert_array_equal(artifact["series"][key], original["series"][key])

    def test_invalid_ratios_and_windows_are_rejected(self):
        for ratio in (0, 1, -0.1, np.nan, np.inf, None, True):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, "model_selection_valid_ratio"):
                prepare_spci_tuning_data(self.artifact(), self.config(ratio=ratio))
        for window in (0, -1, 1.5, True):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "window_size"):
                prepare_spci_tuning_data(self.artifact(), self.config(window=window))
        for window in (40, 100):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "Insufficient data"):
                prepare_spci_tuning_data(self.artifact(), self.config(window=window))

    def test_invalid_horizon_outer_split_and_sequence_lengths_are_rejected(self):
        for horizon in (0, 2, True, 1.0):
            config = self.config()
            config.model.prediction_step = horizon
            with self.subTest(horizon=horizon), self.assertRaisesRegex(ValueError, "prediction_step=1"):
                prepare_spci_tuning_data(self.artifact(), config)
        for train_ratio, valid_ratio in ((0, 0.16), (0.5, np.inf), (0.8, 0.2)):
            config = self.config()
            config.data.train_ratio, config.data.valid_ratio = train_ratio, valid_ratio
            with self.subTest(train=train_ratio, valid=valid_ratio), self.assertRaises(ValueError):
                prepare_spci_tuning_data(self.artifact(), config)
        with self.assertRaisesRegex(ValueError, "at least one sequence"):
            prepare_spci_tuning_data({}, self.config())
        artifact = self.artifact()
        artifact["series"]["heldout_predictions"] = np.zeros(99)
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            prepare_spci_tuning_data(artifact, self.config())


if __name__ == "__main__":
    unittest.main()
