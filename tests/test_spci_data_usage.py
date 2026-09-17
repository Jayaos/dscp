"""SPCI's final run fits one training prefix and evaluates its full suffix."""

import copy
import unittest

import numpy as np
from omegaconf import OmegaConf

from baselines.spci.data import prepare_spci_data


class SPCIDataUsageTests(unittest.TestCase):
    @staticmethod
    def artifact(length=100):
        values = np.arange(length, dtype=np.float32)
        return {"series": {
            "heldout_x": np.column_stack((values, values * 2)),
            "heldout_y": values,
            "heldout_predictions": np.zeros_like(values),
        }}

    @staticmethod
    def config(window=5, train_ratio=0.66, normalize=False):
        return OmegaConf.create({
            "model": {"window_size": window, "prediction_step": 1},
            "data": {"train_ratio": train_ratio, "normalize": normalize},
        })

    def test_final_split_has_exact_training_boundary_and_all_remaining_test_targets(self):
        for length, ratio, train_end in ((100, 0.66, 66), (101, 0.66, 66), (101, 0.5, 50)):
            for window in (1, 5, 30):
                with self.subTest(length=length, ratio=ratio, window=window):
                    prepared = prepare_spci_data(self.artifact(length), self.config(window, ratio))
                    datasets = prepared.dataset["series"]
                    self.assertEqual(set(datasets), {"train_dataset", "test_dataset"})
                    np.testing.assert_array_equal(
                        datasets["train_dataset"].target_y.flatten().numpy(), np.arange(window, train_end)
                    )
                    np.testing.assert_array_equal(
                        datasets["test_dataset"].target_y.flatten().numpy(), np.arange(train_end, length)
                    )
                    np.testing.assert_array_equal(
                        datasets["test_dataset"].strided_residual[0].numpy(),
                        np.arange(train_end - window, train_end),
                    )

    def test_saved_tuning_setting_does_not_reduce_final_training_prefix(self):
        config = self.config()
        config.tuning = {"model_selection_valid_ratio": 0.5}
        final = prepare_spci_data(self.artifact(), config)
        self.assertEqual(len(final.dataset["series"]["train_dataset"]), 61)
        self.assertEqual(len(final.dataset["series"]["test_dataset"]), 34)
        tuning = prepare_spci_data(self.artifact(), config, model_selection_valid_ratio=0.5)
        self.assertEqual(set(tuning.dataset["series"]), {
            "train_dataset", "model_selection_valid_dataset",
        })
        self.assertEqual(len(tuning.dataset["series"]["train_dataset"]), 28)
        np.testing.assert_array_equal(
            tuning.dataset["series"]["model_selection_valid_dataset"].target_y.flatten().numpy(),
            np.arange(33, 66),
        )

    def test_final_normalization_uses_entire_training_prefix_only_and_preserves_input(self):
        artifact = self.artifact()
        artifact["series"]["heldout_y"][66:] += 100_000
        artifact["series"]["heldout_x"][66:] -= 100_000
        original = copy.deepcopy(artifact)
        prepared = prepare_spci_data(artifact, self.config(normalize=True))
        metadata = prepared.data["series"]
        np.testing.assert_allclose(metadata["train_y_mu"], np.arange(66).mean())
        np.testing.assert_allclose(metadata["train_y_std"], np.arange(66).std(), rtol=1e-6)
        np.testing.assert_allclose(metadata["train_x_mu"], original["series"]["heldout_x"][:66].mean(axis=0))
        np.testing.assert_array_equal(
            prepared.dataset["series"]["test_dataset"].target_y.flatten().numpy(),
            original["series"]["heldout_y"][66:],
        )
        self.assertEqual(set(artifact["series"]), set(original["series"]))
        for key in original["series"]:
            np.testing.assert_array_equal(artifact["series"][key], original["series"][key])

    def test_test_targets_do_not_leak_into_training_or_their_own_residual_context(self):
        baseline = prepare_spci_data(self.artifact(), self.config(normalize=True))
        changed_artifact = self.artifact()
        changed_artifact["series"]["heldout_y"][66:] += 100_000
        changed = prepare_spci_data(changed_artifact, self.config(normalize=True))
        for field in ("strided_residual", "target_residual", "target_y"):
            np.testing.assert_array_equal(
                getattr(baseline.dataset["series"]["train_dataset"], field).numpy(),
                getattr(changed.dataset["series"]["train_dataset"], field).numpy(),
            )
        np.testing.assert_array_equal(
            baseline.dataset["series"]["test_dataset"].strided_residual[0].numpy(),
            changed.dataset["series"]["test_dataset"].strided_residual[0].numpy(),
        )
        self.assertNotEqual(
            baseline.dataset["series"]["test_dataset"].target_residual[0].item(),
            changed.dataset["series"]["test_dataset"].target_residual[0].item(),
        )

    def test_sequence_order_and_mixed_vector_column_targets_are_preserved(self):
        for column_y, column_predictions in ((True, True), (True, False), (False, True)):
            with self.subTest(column_y=column_y, column_predictions=column_predictions):
                artifact = {"second": self.artifact(101)["series"], "first": self.artifact()["series"]}
                for item in artifact.values():
                    if column_y:
                        item["heldout_y"] = item["heldout_y"][:, None]
                    if column_predictions:
                        item["heldout_predictions"] = item["heldout_predictions"][:, None]
                prepared = prepare_spci_data(artifact, self.config())
                self.assertEqual(list(prepared.dataset), ["second", "first"])
                for key, length in (("second", 101), ("first", 100)):
                    np.testing.assert_array_equal(
                        prepared.dataset[key]["test_dataset"].target_y.flatten().numpy(), np.arange(66, length)
                    )
                    np.testing.assert_array_equal(
                        prepared.dataset[key]["train_dataset"].target_residual.flatten().numpy(),
                        np.arange(5, 66),
                    )

    def test_invalid_train_ratio_window_horizon_and_empty_sequences_are_rejected(self):
        for ratio in (0, 1, -0.5, np.nan, np.inf, True, None):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, "train_ratio"):
                prepare_spci_data(self.artifact(), self.config(train_ratio=ratio))
        for window in (0, -1, True, 1.5):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "window_size"):
                prepare_spci_data(self.artifact(), self.config(window=window))
        for window in (66, 100):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "Insufficient data"):
                prepare_spci_data(self.artifact(), self.config(window=window))
        for horizon in (0, 2, True, 1.0):
            config = self.config()
            config.model.prediction_step = horizon
            with self.subTest(horizon=horizon), self.assertRaisesRegex(ValueError, "prediction_step=1"):
                prepare_spci_data(self.artifact(), config)
        with self.assertRaisesRegex(ValueError, "at least one sequence"):
            prepare_spci_data({}, self.config())
        with self.assertRaisesRegex(ValueError, "Insufficient data"):
            prepare_spci_data(self.artifact(0), self.config())

    def test_obsolete_validation_fraction_requires_explicit_migration(self):
        config = self.config()
        config.data.valid_ratio = 0.16
        with self.assertRaises(ValueError) as raised:
            prepare_spci_data(self.artifact(), config)
        self.assertIn("valid_ratio", str(raised.exception))
        self.assertIn("train_ratio", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
