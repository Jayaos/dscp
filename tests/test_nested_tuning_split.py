import unittest
from pathlib import Path

import numpy as np

from dscp.data import ConformalPredictionData


class NestedTuningSplitTests(unittest.TestCase):
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

    def _prepare(self, train_ratio, valid_ratio, normalize=False, length=100):
        prepared = ConformalPredictionData(self._artifact(length=length))
        prepared.prepare_quantile_regression_datasets(
            past_window=5,
            prediction_steps=1,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            normalize=normalize,
            model_selection_valid_ratio=0.2,
        )
        return prepared

    def _prepare_local(self, normalize=False, length=100):
        prepared = ConformalPredictionData(self._artifact(length=length))
        prepared.prepare_quantile_regression_datasets(
            past_window=5,
            prediction_steps=1,
            train_ratio=0.6,
            valid_ratio=0.1,
            normalize=normalize,
            calibration_ratio=0.1,
            test_ratio=0.2,
            model_selection_valid_ratio=0.2,
        )
        return prepared

    def _assert_targets(self, dataset, expected):
        actual = dataset.target_residual.squeeze(-1).numpy()
        np.testing.assert_array_equal(actual, np.arange(*expected))

    def test_qr_nested_split_uses_nominal_validation_not_test(self):
        prepared = self._prepare(train_ratio=0.5, valid_ratio=0.16)
        datasets = prepared.dataset["series"]

        self._assert_targets(datasets["train_dataset"], (5, 40))
        self._assert_targets(
            datasets["model_selection_valid_dataset"], (40, 50)
        )
        self._assert_targets(datasets["tuning_evaluation_dataset"], (50, 66))
        self.assertNotIn("test_dataset", datasets)

        metadata = prepared.data["series"]
        self.assertEqual(metadata["nominal_train_size"], 50)
        self.assertEqual(metadata["train_size"], 40)
        self.assertEqual(metadata["model_selection_valid_size"], 10)
        self.assertEqual(metadata["tuning_evaluation_size"], 16)
        self.assertEqual(metadata["test_size"], 34)

    def test_iqn_nested_split_uses_nominal_validation_not_test(self):
        prepared = self._prepare(train_ratio=0.6, valid_ratio=0.2)
        datasets = prepared.dataset["series"]

        self._assert_targets(datasets["train_dataset"], (5, 48))
        self._assert_targets(
            datasets["model_selection_valid_dataset"], (48, 60)
        )
        self._assert_targets(datasets["tuning_evaluation_dataset"], (60, 80))
        self.assertNotIn("test_dataset", datasets)

        metadata = prepared.data["series"]
        self.assertEqual(metadata["nominal_train_size"], 60)
        self.assertEqual(metadata["train_size"], 48)
        self.assertEqual(metadata["model_selection_valid_size"], 12)
        self.assertEqual(metadata["tuning_evaluation_size"], 20)
        self.assertEqual(metadata["test_size"], 20)

    def test_local_nested_split_has_separate_calibration_and_no_test(self):
        prepared = self._prepare_local()
        datasets = prepared.dataset["series"]

        self._assert_targets(datasets["train_dataset"], (5, 48))
        self._assert_targets(
            datasets["model_selection_valid_dataset"], (48, 60)
        )
        self._assert_targets(datasets["calibration_dataset"], (60, 70))
        self._assert_targets(datasets["tuning_evaluation_dataset"], (70, 80))
        self.assertNotIn("test_dataset", datasets)

        metadata = prepared.data["series"]
        self.assertEqual(metadata["nominal_train_size"], 60)
        self.assertEqual(metadata["train_size"], 48)
        self.assertEqual(metadata["model_selection_valid_size"], 12)
        self.assertEqual(metadata["calibration_size"], 10)
        self.assertEqual(metadata["tuning_evaluation_size"], 10)
        self.assertEqual(metadata["test_size"], 20)

    def test_nondivisible_length_preserves_outer_split_rounding(self):
        prepared = self._prepare(
            train_ratio=0.5,
            valid_ratio=0.16,
            length=101,
        )
        datasets = prepared.dataset["series"]

        self._assert_targets(datasets["train_dataset"], (5, 40))
        self._assert_targets(
            datasets["model_selection_valid_dataset"], (40, 50)
        )
        self._assert_targets(datasets["tuning_evaluation_dataset"], (50, 67))
        self.assertNotIn("test_dataset", datasets)

        metadata = prepared.data["series"]
        self.assertEqual(metadata["nominal_train_size"], 50)
        self.assertEqual(metadata["tuning_evaluation_size"], 17)
        self.assertEqual(metadata["test_size"], 34)

    def test_normalization_statistics_use_only_inner_fit_prefix(self):
        for train_ratio, valid_ratio, fit_end in (
            (0.5, 0.16, 40),
            (0.6, 0.2, 48),
        ):
            with self.subTest(train_ratio=train_ratio):
                artifact = self._artifact()
                artifact["series"]["heldout_x"][fit_end:] += 10_000.0
                artifact["series"]["heldout_y"][fit_end:] += 20_000.0
                prepared = ConformalPredictionData(artifact)
                prepared.prepare_quantile_regression_datasets(
                    past_window=5,
                    prediction_steps=1,
                    train_ratio=train_ratio,
                    valid_ratio=valid_ratio,
                    normalize=True,
                    model_selection_valid_ratio=0.2,
                )

                expected = np.arange(fit_end, dtype=np.float32)
                metadata = prepared.data["series"]
                np.testing.assert_allclose(metadata["train_x_mu"], expected.mean())
                np.testing.assert_allclose(metadata["train_x_std"], expected.std())
                np.testing.assert_allclose(metadata["train_y_mu"], expected.mean())
                np.testing.assert_allclose(metadata["train_y_std"], expected.std())

    def test_local_tuning_normalization_uses_only_inner_fit_prefix(self):
        artifact = self._artifact()
        artifact["series"]["heldout_x"][48:] += 10_000.0
        artifact["series"]["heldout_y"][48:] += 20_000.0
        prepared = ConformalPredictionData(artifact)
        prepared.prepare_quantile_regression_datasets(
            past_window=5,
            prediction_steps=1,
            train_ratio=0.6,
            valid_ratio=0.1,
            calibration_ratio=0.1,
            test_ratio=0.2,
            normalize=True,
            model_selection_valid_ratio=0.2,
        )

        expected = np.arange(48, dtype=np.float32)
        metadata = prepared.data["series"]
        np.testing.assert_allclose(metadata["train_x_mu"], expected.mean())
        np.testing.assert_allclose(metadata["train_x_std"], expected.std())
        np.testing.assert_allclose(metadata["train_y_mu"], expected.mean())
        np.testing.assert_allclose(metadata["train_y_std"], expected.std())

    def test_local_tuning_datasets_do_not_depend_on_reserved_test_values(self):
        baseline = self._prepare_local(normalize=True)
        modified_artifact = self._artifact()
        modified_artifact["series"]["heldout_x"][80:] += 10_000.0
        modified_artifact["series"]["heldout_y"][80:] += 20_000.0
        modified_artifact["series"]["heldout_predictions"][80:] -= 30_000.0
        modified = ConformalPredictionData(modified_artifact)
        modified.prepare_quantile_regression_datasets(
            past_window=5,
            prediction_steps=1,
            train_ratio=0.6,
            valid_ratio=0.1,
            calibration_ratio=0.1,
            test_ratio=0.2,
            normalize=True,
            model_selection_valid_ratio=0.2,
        )

        for split_name in (
            "train_dataset",
            "model_selection_valid_dataset",
            "calibration_dataset",
            "tuning_evaluation_dataset",
        ):
            with self.subTest(split=split_name):
                baseline_dataset = baseline.dataset["series"][split_name]
                modified_dataset = modified.dataset["series"][split_name]
                for field_name in (
                    "strided_x",
                    "strided_residual",
                    "strided_y",
                    "target_x",
                    "target_residual",
                    "target_y",
                    "target_predictions",
                ):
                    np.testing.assert_array_equal(
                        getattr(baseline_dataset, field_name).numpy(),
                        getattr(modified_dataset, field_name).numpy(),
                    )

    def test_invalid_nested_ratios_are_rejected(self):
        for inner_ratio in (0.0, 1.0, -0.1, np.nan):
            with self.subTest(model_selection_valid_ratio=inner_ratio):
                prepared = ConformalPredictionData(self._artifact())
                with self.assertRaises(ValueError):
                    prepared.prepare_quantile_regression_datasets(
                        past_window=5,
                        prediction_steps=1,
                        train_ratio=0.5,
                        valid_ratio=0.16,
                        model_selection_valid_ratio=inner_ratio,
                    )

        prepared = ConformalPredictionData(self._artifact())
        with self.assertRaises(ValueError):
            prepared.prepare_quantile_regression_datasets(
                past_window=5,
                prediction_steps=1,
                train_ratio=0.8,
                valid_ratio=0.2,
                model_selection_valid_ratio=0.2,
            )

    def test_nested_split_rejects_multi_step_targets(self):
        prepared = ConformalPredictionData(self._artifact())
        with self.assertRaisesRegex(ValueError, "prediction_steps=1"):
            prepared.prepare_quantile_regression_datasets(
                past_window=5,
                prediction_steps=2,
                train_ratio=0.5,
                valid_ratio=0.16,
                model_selection_valid_ratio=0.2,
            )

    def test_invalid_local_nested_split_arguments_are_rejected(self):
        invalid_ratio_sets = (
            (0.6, 0.1, 0.1, 0.3),
            (0.6, 0.1, 0.0, 0.3),
            (0.6, 0.1, np.nan, 0.2),
        )
        for train_ratio, valid_ratio, calibration_ratio, test_ratio in invalid_ratio_sets:
            with self.subTest(
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                calibration_ratio=calibration_ratio,
                test_ratio=test_ratio,
            ):
                prepared = ConformalPredictionData(self._artifact())
                with self.assertRaises(ValueError):
                    prepared.prepare_quantile_regression_datasets(
                        past_window=5,
                        prediction_steps=1,
                        train_ratio=train_ratio,
                        valid_ratio=valid_ratio,
                        calibration_ratio=calibration_ratio,
                        test_ratio=test_ratio,
                        model_selection_valid_ratio=0.2,
                    )

        prepared = ConformalPredictionData(self._artifact())
        with self.assertRaises(ValueError):
            prepared.prepare_quantile_regression_datasets(
                past_window=5,
                prediction_steps=1,
                train_ratio=0.6,
                valid_ratio=0.1,
                calibration_ratio=0.1,
                model_selection_valid_ratio=0.2,
            )


class TuningRunnerIsolationTests(unittest.TestCase):
    def test_qr_iqn_and_local_tuners_do_not_reference_a_test_dataset(self):
        repository_root = Path(__file__).resolve().parents[1]
        for relative_path in (
            "sbatch/sbatch_run_tuning/run_qr_cp_tuning.py",
            "sbatch/sbatch_run_tuning/run_iqn_cp_tuning.py",
            "sbatch/sbatch_run_tuning/run_local_cp_tuning.py",
        ):
            with self.subTest(runner=relative_path):
                source = (repository_root / relative_path).read_text(encoding="utf-8")
                self.assertNotIn("test_dataset", source)
                self.assertIn("model_selection_valid_dataset", source)
                self.assertIn("tuning_evaluation_dataset", source)

    def test_local_tuning_launchers_use_current_paths(self):
        repository_root = Path(__file__).resolve().parents[1]
        launcher_names = (
            "run_lcp_rnn_chronos_air_tuning.sbatch",
            "run_lcp_transformer_chronos_air_tuning.sbatch",
        )
        for launcher_name in launcher_names:
            with self.subTest(launcher=launcher_name):
                source = (
                    repository_root / "sbatch" / "sbatch_run_tuning" / launcher_name
                ).read_text(encoding="utf-8")
                self.assertIn('$RUNPATH/sbatch:$PYTHONPATH', source)
                self.assertIn(
                    "python sbatch/sbatch_run_tuning/run_local_cp_tuning.py",
                    source,
                )

    def test_local_tuning_configs_enable_nested_checkpoint_split(self):
        repository_root = Path(__file__).resolve().parents[1]
        config_names = (
            "lcp_rnn_air_tuning_config.yaml",
            "lcp_transformer_air_tuning_config.yaml",
        )
        for config_name in config_names:
            with self.subTest(config=config_name):
                source = (
                    repository_root / "configs" / "lcp_configs" / config_name
                ).read_text(encoding="utf-8")
                self.assertIn("model_selection_valid_ratio: 0.2", source)


if __name__ == "__main__":
    unittest.main()
