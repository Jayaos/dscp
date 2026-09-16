import contextlib
import importlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf


def _import_tuning_module(name):
    sbatch_path = str(Path(__file__).resolve().parents[1] / "sbatch")
    with patch.object(sys, "path", [sbatch_path, *sys.path]):
        return importlib.import_module(f"sbatch_run_tuning.{name}")


common = _import_tuning_module("common")


class TuningSequenceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.dataset = {"zeta": {}, "alpha": {}, "mu": {}}

    def test_resolver_preserves_default_and_numeric_counts(self):
        for config in (None, {}, OmegaConf.create({})):
            with self.subTest(config=config):
                self.assertEqual(common.resolve_num_sequences(config), 1)
        for value in (2, "2"):
            with self.subTest(value=value):
                config = OmegaConf.create({"num_sequences": value})
                resolved = common.resolve_num_sequences(config)
                self.assertEqual(resolved, 2)
                self.assertIsInstance(resolved, int)

    def test_resolver_normalizes_all_from_omegaconf(self):
        for value in ("all", "ALL", " All "):
            with self.subTest(value=value):
                config = OmegaConf.create({"num_sequences": value})
                self.assertEqual(common.resolve_num_sequences(config), "all")

    def test_resolver_rejects_invalid_values(self):
        for value in (0, -1, "0", "-2", "", "everything", None, [], {}):
            with self.subTest(value=value):
                config = OmegaConf.create({"num_sequences": value})
                with self.assertRaises(ValueError):
                    common.resolve_num_sequences(config)

    def test_all_returns_every_sorted_key_regardless_of_index(self):
        for index in (0, 1, -1, 99):
            with self.subTest(index=index):
                self.assertEqual(
                    common.choose_sequence_keys(self.dataset, None, index, "all"),
                    ["alpha", "mu", "zeta"],
                )

    def test_explicit_key_overrides_all_and_count(self):
        for count in ("all", 2):
            with self.subTest(count=count):
                self.assertEqual(
                    common.choose_sequence_keys(self.dataset, "mu", 99, count),
                    ["mu"],
                )
                with self.assertRaisesRegex(ValueError, "was not found"):
                    common.choose_sequence_keys(self.dataset, "missing", 0, count)

    def test_numeric_counts_keep_existing_slice_and_bounds(self):
        self.assertEqual(
            common.choose_sequence_keys(self.dataset, None, 1, 2),
            ["mu", "zeta"],
        )
        for index in (-1, 3):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, "out of range"):
                    common.choose_sequence_keys(self.dataset, None, index, 1)
        with self.assertRaisesRegex(ValueError, "only 2 are available"):
            common.choose_sequence_keys(self.dataset, None, 1, 3)

    def test_empty_dataset_is_rejected_for_all_and_count(self):
        for count in ("all", 1):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "No sequence keys"):
                    common.choose_sequence_keys({}, None, 0, count)


class TuningSequenceOrchestrationTests(unittest.TestCase):
    @staticmethod
    def _base_config():
        return OmegaConf.create(
            {
                "model": {
                    "window_size": 3,
                    "prediction_step": 1,
                    "target_quantiles": [[0.1, 0.9]],
                },
                "data": {
                    "data_path": "unused_data.pkl",
                    "train_ratio": 0.5,
                    "valid_ratio": 0.2,
                    "calibration_ratio": 0.1,
                    "test_ratio": 0.2,
                    "normalize": False,
                },
            }
        )

    @staticmethod
    def _sequence_result():
        return {
            "best_valid_loss": 0.1,
            "best_epoch": 1,
            "prediction_head": "iqn",
            "pair_metrics": {
                "(0.1, 0.9)": {
                    "avg_coverage": 0.9,
                    "target_coverage": 0.8,
                    "avg_delta_coverage": 0.1,
                    "avg_interval_width": 1.0,
                    "avg_winkler_score": 1.0,
                }
            },
            "selection_score": 1.0,
            "positive_delta_coverage": True,
        }

    def test_all_three_tuners_select_and_report_actual_sequence_count(self):
        cases = (
            ("all", None, 99, ["alpha", "mu", "zeta"]),
            ("all", "mu", 99, ["mu"]),
            (2, None, 1, ["mu", "zeta"]),
        )
        for method in ("qr_cp", "iqn_cp", "local_cp"):
            tuning = _import_tuning_module(f"run_{method}_tuning")
            for count, sequence_key, index, expected_keys in cases:
                with self.subTest(method=method, count=count, sequence_key=sequence_key):
                    args = SimpleNamespace(
                        save_dir=Path("unused_tuning_results"),
                        base_config=Path("unused_base.yaml"),
                        grid_config=Path("unused_grid.yaml"),
                        sequence_key=sequence_key,
                        sequence_index=index,
                        top_k=3,
                        seed=42,
                    )
                    base_config = self._base_config()
                    grid_config = OmegaConf.create(
                        {
                            "grid": {"model.window_size": [3]},
                            "tuning": {"num_sequences": count},
                        }
                    )
                    data = {"zeta": {}, "alpha": {}, "mu": {}}

                    def prepare(selected_data):
                        self.assertEqual(list(selected_data), expected_keys)
                        return SimpleNamespace(
                            data=selected_data,
                            dataset={key: {"key": key} for key in selected_data},
                            prepare_quantile_regression_datasets=Mock(),
                        )

                    with (
                        patch.object(tuning, "parse_args", return_value=args),
                        patch.object(
                            OmegaConf,
                            "load",
                            side_effect=lambda path: (
                                base_config if path == args.base_config else grid_config
                            ),
                        ),
                        patch.object(tuning, "load_data", return_value=data),
                        patch.object(tuning, "ConformalPredictionData", side_effect=prepare),
                        patch.object(tuning, "set_global_seed"),
                        patch.object(
                            tuning, "_run_single_trial", return_value=self._sequence_result()
                        ) as train,
                        patch.object(Path, "mkdir"),
                        patch.object(tuning, "write_trial_artifacts") as write_trial,
                        patch.object(tuning, "finalize_and_save_results") as finalize,
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        tuning.main()

                    self.assertEqual(
                        [call.args[1]["key"] for call in train.call_args_list],
                        expected_keys,
                    )
                    write_trial.assert_called_once()
                    finalize.assert_called_once()
                    payload = finalize.call_args.args[1]
                    self.assertEqual(payload["sequence_keys"], expected_keys)
                    self.assertIsInstance(payload["num_sequences"], int)
                    self.assertEqual(payload["num_sequences"], len(expected_keys))
                    trial_result = payload["all_trials"][0]["result"]
                    self.assertEqual(list(trial_result["sequence_results"]), expected_keys)
                    self.assertEqual(
                        trial_result["num_sequences_evaluated"], len(expected_keys)
                    )


if __name__ == "__main__":
    unittest.main()
