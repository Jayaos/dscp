"""Causal data use, original initialization, tuning, and saved-output checks."""

from copy import deepcopy
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf

from baselines.aci.tuning import tune_hyperparameters
from baselines.aci.data import prepare_sequence, split_boundaries
from baselines.aci.run_aci import (
    evaluate_sequence, evaluate_sequences, run_aci, validate_config,
)


def config():
    return {"data": {"calibration_ratio": .66, "test_ratio": .34},
            "model": {"window_size": 7, "gamma": .005, "target_quantiles": [[.05, .95], [.1, .9]]},
            "plotting": {"plotting": False}}


def artifact(n=101):
    index = np.arange(n, dtype=float)
    predictions = 100. + index * .1
    return {"heldout_y": predictions + np.sin(index) * 5,
            "heldout_predictions": predictions[:, None]}


class ACIDataTests(unittest.TestCase):
    def test_split_matches_nexcp_floating_point_floor(self):
        for n, ratio in ((101, .66), (100, .29), (200, .7)):
            bounds = split_boundaries(n, ratio)
            self.assertEqual(bounds["test_start"], int(np.floor(n * ratio)))
            self.assertEqual(bounds["calibration_size"] + bounds["test_size"], n)
        self.assertEqual(split_boundaries(101, test_start=67)["test_start"], 67)

    def test_mixed_shapes_no_features_and_per_series_boundaries(self):
        settings = config()
        settings["data"]["test_start"] = {"a": 67}
        prepared = prepare_sequence(artifact(), settings, "a")
        np.testing.assert_array_equal(prepared["target_indices"], np.arange(67, 101))
        self.assertEqual(len(prepared["calibration_scores"]), 67)
        with self.assertRaises(ValueError):
            prepare_sequence(artifact(), settings, "missing")

    def test_current_and_future_targets_do_not_change_issued_intervals(self):
        item = artifact()
        original = evaluate_sequence("a", item, config())
        changed = deepcopy(item)
        changed["heldout_y"][75:] += 10000.
        altered = evaluate_sequence("a", changed, config())
        for pair in original["evaluation_results"]:
            reference = original["evaluation_results"][pair]
            actual = altered["evaluation_results"][pair]
            # Test starts at66: interval75 must still be identical before y75.
            for field in ("lower_interval", "upper_interval", "alpha_used"):
                np.testing.assert_array_equal(reference[field][:10], actual[field][:10])
            self.assertNotEqual(reference["alpha_used"][10], actual["alpha_used"][10])

    def test_validation_and_selection_never_consume_final_test_values(self):
        settings = config()
        settings["tuning"] = {"gamma_grid": [.001, .005, .05],
                              "window_size_grid": [5, 10], "validation_ratio": .2}
        item = artifact()
        reference = tune_hyperparameters({"a": item}, settings)
        changed = deepcopy(item)
        changed["heldout_y"][66:] = np.nan
        changed["heldout_predictions"][66:] = np.nan
        self.assertEqual(reference, tune_hyperparameters({"a": changed}, settings))
        del settings["tuning"]
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_sequence("a", changed, settings)
        self.assertTrue(all(index < 66 for index in reference["validation_target_indices"]["a"]))

    def test_calibration_initializes_pool_without_adapting_alpha(self):
        settings = config()
        output = evaluate_sequence("a", artifact(), settings)
        for pair, result in output["evaluation_results"].items():
            expected = .1 if pair == (.05, .95) else .2
            self.assertAlmostEqual(result["alpha_used"][0], expected)
            self.assertEqual(result["final_history_size"], 7)

    def test_independent_sequence_and_coverage_states(self):
        first, second = artifact(101), artifact(80)
        second["heldout_y"] += 50
        settings = config()
        combined = evaluate_sequences({"a": first, "b": second}, settings)
        for key, item in (("a", first), ("b", second)):
            alone = evaluate_sequence(key, item, settings)
            self.assertEqual(combined[key]["evaluation_results"], alone["evaluation_results"])
        single_level = deepcopy(settings)
        single_level["model"]["target_quantiles"] = [[.05, .95]]
        alone = evaluate_sequence("a", first, single_level)
        self.assertEqual(alone["evaluation_results"][(.05, .95)],
                         combined["a"]["evaluation_results"][(.05, .95)])

    def test_relative_score_restores_response_scale(self):
        settings = config()
        settings["model"].update({"score_type": "relative", "window_size": None})
        item = {"heldout_y": np.array([11., 18., 26., 44.]),
                "heldout_predictions": np.array([10., 20., 30., 40.])}
        settings["data"].update({"calibration_ratio": .75, "test_ratio": .25})
        output = evaluate_sequence("a", item, settings)["evaluation_results"][(.05, .95)]
        threshold = np.quantile([.1, .1, 4/30], .9, method="linear")
        self.assertAlmostEqual(output["lower_interval"][0], 40 - 40 * threshold)
        self.assertAlmostEqual(output["upper_interval"][0], 40 + 40 * threshold)
        item["heldout_predictions"][0] = 0
        with self.assertRaises(ValueError):
            evaluate_sequence("a", item, settings)

    def test_exceptional_predictions_remain_in_timeline_and_state_updates(self):
        settings = config()
        settings["model"]["initial_alpha"] = 1.
        output = evaluate_sequence("a", artifact(), settings)["evaluation_results"][(.05, .95)]
        self.assertEqual(output["num_test_points"], 35)
        self.assertEqual(output["interval_kind"][0], "empty")
        self.assertEqual(output["coverage"][0], False)
        self.assertAlmostEqual(output["alpha_next"][0], .9955)
        self.assertGreater(output["num_finite_intervals"], 0)
        self.assertEqual(len(output["finite_interval_mask"]), 35)

    def test_parallel_matches_serial(self):
        data = {"a": artifact(30), "b": artifact(40)}
        serial = evaluate_sequences(data, config(), num_cores=1)
        parallel = evaluate_sequences(data, config(), num_cores=2)
        for key in data:
            self.assertEqual(serial[key]["evaluation_results"], parallel[key]["evaluation_results"])

    def test_normal_run_persists_fixed_configuration_without_tuning_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "forecasts.pkl"
            with path.open("wb") as stream:
                pickle.dump({"a": artifact(101)}, stream)
            settings = config()
            settings["data"]["data_path"] = str(path)
            settings["saving_dir"] = str(directory / "output")
            output = run_aci(settings)
            folder = directory / "output"
            self.assertFalse((folder / "tuning_results.pkl").exists())
            chosen = OmegaConf.load(folder / "resolved_config.yaml")
            self.assertNotIn("tuning", chosen)
            self.assertNotIn("tuning", OmegaConf.load(folder / "requested_config.yaml"))
            self.assertEqual(chosen.model.gamma, settings["model"]["gamma"])
            self.assertEqual(chosen.model.window_size, settings["model"]["window_size"])
            self.assertEqual(output["a"]["metadata"]["model"]["gamma"], chosen.model.gamma)
            with (folder / "log.pkl").open("rb") as stream:
                self.assertEqual(pickle.load(stream), output)
            self.assertEqual(output["a"]["metadata"]["target_indices"], list(range(66, 101)))
            chosen.saving_dir = str(directory / "replay")
            replay = run_aci(chosen)
            self.assertEqual(output["a"]["evaluation_results"], replay["a"]["evaluation_results"])

    def test_invalid_settings_fail_before_evaluation(self):
        for section, key, value in (("data", "calibration_ratio", 1), ("data", "normalize", True),
                                    ("model", "prediction_step", 10), ("model", "clip_alpha", True),
                                    ("model", "target_quantiles", [[.1, .8]])):
            settings = config()
            settings.setdefault(section, {})[key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_config(settings)


if __name__ == "__main__":
    unittest.main()
