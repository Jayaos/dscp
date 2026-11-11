import copy
import unittest

import numpy as np

from conditional_coverage.data import observed_sequence, prepare_contexts, shared_indices


def comparison(method, **data):
    return {"name": method, "method": method, "config": {"data": data}}


def source(length=101):
    t = np.arange(length, dtype=float)
    return {"heldout_x": np.column_stack((t / 10, np.ones(length))),
            "heldout_y": 3 + np.sin(t / 4)}


def contexts(item=None, test_start=67):
    item = source() if item is None else item
    x, y = observed_sequence(item)
    return prepare_contexts(x, y, np.arange(66), np.arange(test_start, len(y)), 5, .2)


class SharedGroupSplitTests(unittest.TestCase):
    def test_all_current_methods_use_their_exact_rounding_rules(self):
        specs = [comparison(method, train_ratio=.5, valid_ratio=.16)
                 for method in ("qr_cp", "iqn_cp", "distmatch")]
        specs += [comparison("hopcpt", train_ratio=.33, valid_ratio=.33),
                  comparison("local_cp", train_ratio=.4, valid_ratio=.1,
                             calibration_ratio=.16, test_ratio=.34),
                  comparison("spci", train_ratio=.66)]
        specs += [comparison(method, calibration_ratio=.66, test_ratio=.34)
                  for method in ("nexcp", "aci", "cop", "lqt", "split_cp", "rescp", "kowcpi")]
        result = shared_indices(101, specs, "station")
        np.testing.assert_array_equal(result["calibration_indices"], np.arange(66))
        np.testing.assert_array_equal(result["test_indices"], np.arange(67, 101))
        np.testing.assert_array_equal(result["excluded_boundary_indices"], [66])
        for method, boundary in result["member_boundaries"].items():
            expected = 67 if method in {"qr_cp", "iqn_cp", "distmatch", "hopcpt"} else 66
            self.assertEqual(boundary["test_start"], expected, method)

    def test_float_rounding_matches_method_implementations(self):
        plain = comparison("nexcp", calibration_ratio=.29)
        decimal = comparison("split_cp", calibration_ratio=.29)
        snapped = comparison("kowcpi", calibration_ratio=.29)
        result = shared_indices(100, [plain, decimal, snapped], "s")
        self.assertEqual(result["member_boundaries"]["nexcp"]["test_start"], 28)
        self.assertEqual(result["member_boundaries"]["split_cp"]["test_start"], 29)
        self.assertEqual(result["member_boundaries"]["kowcpi"]["test_start"], 29)

    def test_exact_per_sequence_override_and_identical_boundaries(self):
        first = comparison("split_cp", calibration_ratio=.66, test_start={"s": 70})
        second = comparison("aci", calibration_ratio=.66, test_start=70)
        result = shared_indices(101, [first, second], "s")
        self.assertEqual(len(result["calibration_indices"]), 70)
        self.assertEqual(len(result["test_indices"]), 31)
        self.assertEqual(len(result["excluded_boundary_indices"]), 0)
        with self.assertRaisesRegex(ValueError, "missing sequence"):
            shared_indices(101, [first], "other")

    def test_prediction_exclusions_do_not_shrink_shared_test(self):
        spec = comparison("qr_cp", train_ratio=.5, valid_ratio=.16)
        complete = np.arange(67, 101)
        valid = np.ones(len(complete), dtype=bool)
        valid[[0, 8, 33]] = False
        spec["log"] = {"s": {
            "metadata": {"split": "test", "target_indices": complete.tolist(),
                         "valid_prediction_mask": valid.tolist(), "total_points": len(complete)},
            "evaluation_results": {(.05, .95): {
                "target_indices": complete[valid].tolist(), "coverage": [True] * int(valid.sum())}},
        }}
        result = shared_indices(101, [spec], "s")
        np.testing.assert_array_equal(result["test_indices"], complete)

    def test_all_failed_predictions_still_preserve_the_test_partition(self):
        spec = comparison("qr_cp", train_ratio=.5, valid_ratio=.16)
        spec["log"] = {"s": {
            "metadata": {"target_indices": list(range(67, 101)),
                         "valid_prediction_mask": [False] * 34},
            "evaluation_results": {(.05, .95): {"target_indices": [], "coverage": []}},
        }}
        np.testing.assert_array_equal(shared_indices(101, [spec], "s")["test_indices"], np.arange(67, 101))

    def test_ambiguous_or_incompatible_logs_are_rejected(self):
        base = comparison("split_cp", calibration_ratio=.66)
        bad_records = [
            {"metadata": {"boundaries": {"test_start": 67}}},
            {"metadata": {"target_indices": list(range(67, 101))}},
            {"metadata": {"split": "validation"}},
            {"evaluation_results": {(.05, .95): {"coverage": [1] * 34}}},
            {"metadata": {"valid_prediction_mask": [1] * 35}},
            {"evaluation_results": {(.05, .95): {"target_indices": list(range(66, 100))}}},
        ]
        for record in bad_records:
            with self.subTest(record=record), self.assertRaises(ValueError):
                shared_indices(101, [dict(base, log={"s": record})], "s")
        with self.assertRaisesRegex(ValueError, "no log"):
            shared_indices(101, [dict(base, log={})], "s")

    def test_invalid_splits_do_not_fall_back_to_a_default(self):
        bad_specs = [comparison("unknown", calibration_ratio=.66),
                     comparison("qr_cp", train_ratio=.8, valid_ratio=.3),
                     comparison("spci", train_ratio=.66, valid_ratio=.1),
                     comparison("nexcp", calibration_ratio=.66, test_start=66),
                     comparison("split_cp", calibration_ratio=.66, test_ratio=.2),
                     comparison("local_cp", train_ratio=.4, valid_ratio=.1,
                                calibration_ratio=.16, test_ratio=.2)]
        for spec in bad_specs:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                shared_indices(101, [spec], "s")
        with self.assertRaisesRegex(ValueError, "unique"):
            shared_indices(101, [comparison("spci", train_ratio=.66)] * 2, "s")


class ObservedContextTests(unittest.TestCase):
    def test_original_targets_need_no_forecasts_and_preserve_shapes(self):
        item = source()
        x, y = observed_sequence(item)
        self.assertEqual(x.shape, (101, 2))
        self.assertEqual(y.shape, (101,))
        item["heldout_y"] = item["heldout_y"][:, None]
        item["heldout_predictions"] = np.full(101, np.nan)
        xx, yy = observed_sequence(item)
        np.testing.assert_array_equal(xx, x)
        np.testing.assert_array_equal(yy, y)
        one_x, _ = observed_sequence({"heldout_x": x[:, 0], "heldout_y": y})
        self.assertEqual(one_x.shape, (101, 1))

    def test_scalar_target_and_numeric_covariate_contract(self):
        for x, y in ((np.zeros((10, 2)), np.zeros((10, 2))),
                     (np.zeros(9), np.zeros(10)),
                     (np.zeros((10, 0)), np.zeros(10)),
                     (np.zeros(10, dtype=complex), np.zeros(10))):
            with self.subTest(x=x.shape, y=y.shape), self.assertRaises(ValueError):
                observed_sequence({"heldout_x": x, "heldout_y": y})

    def test_training_validation_indices_and_scaler_use_only_fit_prefix(self):
        item = source()
        prepared = contexts(item)
        np.testing.assert_array_equal(prepared["fit_indices"], np.arange(5, 52))
        np.testing.assert_array_equal(prepared["validation_indices"], np.arange(52, 66))
        np.testing.assert_array_equal(prepared["calibration_target_indices"], np.arange(5, 66))
        expected = np.column_stack((item["heldout_x"], item["heldout_y"]))[:52]
        np.testing.assert_allclose(prepared["scaler"]["feature_mean"], expected.mean(axis=0))
        self.assertEqual(prepared["scaler"]["feature_std"][1], 1.0)
        self.assertTrue(np.isnan(prepared["y_normalized"][66:]).all())

    def test_validation_and_test_targets_never_fit_normalization(self):
        original = source()
        changed = copy.deepcopy(original)
        changed["heldout_y"][52:] += 10000
        changed["heldout_x"][52:] += 500
        first, second = contexts(original), contexts(changed)
        for field in ("feature_mean", "feature_std"):
            np.testing.assert_array_equal(first["scaler"][field], second["scaler"][field])
        np.testing.assert_array_equal(first["features"][:52], second["features"][:52])
        np.testing.assert_array_equal(first["y_normalized"][:52], second["y_normalized"][:52])

    def test_current_target_does_not_enter_its_own_context_but_can_enter_next(self):
        original = source()
        changed = copy.deepcopy(original)
        changed["heldout_y"][75] += 500
        first, second = contexts(original), contexts(changed)
        np.testing.assert_array_equal(first["features"][70:75], second["features"][70:75])
        self.assertFalse(np.array_equal(first["features"][71:76], second["features"][71:76]))
        np.testing.assert_array_equal(first["features"][:66], second["features"][:66])

    def test_final_test_target_and_covariates_are_unnecessary_for_representation(self):
        item = source()
        item["heldout_y"][-1] = np.nan
        item["heldout_x"][-1] = np.nan
        prepared = contexts(item)
        self.assertTrue(np.isfinite(prepared["features"][-6:-1]).all())
        self.assertEqual(prepared["test_indices"][-1], 100)

    def test_boundary_gap_is_history_only(self):
        item = source()
        prepared = contexts(item)
        self.assertNotIn(66, prepared["calibration_target_indices"])
        self.assertNotIn(66, prepared["test_indices"])
        expected_y = (item["heldout_y"][66] - prepared["scaler"]["y_mean"]) / prepared["scaler"]["y_std"]
        self.assertAlmostEqual(float(prepared["features"][66, -1]), expected_y, places=5)

    def test_nonfinite_observed_history_short_calibration_and_overlap_fail(self):
        item = source()
        item["heldout_y"][80] = np.nan
        with self.assertRaisesRegex(ValueError, "contexts"):
            contexts(item)
        x, y = observed_sequence(source())
        for calibration, test, window, ratio in ((np.arange(6), np.arange(67, 101), 5, .2),
                                                  (np.arange(66), np.arange(65, 101), 5, .2),
                                                  (np.arange(1, 66), np.arange(67, 101), 5, .2),
                                                  (np.arange(66), np.arange(67, 101), 5, 0)):
            with self.subTest(calibration=calibration, window=window), self.assertRaises(ValueError):
                prepare_contexts(x, y, calibration, test, window, ratio)


if __name__ == "__main__":
    unittest.main()
