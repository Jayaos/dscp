"""Saved-result alignment must preserve target identity across old and new logs."""

import copy
import unittest

import numpy as np

from conditional_coverage.alignment import align_result


class SavedCoverageAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.y = np.arange(101, dtype=float) + 0.125
        self.ids = np.arange(67, 101)

    def log(self, indices=None, *, explicit=True, pair=(.05, .95)):
        indices = self.ids if indices is None else np.asarray(indices, dtype=int)
        result = {"coverage": (indices % 3 != 0).tolist(),
                  "target_y": self.y[indices].tolist()}
        if explicit:
            result["target_indices"] = indices.tolist()
        return {"evaluation_results": {pair: result}}

    @staticmethod
    def result(log):
        return next(iter(log["evaluation_results"].values()))

    def test_explicit_indices_preserve_saved_order_and_need_no_fallback(self):
        indices = np.array([100, 67, 89])
        actual = align_result(self.log(indices), .9, self.y,
                              expected_test_indices=self.ids)
        np.testing.assert_array_equal(actual["target_indices"], indices)
        np.testing.assert_array_equal(actual["coverage"], indices % 3 != 0)
        self.assertEqual(actual["alignment"], "result_indices")
        self.assertEqual(actual["full_test_count"], 34)
        self.assertEqual(actual["excluded_count"], 31)

    def test_chronological_legacy_order_requires_opt_in(self):
        log = self.log(explicit=False)
        with self.assertRaisesRegex(ValueError, "enable allow_chronological_fallback"):
            align_result(log, .9, self.y, expected_test_indices=self.ids)
        actual = align_result(log, .9, self.y, expected_test_indices=self.ids,
                              allow_chronological_fallback=True)
        np.testing.assert_array_equal(actual["target_indices"], self.ids)
        self.assertEqual(actual["alignment"], "chronological_config")

    def test_sparse_explicit_ids_do_not_invent_original_timeline_counts(self):
        for metadata in ({}, {"evaluation_status": "completed_with_exclusions"},
                         {"exclusion_rate": .5, "evaluated_points": 3}):
            with self.subTest(metadata=metadata):
                log = self.log([70, 82, 99])
                log["metadata"] = metadata
                actual = align_result(log, .9, self.y)
                np.testing.assert_array_equal(actual["target_indices"], [70, 82, 99])
                self.assertIsNone(actual["full_test_count"])
                self.assertIsNone(actual["excluded_count"])

    def test_reported_exclusion_count_can_recover_explicit_timeline_size(self):
        log = self.log([70, 82, 99])
        log["metadata"] = {"excluded_points_count": 31, "evaluated_points": 3}
        actual = align_result(log, .9, self.y)
        self.assertEqual(actual["full_test_count"], 34)
        self.assertEqual(actual["excluded_count"], 31)

    def test_suffix_fallback_retains_rounding_boundary_observation(self):
        # A Local-CP run may start one point before the shared groups at index 67.
        local_indices = np.arange(66, 101)
        actual = align_result(self.log(local_indices, explicit=False), .9, self.y,
                              allow_chronological_fallback=True)
        np.testing.assert_array_equal(actual["target_indices"], local_indices)
        self.assertEqual(actual["alignment"], "chronological_suffix")
        self.assertEqual(actual["excluded_count"], 0)

    def test_metadata_indices_are_usable_without_legacy_opt_in(self):
        log = self.log(explicit=False)
        log["metadata"] = {"target_indices": self.ids.tolist(), "split": "test"}
        actual = align_result(log, .9, self.y)
        np.testing.assert_array_equal(actual["target_indices"], self.ids)
        self.assertEqual(actual["alignment"], "metadata_indices")

    def test_validity_mask_preserves_excluded_holes_for_all_index_sources(self):
        mask = np.ones(34, dtype=bool)
        mask[[0, 8, 33]] = False
        retained = self.ids[mask]
        for explicit, metadata_ids, configured in ((True, True, True),
                                                  (False, True, True),
                                                  (False, False, True),
                                                  (False, False, False)):
            with self.subTest(explicit=explicit, metadata_ids=metadata_ids,
                              configured=configured):
                log = self.log(retained, explicit=explicit)
                log["metadata"] = {"valid_prediction_mask": mask.tolist(),
                                   "total_points": 34, "evaluated_points": 31,
                                   "excluded_points_count": 3}
                if metadata_ids:
                    log["metadata"]["target_indices"] = self.ids.tolist()
                actual = align_result(
                    log, .9, self.y,
                    expected_test_indices=self.ids if configured else None,
                    allow_chronological_fallback=True,
                )
                np.testing.assert_array_equal(actual["target_indices"], retained)
                self.assertEqual(actual["full_test_count"], 34)
                self.assertEqual(actual["excluded_count"], 3)

    def test_all_excluded_results_keep_empty_arrays_and_original_count(self):
        log = self.log([], explicit=False)
        log["metadata"] = {"target_indices": self.ids.tolist(),
                           "valid_prediction_mask": [False] * 34,
                           "total_points": 34, "evaluated_points": 0,
                           "excluded_points_count": 34}
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                if explicit:
                    self.result(log)["target_indices"] = []
                actual = align_result(log, .9, self.y)
                self.assertEqual(actual["target_indices"].dtype, np.int64)
                self.assertEqual(actual["coverage"].dtype, np.bool_)
                self.assertEqual(len(actual["target_indices"]), 0)
                self.assertEqual(actual["excluded_count"], 34)

    def test_known_exclusions_without_indices_or_mask_are_never_compressed(self):
        for metadata in ({"excluded_points_count": 1}, {"total_points": 34},
                         {"evaluation_status": "completed_with_exclusions"},
                         {"exclusion_rate": .1}, {"excluded_points": [{"target_index": 67}]}):
            with self.subTest(metadata=metadata):
                log = self.log(self.ids[1:], explicit=False)
                log["metadata"] = metadata
                with self.assertRaisesRegex(ValueError, "require retained indices or valid_prediction_mask"):
                    align_result(log, .9, self.y, allow_chronological_fallback=True)

    def test_configured_timeline_rejects_truncated_legacy_suffix(self):
        with self.assertRaisesRegex(ValueError, "Excluded or missing predictions"):
            align_result(self.log(self.ids[1:], explicit=False), .9, self.y,
                         expected_test_indices=self.ids, allow_chronological_fallback=True)

    def test_logged_targets_validate_explicit_and_inferred_alignment(self):
        for explicit in (True, False):
            log = self.log(explicit=explicit)
            self.result(log)["target_y"][3] += 10
            with self.subTest(explicit=explicit), self.assertRaisesRegex(
                ValueError, "Saved target_y disagrees",
            ):
                align_result(log, .9, self.y, allow_chronological_fallback=True)

    def test_malformed_explicit_indices_never_trigger_fallback(self):
        for values in ([67, 67], [-1, 100], [67, 101], [67., 100.],
                       [True, False], [[67, 100]], None):
            with self.subTest(values=values):
                log = self.log([67, 100])
                self.result(log)["target_indices"] = values
                with self.assertRaises(ValueError):
                    align_result(log, .9, self.y, allow_chronological_fallback=True)

    def test_metadata_and_mask_must_agree_with_result_and_config(self):
        log = self.log()
        log["metadata"] = {"target_indices": self.ids.tolist(),
                           "valid_prediction_mask": [True] * 34}
        cases = [
            ("mask_shape", lambda item: item["metadata"].update(valid_prediction_mask=[True] * 33)),
            ("mask_type", lambda item: item["metadata"].update(valid_prediction_mask=[1] * 34)),
            ("mask_conflict", lambda item: item["metadata"]["valid_prediction_mask"].__setitem__(0, False)),
            ("indices_conflict", lambda item: item["metadata"]["target_indices"].__setitem__(0, 66)),
            ("count_conflict", lambda item: item["metadata"].update(evaluated_points=33)),
            ("boundary_conflict", lambda item: item["metadata"].update(test_start=66)),
        ]
        for name, modify in cases:
            item = copy.deepcopy(log)
            modify(item)
            with self.subTest(name=name), self.assertRaises(ValueError):
                align_result(item, .9, self.y, expected_test_indices=self.ids)

    def test_coverage_must_be_finite_binary_and_have_matching_targets(self):
        for values in ([1, .8], [1, np.nan], [1, np.inf], [[True, False]], [True]):
            log = self.log([67, 68])
            self.result(log)["coverage"] = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                align_result(log, .9, self.y)

    def test_interval_selection_requires_correct_nominal_coverage(self):
        with self.assertRaisesRegex(ValueError, "No saved interval matches"):
            align_result(self.log(pair=(.1, .9)), .9, self.y)
        actual = align_result(self.log(pair=(.95, .05)), .9, self.y)
        self.assertEqual(actual["quantile_pair"], [.05, .95])

    def test_ambiguous_coverage_requires_pair_selection(self):
        log = self.log()
        log["evaluation_results"][(.025, .925)] = copy.deepcopy(self.result(log))
        with self.assertRaisesRegex(ValueError, "Multiple saved intervals"):
            align_result(log, .9, self.y)
        actual = align_result(log, .9, self.y, quantile_pair=[.925, .025])
        self.assertEqual(actual["quantile_pair"], [.025, .925])
        with self.assertRaisesRegex(ValueError, "does not match nominal_coverage"):
            align_result(log, .9, self.y, quantile_pair=[.1, .9])

    def test_target_columns_and_numeric_binary_coverage_are_supported(self):
        log = self.log([67, 68])
        result = self.result(log)
        result["coverage"] = [1., 0.]
        result["target_y"] = [[self.y[67]], [self.y[68]]]
        actual = align_result(log, .9, self.y[:, None])
        np.testing.assert_array_equal(actual["coverage"], [True, False])

    def test_validation_logs_and_inconsistent_counts_fail_with_context(self):
        log = self.log()
        log["metadata"] = {"split": "validation"}
        with self.assertRaisesRegex(ValueError, "qr/station: Only final test"):
            align_result(log, .9, self.y, context="qr/station")
        log["metadata"] = {"evaluated_points": 34}
        self.result(log)["evaluated_points"] = 33
        with self.assertRaisesRegex(ValueError, "Conflicting saved evaluated_points"):
            align_result(log, .9, self.y)


if __name__ == "__main__":
    unittest.main()
