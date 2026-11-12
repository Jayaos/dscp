"""Numerical checks for group coverage and sequence-level aggregation."""

import unittest

import numpy as np

from conditional_coverage.metrics import aggregate_sequences, evaluate_sequence


def record(indices, labels, k):
    return {"test_indices": np.asarray(indices, dtype=np.int64),
            "groups": {k: {"test_labels": np.asarray(labels, dtype=np.int64)}}}


def predictions(indices, coverage):
    return {"target_indices": np.asarray(indices, dtype=np.int64),
            "coverage": np.asarray(coverage, dtype=bool)}


class ConditionalCoverageMetricTests(unittest.TestCase):
    def test_unequal_groups_have_equal_weight_in_cace(self):
        result = evaluate_sequence(
            record(range(10, 16), [0, 0, 1, 1, 1, 1], 2),
            predictions(range(10, 16), [True, False, True, True, True, False]), 2, .9,
        )
        first, second = result["groups"]
        self.assertEqual((first["group_size"], second["group_size"]), (2, 4))
        self.assertEqual((first["coverage"], second["coverage"]), (.5, .75))
        self.assertAlmostEqual(result["sequence"]["cace"], .275)
        self.assertAlmostEqual(result["sequence"]["wuc"], .4)
        self.assertAlmostEqual(result["sequence"]["coverage"], 4 / 6)
        self.assertEqual(result["sequence"]["status"], "complete")

    def test_overcoverage_has_zero_cace_and_retains_absolute_error_diagnostic(self):
        result = evaluate_sequence(
            record([7, 8], [0, 1], 2), predictions([7, 8], [True, True]), 2, .9,
        )
        self.assertEqual(result["sequence"]["cace"], 0)
        self.assertEqual(result["sequence"]["wuc"], 0)
        for group in result["groups"]:
            self.assertAlmostEqual(group["absolute_coverage_error"], .1)
            self.assertEqual(group["undercoverage"], 0)

    def test_mean_undercoverage_includes_zero_errors_from_at_or_above_nominal_groups(self):
        covered_counts = [85, 88, 90, 92, 95]
        coverage = [covered for count in covered_counts
                    for covered in ([True] * count + [False] * (100 - count))]
        result = evaluate_sequence(
            record(range(500), np.repeat(np.arange(5), 100), 5),
            predictions(range(500), coverage), 5, .9,
        )
        self.assertEqual(result["sequence"]["evaluated_group_count"], 5)
        self.assertAlmostEqual(result["sequence"]["cace"], (.05 + .02 + 0 + 0 + 0) / 5)
        self.assertAlmostEqual(result["sequence"]["wuc"], .05)
        np.testing.assert_allclose(
            [group["undercoverage"] for group in result["groups"]], [.05, .02, 0, 0, 0],
        )
        np.testing.assert_allclose(
            [group["absolute_coverage_error"] for group in result["groups"]], [.05, .02, 0, .02, .05],
        )

    def test_indices_join_noncontiguous_and_permuted_rows_without_compressing_holes(self):
        result = evaluate_sequence(
            record([10, 12, 17, 22], [0, 1, 0, 1], 2),
            predictions([22, 99, 10, 12], [False, True, True, True]), 2, .9,
        )
        first, second = result["groups"]
        self.assertEqual((first["coverage"], second["coverage"]), (1, .5))
        self.assertEqual((first["missing_count"], second["missing_count"]), (1, 0))
        self.assertEqual(result["sequence"]["evaluated_count"], 3)
        self.assertEqual(result["sequence"]["outside_shared_count"], 1)
        self.assertAlmostEqual(result["sequence"]["cace"], .2)
        self.assertEqual(result["sequence"]["status"], "partial")

    def test_rounding_boundary_row_is_cropped_by_original_id(self):
        result = evaluate_sequence(
            record(range(67, 71), [0, 0, 1, 1], 2),
            predictions(range(66, 71), [False, True, True, True, True]), 2, .9,
        )
        self.assertEqual(result["sequence"]["outside_shared_count"], 1)
        self.assertEqual(result["sequence"]["coverage"], 1)
        self.assertEqual(result["sequence"]["missing_count"], 0)
        self.assertEqual(result["sequence"]["status"], "complete")

    def test_empty_and_fully_excluded_groups_remain_visible(self):
        result = evaluate_sequence(
            record([10, 11, 12], [0, 0, 2], 3), predictions([10], [False]), 3, .9,
        )
        self.assertEqual([group["group_size"] for group in result["groups"]], [2, 0, 1])
        self.assertEqual([group["missing_count"] for group in result["groups"]], [1, 0, 1])
        for group in result["groups"][1:]:
            self.assertIsNone(group["coverage"])
            self.assertIsNone(group["absolute_coverage_error"])
            self.assertIsNone(group["undercoverage"])
        self.assertEqual(result["sequence"]["evaluated_group_count"], 1)
        self.assertEqual(result["sequence"]["empty_group_count"], 2)
        self.assertAlmostEqual(result["sequence"]["cace"], .9)
        self.assertAlmostEqual(result["sequence"]["wuc"], .9)
        self.assertEqual(result["sequence"]["status"], "partial")

    def test_all_excluded_predictions_produce_unavailable_metrics(self):
        result = evaluate_sequence(record([10, 11], [0, 1], 2), predictions([], []), 2, .9)
        sequence = result["sequence"]
        self.assertEqual(sequence["missing_count"], 2)
        self.assertEqual(sequence["evaluated_group_count"], 0)
        self.assertEqual(sequence["status"], "no_valid_predictions")
        for field in ("coverage", "cace", "wuc"):
            self.assertIsNone(sequence[field])
        summary = aggregate_sequences([sequence])
        self.assertEqual(summary["unavailable_sequence_count"], 1)
        self.assertIsNone(summary["cace"])
        self.assertIsNone(summary["wuc"])
        self.assertIsNone(summary["pooled_coverage"])

    def test_sequence_errors_have_equal_weight_and_ignore_cluster_label_names(self):
        first = evaluate_sequence(
            record(range(10), [0] * 5 + [1] * 5, 2),
            predictions(range(10), [True] * 10), 2, .9,
        )["sequence"]
        second_record = record([20, 21], [0, 1], 2)
        second_predictions = predictions([20, 21], [False, True])
        second = evaluate_sequence(second_record, second_predictions, 2, .9)["sequence"]
        summary = aggregate_sequences([first, second])
        self.assertAlmostEqual(summary["cace"], (0 + .45) / 2)
        self.assertAlmostEqual(summary["wuc"], .9 / 2)
        self.assertAlmostEqual(summary["worst_sequence_wuc"], .9)
        self.assertAlmostEqual(summary["mean_sequence_coverage"], .75)
        self.assertAlmostEqual(summary["pooled_coverage"], 11 / 12)
        self.assertEqual(summary["sequence_count"], 2)
        swapped = evaluate_sequence(record([20, 21], [1, 0], 2), second_predictions, 2, .9)["sequence"]
        self.assertEqual(summary, aggregate_sequences([first, swapped]))

    def test_unavailable_sequences_are_counted_without_poisoning_averages(self):
        group = record([10, 11], [0, 1], 2)
        available = evaluate_sequence(group, predictions([10, 11], [True, True]), 2, .9)["sequence"]
        missing = evaluate_sequence(group, predictions([], []), 2, .9)["sequence"]
        summary = aggregate_sequences([available, missing])
        self.assertEqual(summary["evaluated_sequence_count"], 1)
        self.assertEqual(summary["unavailable_sequence_count"], 1)
        self.assertEqual(summary["cace"], 0)
        self.assertEqual(summary["group_size"], 4)
        self.assertEqual(summary["missing_count"], 2)
        self.assertEqual(summary["pooled_coverage"], 1)

    def test_duplicate_ids_or_invalid_coverage_cannot_silently_distort_counts(self):
        group = record([10, 11], [0, 1], 2)
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_sequence(group, predictions([10, 10], [True, False]), 2, .9)
        for coverage in ([True], [.1, 1], [np.nan, 1], [2, 0]):
            with self.subTest(coverage=coverage), self.assertRaisesRegex(ValueError, "Coverage"):
                evaluate_sequence(group, {"target_indices": [10, 11], "coverage": coverage}, 2, .9)


if __name__ == "__main__":
    unittest.main()
