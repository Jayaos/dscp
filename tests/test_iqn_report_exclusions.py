"""Saved IQN-CP direct cosine-head reports retain explicit crossing exclusions."""

import contextlib
import copy
import io
import pickle
import tempfile
import unittest
from pathlib import Path

from report.inspect_run_results import METRICS, inspect_results, summarize_results


PAIR = (0.1, 0.9)


def sequence(coverage, mask, width=2.0):
    evaluated = sum(mask)
    excluded = len(mask) - evaluated
    status = ("no_valid_predictions" if not evaluated else
              "completed_with_exclusions" if excluded else "complete")
    return {
        "metadata": {
            "method": "iqn_cp", "prediction_head": "cosine_embedding",
            "interval_mode": "direct", "evaluation_status": status,
            "total_points": len(mask), "evaluated_points": evaluated,
            "excluded_points_count": excluded,
            "exclusion_rate": excluded / len(mask),
            "valid_prediction_mask": mask,
            "exclusion_policy": "skip_crossed_quantiles_timestamp_all_coverage_levels",
            "reporting_timeline": "successful_predictions_only",
        },
        "evaluation_results": {PAIR: {
            "coverage": coverage,
            "interval_width": [width] * evaluated,
            "winkler_score": [width * 2] * evaluated,
        }},
    }


class IQNReportExclusionTests(unittest.TestCase):
    def test_retained_points_preserve_sequence_weighting_and_rolling_windows(self):
        log = {
            "partial": sequence([True, True, False], [True, False, True, True], width=2),
            "complete": sequence([False], [True], width=8),
            "excluded": sequence([], [False, False]),
        }
        result = summarize_results(log, 2)[PAIR]

        self.assertEqual(result["num_sequences"], 2)
        self.assertEqual(result["num_total_sequences"], 3)
        self.assertEqual(result["num_no_valid_sequences"], 1)
        self.assertEqual(result["total_points"], 7)
        self.assertEqual(result["evaluated_points"], 4)
        self.assertEqual(result["excluded_points_count"], 3)
        self.assertAlmostEqual(result["exclusion_rate"], 3 / 7)
        # Each nonempty sequence has equal weight, regardless of retained length.
        self.assertAlmostEqual(result["avg_coverage_mean"], 1 / 3)
        self.assertAlmostEqual(result["avg_coverage_std"], 1 / 3)
        self.assertEqual(result["avg_interval_width_mean"], 5)
        self.assertEqual(result["num_rolling_sequences"], 1)
        # Retained observations form windows [True, True] and [True, False].
        self.assertEqual(result["avg_rolling_coverage_mean"], 0.75)
        self.assertAlmostEqual(result["avg_rolling_undercoverage_mean"], 0.15)

    def test_all_crossed_run_reports_unavailable_scores_and_method_neutral_counts(self):
        item = sequence([], [False, False])
        item["evaluation_results"][(0.025, 0.975)] = copy.deepcopy(item["evaluation_results"][PAIR])
        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory) / "log.pkl").open("wb") as handle:
                pickle.dump({"excluded": item}, handle)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                summary = inspect_results(directory, 2)

        for result in summary.values():
            self.assertEqual(result["num_sequences"], 0)
            self.assertEqual(result["num_rolling_sequences"], 0)
            self.assertEqual(result["num_no_valid_sequences"], 1)
            self.assertEqual(result["evaluated_points"], 0)
            for _, metric in METRICS:
                self.assertIsNone(result[f"avg_{metric}_mean"])
                self.assertIsNone(result[f"avg_{metric}_std"])
        text = output.getvalue()
        self.assertIn("Sequences with no valid predictions: 1/1", text)
        self.assertNotIn("DistMatch sequences", text)
        self.assertIn("evaluated=0, excluded=2, total=2", text)
        self.assertIn("rolling windows skip excluded points", text)
        self.assertIn("n/a", text)

    def test_only_direct_cosine_iqn_logs_accept_empty_metrics(self):
        for field, value in (
            ("method", None), ("method", "iqn"),
            ("prediction_head", None), ("prediction_head", "partially_monotonic"),
            ("interval_mode", None), ("interval_mode", "sampling"),
        ):
            item = sequence([], [False, False])
            item["metadata"][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "nonempty"):
                summarize_results({"bad": item}, 2)

    def test_other_head_or_interval_mode_does_not_enable_exclusion_counts(self):
        for field, value in (
            ("prediction_head", "partially_monotonic"), ("interval_mode", "sampling"),
        ):
            item = sequence([True, False], [True, False, True])
            item["metadata"][field] = value
            with self.subTest(field=field, value=value):
                result = summarize_results({"ordinary": item}, 2)[PAIR]
                self.assertEqual(result["avg_coverage_mean"], 0.5)
                self.assertNotIn("excluded_points_count", result)

    def test_exclusion_metadata_is_validated(self):
        for field, value in (
            ("total_points", 5),
            ("evaluated_points", True),
            ("excluded_points_count", -1),
            ("valid_prediction_mask", [True, True, True]),
            ("valid_prediction_mask", [1, 0, 1]),
            ("exclusion_rate", 0.5),
            ("evaluation_status", "no_valid_predictions"),
            ("exclusion_policy", None),
            ("exclusion_policy", "skip_crossed_pairs_only"),
        ):
            item = sequence([True, False], [True, False, True])
            item["metadata"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "IQN-CP"):
                summarize_results({"bad": item}, 2)

    def test_empty_status_cannot_hide_metric_data(self):
        item = sequence([], [False, False])
        for name in ("coverage", "interval_width", "winkler_score"):
            item["evaluation_results"][PAIR][name] = [1]
        with self.assertRaisesRegex(ValueError, "metric length disagrees"):
            summarize_results({"bad": item}, 2)

    def test_no_valid_predictions_requires_matching_status(self):
        item = sequence([], [False, False])
        item["metadata"]["evaluation_status"] = "completed_with_exclusions"
        with self.assertRaisesRegex(ValueError, "evaluation_status"):
            summarize_results({"bad": item}, 2)


if __name__ == "__main__":
    unittest.main()
