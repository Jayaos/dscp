"""Report compatibility for explicit DistMatch prediction exclusions."""

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
            "method": "distmatch", "evaluation_status": status,
            "total_points": len(mask), "evaluated_points": evaluated,
            "excluded_points_count": excluded,
            "exclusion_rate": excluded / len(mask),
            "valid_prediction_mask": mask,
        },
        "evaluation_results": {PAIR: {
            "coverage": coverage,
            "interval_width": [width] * evaluated,
            "winkler_score": [width * 2] * evaluated,
        }},
    }


class DistMatchReportExclusionTests(unittest.TestCase):
    def test_partial_complete_and_empty_sequences_preserve_equal_sequence_weighting(self):
        log = {
            "partial": sequence([True, True, False], [True, False, True, True], width=2),
            "complete": sequence([False], [True], width=8),
            "excluded": sequence([], [False, False]),
        }

        summary = summarize_results(log, 2)[PAIR]

        self.assertEqual(summary["num_sequences"], 2)
        self.assertEqual(summary["num_total_sequences"], 3)
        self.assertEqual(summary["num_no_valid_sequences"], 1)
        self.assertEqual(summary["total_points"], 7)
        self.assertEqual(summary["evaluated_points"], 4)
        self.assertEqual(summary["excluded_points_count"], 3)
        self.assertAlmostEqual(summary["exclusion_rate"], 3 / 7)
        # Equal sequence weights give 1/3, rather than the pooled coverage 1/2.
        self.assertAlmostEqual(summary["avg_coverage_mean"], 1 / 3)
        self.assertAlmostEqual(summary["avg_coverage_std"], 1 / 3)
        self.assertEqual(summary["avg_interval_width_mean"], 5)
        self.assertEqual(summary["num_rolling_sequences"], 1)
        # Complete windows over successful points are [1,1] and [1,0].
        self.assertEqual(summary["avg_rolling_coverage_mean"], 0.75)
        self.assertAlmostEqual(summary["avg_rolling_undercoverage_mean"], 0.15)

    def test_all_excluded_sequences_have_unavailable_scores_for_every_pair(self):
        log = {"first": sequence([], [False, False]), "second": sequence([], [False])}
        second_pair = (0.025, 0.975)
        for item in log.values():
            item["evaluation_results"][second_pair] = copy.deepcopy(item["evaluation_results"][PAIR])

        summary = summarize_results(log, 2)

        for pair in (PAIR, second_pair):
            result = summary[pair]
            self.assertEqual(result["num_sequences"], 0)
            self.assertEqual(result["num_rolling_sequences"], 0)
            self.assertEqual(result["num_no_valid_sequences"], 2)
            self.assertEqual(result["total_points"], 3)
            self.assertEqual(result["evaluated_points"], 0)
            self.assertEqual(result["exclusion_rate"], 1)
            for _, metric in METRICS:
                self.assertIsNone(result[f"avg_{metric}_mean"])
                self.assertIsNone(result[f"avg_{metric}_std"])

    def test_inspection_prints_exclusions_and_handles_an_all_invalid_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory) / "log.pkl").open("wb") as handle:
                pickle.dump({"excluded": sequence([], [False, False])}, handle)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                summary = inspect_results(directory, 2)

        self.assertIsNone(summary[PAIR]["avg_coverage_mean"])
        text = output.getvalue()
        self.assertIn("no valid predictions: 1/1", text)
        self.assertIn("evaluated=0, excluded=2, total=2", text)
        self.assertIn("rolling windows skip excluded points", text)
        self.assertIn("n/a", text)

    def test_unmarked_and_other_methods_still_reject_empty_metric_arrays(self):
        for method in (None, "split_cp", "rescp"):
            item = sequence([], [False, False])
            if method is None:
                item["metadata"].pop("method")
            else:
                item["metadata"]["method"] = method
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "nonempty"):
                summarize_results({"bad": item}, 2)

    def test_no_valid_marker_cannot_hide_nonempty_metrics(self):
        item = sequence([], [False, False])
        for name in ("coverage", "interval_width", "winkler_score"):
            item["evaluation_results"][PAIR][name] = [1]
        with self.assertRaisesRegex(ValueError, "metric length disagrees"):
            summarize_results({"bad": item}, 2)

    def test_metadata_must_match_saved_counts_mask_and_status(self):
        for field, value in (
            ("total_points", 5),
            ("evaluated_points", True),
            ("excluded_points_count", -1),
            ("valid_prediction_mask", [True, True, True]),
            ("valid_prediction_mask", [1, 0, 1]),
            ("exclusion_rate", 0.5),
            ("evaluation_status", "no_valid_predictions"),
        ):
            item = sequence([True, False], [True, False, True])
            item["metadata"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                summarize_results({"bad": item}, 2)

    def test_empty_distmatch_arrays_require_the_no_valid_status(self):
        item = sequence([], [False, False])
        item["metadata"]["evaluation_status"] = "completed_with_exclusions"
        with self.assertRaisesRegex(ValueError, "evaluation_status"):
            summarize_results({"bad": item}, 2)


if __name__ == "__main__":
    unittest.main()
