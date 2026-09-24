"""Report compatibility for independent-head QR-CP quantile exclusions."""

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
            "method": "qr_cp", "head_type": "independent",
            "evaluation_status": status,
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


class QRReportExclusionTests(unittest.TestCase):
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
        self.assertAlmostEqual(result["avg_coverage_mean"], 1 / 3)
        self.assertEqual(result["avg_interval_width_mean"], 5)
        self.assertEqual(result["num_rolling_sequences"], 1)
        self.assertEqual(result["avg_rolling_coverage_mean"], 0.75)

    def test_all_crossed_run_reports_unavailable_scores_for_every_pair(self):
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
            self.assertEqual(result["num_no_valid_sequences"], 1)
            self.assertEqual(result["evaluated_points"], 0)
            for _, metric in METRICS:
                self.assertIsNone(result[f"avg_{metric}_mean"])
                self.assertIsNone(result[f"avg_{metric}_std"])
        self.assertIn("evaluated=0, excluded=2, total=2", output.getvalue())
        self.assertIn("rolling windows skip excluded points", output.getvalue())
        self.assertIn("n/a", output.getvalue())

    def test_only_independent_qr_logs_accept_empty_metrics(self):
        for head_type in (None, "nondecreasing"):
            item = sequence([], [False, False])
            item["metadata"]["head_type"] = head_type
            with self.subTest(head_type=head_type), self.assertRaisesRegex(ValueError, "nonempty"):
                summarize_results({"bad": item}, 2)

    def test_exclusion_metadata_is_validated(self):
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
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "QR-CP"):
                summarize_results({"bad": item}, 2)

    def test_empty_status_cannot_hide_metric_data(self):
        item = sequence([], [False, False])
        for name in ("coverage", "interval_width", "winkler_score"):
            item["evaluation_results"][PAIR][name] = [1]
        with self.assertRaisesRegex(ValueError, "metric length disagrees"):
            summarize_results({"bad": item}, 2)


if __name__ == "__main__":
    unittest.main()
