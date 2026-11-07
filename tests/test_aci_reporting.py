import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

from baselines.aci.reporting import evaluate_intervals, plot_aci_results, summarize_results


PAIR = (0.05, 0.95)


def wrap(result):
    return {"evaluation_results": {PAIR: result}}


class ACIReportingTests(unittest.TestCase):
    def test_full_empty_and_finite_predictions_retain_every_timestamp(self):
        result = evaluate_intervals(
            targets=[0, 2, -7, 8], predictions=[0, 0, -5, 10],
            lower=[-1, -1, -math.inf, math.inf],
            upper=[1, 1, math.inf, -math.inf], alpha=0.1,
            target_indices=[10, 11, 12, 13],
        )
        self.assertEqual(result["coverage"], [True, False, True, False])
        self.assertEqual(result["target_indices"], [10, 11, 12, 13])
        self.assertEqual(result["interval_kind"], ["regular", "regular", "full", "empty"])
        self.assertEqual(result["interval_width"], [2, 2, math.inf, 0])
        self.assertEqual(result["winkler_score"], [2, 22, math.inf, math.inf])
        self.assertEqual(result["finite_interval_mask"], [True, True, False, False])
        self.assertEqual(result["finite_interval_width"], [2, 2])
        self.assertEqual(result["finite_winkler_score"], [2, 22])
        self.assertEqual(result["avg_coverage"], 0.5)
        self.assertAlmostEqual(result["avg_delta_coverage"], -0.4)
        self.assertEqual(result["avg_interval_width"], 2)
        self.assertEqual(result["avg_winkler_score"], 12)
        self.assertEqual(result["full_test_avg_interval_width"], math.inf)
        self.assertEqual(result["full_test_avg_winkler_score"], math.inf)
        self.assertEqual(result["num_test_points"], 4)
        self.assertEqual(result["num_finite_intervals"], 2)
        self.assertEqual(result["num_full_sets"], 1)
        self.assertEqual(result["num_empty_sets"], 1)
        self.assertEqual(result["full_set_rate"], 0.25)
        self.assertEqual(result["empty_set_rate"], 0.25)

    def test_no_ordinary_intervals_has_explicitly_undefined_efficiency(self):
        result = evaluate_intervals([2, 3], [2, 3], [-math.inf, math.inf],
                                    [math.inf, -math.inf], 0.1, [100, 101])
        self.assertIsNone(result["avg_interval_width"])
        self.assertIsNone(result["avg_winkler_score"])
        self.assertEqual(result["finite_interval_width"], [])
        self.assertEqual(result["avg_coverage"], 0.5)
        summary = summarize_results({"a": wrap(result)}, [PAIR])[PAIR]
        self.assertIsNone(summary["avg_interval_width_mean"])
        self.assertIsNone(summary["avg_interval_width_std"])
        self.assertEqual(summary["num_undefined_efficiency_sequences"], 1)
        self.assertEqual(summary["full_test_avg_interval_width_mean"], math.inf)
        self.assertIsNone(summary["full_test_avg_interval_width_std"])

    def test_only_empty_sets_have_zero_set_width_but_no_finite_interval_summary(self):
        result = evaluate_intervals([0], [0], [math.inf], [-math.inf], 0.1, [50])
        self.assertEqual(result["coverage"], [False])
        self.assertEqual(result["full_test_avg_interval_width"], 0)
        self.assertEqual(result["full_test_avg_winkler_score"], math.inf)
        self.assertIsNone(result["avg_interval_width"])

    def test_endpoints_are_inclusive_and_singletons_are_nonempty(self):
        result = evaluate_intervals([[0], [1], [-1]], [[0], [0], [0]],
                                    [[0], [-1], [-1]], [[0], [1], [1]], 0.1, [4, 5, 6])
        self.assertEqual(result["coverage"], [True, True, True])
        self.assertEqual(result["num_finite_intervals"], 3)
        self.assertEqual(result["finite_interval_width"], [0, 2, 2])

    def test_nominal_alpha_controls_penalty(self):
        result = evaluate_intervals([-2, 2], [0, 0], [-1, -1], [1, 1], 0.2, [1, 2])
        self.assertEqual(result["winkler_score"], [12, 12])
        self.assertEqual(result["metrics_scope"]["winkler_alpha"], "fixed_nominal_miscoverage")

    def test_response_scale_offsets_are_recorded_separately(self):
        result = evaluate_intervals([101], [100], [98], [103], 0.1, [9])
        self.assertEqual(result["lower_interval"], [98])
        self.assertEqual(result["upper_interval"], [103])
        self.assertEqual(result["lower_residual_quantile"], [-2])
        self.assertEqual(result["upper_residual_quantile"], [3])

    def test_summary_keeps_equal_series_weighting_and_pooled_exception_counts_distinct(self):
        ordinary = evaluate_intervals([0, 0, 0], [0, 0, 0], [-1, -1, -1],
                                      [1, 1, 1], 0.1, [1, 2, 3])
        exceptional = evaluate_intervals([0], [0], [math.inf], [-math.inf], 0.1, [1])
        row = summarize_results({"a": wrap(ordinary), "b": wrap(exceptional)}, [PAIR])[PAIR]
        self.assertEqual(row["avg_coverage_mean"], 0.5)
        self.assertEqual(row["avg_coverage_std"], 0.5)
        self.assertEqual(row["avg_interval_width_mean"], 2)
        self.assertEqual(row["num_efficiency_sequences"], 1)
        self.assertEqual(row["num_undefined_efficiency_sequences"], 1)
        self.assertEqual(row["num_test_points"], 4)
        self.assertEqual(row["empty_set_rate"], 0.25)
        self.assertEqual(row["empty_set_rate_mean"], 0.5)
        self.assertEqual(row["full_test_avg_interval_width_mean"], 1)
        self.assertEqual(row["full_test_avg_winkler_score_mean"], math.inf)

    def test_invalid_endpoint_encodings_are_rejected(self):
        for lower, upper in ((2, 1), (math.nan, 1), (-math.inf, 1),
                             (0, math.inf), (math.inf, math.inf), (-math.inf, -math.inf)):
            with self.subTest(lower=lower, upper=upper), self.assertRaises(ValueError):
                evaluate_intervals([0], [0], [lower], [upper], 0.1, [0])

    def test_invalid_targets_indices_and_alpha_are_rejected(self):
        for indices in ([1, 1], [2, 1], [-1, 0], [0.0, 1.0], [0], [[0], [1]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                evaluate_intervals([0, 0], [0, 0], [-1, -1], [1, 1], 0.1, indices)
        for alpha in (0, 1, math.inf, math.nan, True):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                evaluate_intervals([0], [0], [-1], [1], alpha, [0])
        for targets, predictions in (([math.nan], [0]), ([0], [math.inf])):
            with self.subTest(targets=targets, predictions=predictions), self.assertRaises(ValueError):
                evaluate_intervals(targets, predictions, [-1], [1], 0.1, [0])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    def test_plotting_preserves_exceptional_timestamps_without_infinite_bands(self):
        result = evaluate_intervals([0, 1, 2], [0, 1, 2], [-1, -math.inf, math.inf],
                                    [1, math.inf, -math.inf], 0.1, [100, 101, 102])
        original_indices = list(result["target_indices"])
        with tempfile.TemporaryDirectory() as temporary:
            paths = plot_aci_results({"a/b": wrap(result)}, [PAIR], 3, temporary)
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].is_file())
            self.assertEqual(paths[0].parent, Path(temporary))
            self.assertGreater(paths[0].stat().st_size, 0)
        self.assertEqual(result["target_indices"], original_indices)
        self.assertEqual(result["lower_interval"], [-1, -math.inf, math.inf])


if __name__ == "__main__":
    unittest.main()
