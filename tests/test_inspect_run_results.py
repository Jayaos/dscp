import contextlib
import copy
import io
import math
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from report.inspect_run_results import inspect_results, summarize_results


PAIR = (0.1, 0.9)
SCRIPT = Path(__file__).resolve().parents[1] / "report" / "inspect_run_results.py"


def make_log():
    return {
        "first": {
            "evaluation_results": {
                PAIR: {
                    "coverage": [True, False, False, False, True],
                    "interval_width": [1, 2, 3, 4, 5],
                    "winkler_score": [2, 4, 6, 8, 10],
                }
            }
        },
        "second": {
            "evaluation_results": {
                PAIR: {
                    "coverage": [True, True, True],
                    "interval_width": [7, 9, 11],
                    "winkler_score": [10, 12, 14],
                }
            }
        },
        # Saved aggregate values must not replace recomputation over sequences.
        "summary_results": {PAIR: {"avg_coverage": -100}},
    }


def make_aci_log(include_no_finite=False):
    scope = {
        "coverage": "all_test_predictions",
        "avg_interval_width": "finite_nonempty_intervals_only",
        "avg_winkler_score": "finite_nonempty_intervals_only",
    }
    result = {
        "coverage": [True, True, True, False, False, True],
        "interval_width": [2, math.inf, math.inf, 0, 4, 6],
        "winkler_score": [2, math.inf, math.inf, math.inf, 24, 6],
        "finite_interval_mask": [True, False, False, False, True, True],
        "interval_kind": ["regular", "full", "full", "empty", "regular", "regular"],
        "target_indices": [100, 101, 102, 103, 104, 105],
        "num_test_points": 6, "num_finite_intervals": 3,
        "num_full_sets": 2, "num_empty_sets": 1,
        "full_set_rate": 2 / 6, "empty_set_rate": 1 / 6,
        "metrics_scope": dict(scope),
        # Cached averages must still not replace recomputation from raw scores.
        "avg_interval_width": -100, "avg_winkler_score": -100,
    }
    log = {"first": {"metadata": {"method": "aci"}, "evaluation_results": {PAIR: result}}}
    if include_no_finite:
        exceptional = {
            "coverage": [True, False],
            "interval_width": [math.inf, 0], "winkler_score": [math.inf, math.inf],
            "finite_interval_mask": [False, False], "interval_kind": ["full", "empty"],
            "target_indices": [20, 21],
            "num_test_points": 2, "num_finite_intervals": 0,
            "num_full_sets": 1, "num_empty_sets": 1,
            "full_set_rate": 0.5, "empty_set_rate": 0.5,
            "metrics_scope": dict(scope),
        }
        log["second"] = {"metadata": {"method": "aci"}, "evaluation_results": {PAIR: exceptional}}
    return log


class RunResultSummaryTests(unittest.TestCase):
    def test_sequences_receive_equal_weight_and_population_std(self):
        values = summarize_results(make_log(), 2)[PAIR]

        self.assertEqual(values["num_sequences"], 2)
        self.assertEqual(values["num_rolling_sequences"], 2)
        self.assertAlmostEqual(values["target_coverage"], 0.8)
        expected = {
            "coverage": (0.7, 0.3),
            "interval_width": (6.0, 3.0),
            "winkler_score": (9.0, 3.0),
            "delta_coverage": (-0.1, 0.3),
            "rolling_coverage": (0.625, 0.375),
            "delta_rolling_coverage": (-0.175, 0.375),
            "rolling_undercoverage": (0.275, 0.275),
        }
        for metric, (mean, std) in expected.items():
            with self.subTest(metric=metric):
                self.assertAlmostEqual(values[f"avg_{metric}_mean"], mean)
                self.assertAlmostEqual(values[f"avg_{metric}_std"], std)

    def test_rolling_undercoverage_clips_each_window_and_weights_sequences_equally(self):
        log = make_log()
        log["first"]["evaluation_results"][PAIR]["coverage"] = [True, True, False, False, True]

        values = summarize_results(log, 2)[PAIR]

        # Four windows have deficits [0, 0.3, 0.8, 0.3], averaging 0.35.
        # The second sequence's two windows both have zero deficit.
        # Average the two sequence means, rather than pooling six windows
        # or clipping the first sequence's average coverage gap of 0.3.
        self.assertAlmostEqual(values["avg_rolling_undercoverage_mean"], 0.175)
        self.assertAlmostEqual(values["avg_rolling_undercoverage_std"], 0.175)

    def test_fully_covered_sequences_have_zero_rolling_undercoverage(self):
        log = make_log()
        for name in ("first", "second"):
            metrics = log[name]["evaluation_results"][PAIR]
            metrics["coverage"] = [True] * len(metrics["coverage"])

        values = summarize_results(log, 2)[PAIR]

        self.assertEqual(values["avg_rolling_undercoverage_mean"], 0.0)
        self.assertEqual(values["avg_rolling_undercoverage_std"], 0.0)

    def test_pair_order_and_dictionary_order_do_not_mix_metrics(self):
        log = make_log()
        reverse_pair = (0.9, 0.1)
        for name in ("first", "second"):
            metrics = log[name]["evaluation_results"]
            other = copy.deepcopy(metrics[PAIR])
            other["coverage"] = [False] * len(other["coverage"])
            if name == "first":
                metrics[reverse_pair] = other
            else:
                log[name]["evaluation_results"] = {reverse_pair: other, **metrics}

        summary = summarize_results(log, 2)

        self.assertEqual(set(summary), {PAIR, reverse_pair})
        self.assertAlmostEqual(summary[PAIR]["avg_coverage_mean"], 0.7)
        self.assertAlmostEqual(summary[reverse_pair]["target_coverage"], 0.8)
        self.assertEqual(summary[reverse_pair]["avg_coverage_mean"], 0.0)
        self.assertAlmostEqual(summary[reverse_pair]["avg_delta_coverage_mean"], -0.8)

    def test_window_one_matches_ordinary_coverage(self):
        values = summarize_results(make_log(), 1)[PAIR]

        for stat in ("mean", "std"):
            self.assertAlmostEqual(
                values[f"avg_rolling_coverage_{stat}"],
                values[f"avg_coverage_{stat}"],
            )
            self.assertAlmostEqual(
                values[f"avg_delta_rolling_coverage_{stat}"],
                values[f"avg_delta_coverage_{stat}"],
            )

    def test_window_equal_to_sequence_length_and_shorter_sequence(self):
        values = summarize_results(make_log(), 5)[PAIR]

        self.assertEqual(values["num_sequences"], 2)
        self.assertEqual(values["num_rolling_sequences"], 1)
        self.assertAlmostEqual(values["avg_coverage_mean"], 0.7)
        self.assertAlmostEqual(values["avg_rolling_coverage_mean"], 0.4)
        self.assertEqual(values["avg_rolling_coverage_std"], 0.0)
        self.assertAlmostEqual(values["avg_delta_rolling_coverage_mean"], -0.4)
        self.assertEqual(values["avg_delta_rolling_coverage_std"], 0.0)
        self.assertAlmostEqual(values["avg_rolling_undercoverage_mean"], 0.4)
        self.assertEqual(values["avg_rolling_undercoverage_std"], 0.0)

    def test_no_complete_window_has_unavailable_rolling_statistics(self):
        values = summarize_results(make_log(), 6)[PAIR]

        self.assertEqual(values["num_rolling_sequences"], 0)
        self.assertAlmostEqual(values["avg_coverage_mean"], 0.7)
        for metric in ("rolling_coverage", "delta_rolling_coverage", "rolling_undercoverage"):
            for stat in ("mean", "std"):
                self.assertIsNone(values[f"avg_{metric}_{stat}"])

    def test_column_vectors_match_one_dimensional_arrays(self):
        log = make_log()
        for name in ("first", "second"):
            metrics = log[name]["evaluation_results"][PAIR]
            for key, values in metrics.items():
                metrics[key] = np.asarray(values).reshape(-1, 1)

        self.assertEqual(summarize_results(log, 2), summarize_results(make_log(), 2))

    def test_infinite_interval_metrics_are_retained(self):
        log = make_log()
        for metric in ("interval_width", "winkler_score"):
            log["first"]["evaluation_results"][PAIR][metric][0] = math.inf

        values = summarize_results(log, 2)[PAIR]

        for metric in ("interval_width", "winkler_score"):
            self.assertEqual(values[f"avg_{metric}_mean"], math.inf)
            self.assertIsNone(values[f"avg_{metric}_std"])
        self.assertAlmostEqual(values["avg_coverage_mean"], 0.7)

    def test_invalid_window_sizes_are_rejected(self):
        for window in (0, -1, True, False, 1.5, "2", None):
            with self.subTest(window=window), self.assertRaises(ValueError):
                summarize_results(make_log(), window)

    def test_empty_logs_are_rejected(self):
        for log in ({}, {"summary_results": {}}):
            with self.subTest(log=log), self.assertRaises(ValueError):
                summarize_results(log, 2)

    def test_missing_mismatched_and_empty_metric_arrays_are_rejected(self):
        for error in ("missing", "mismatched", "empty"):
            log = make_log()
            metrics = log["first"]["evaluation_results"][PAIR]
            if error == "missing":
                del metrics["winkler_score"]
            elif error == "mismatched":
                metrics["interval_width"].pop()
            else:
                metrics.update({key: [] for key in metrics})
            with self.subTest(error=error), self.assertRaises(ValueError):
                summarize_results(log, 2)

    def test_nan_in_any_metric_is_rejected(self):
        for metric in ("coverage", "interval_width", "winkler_score"):
            log = make_log()
            log["first"]["evaluation_results"][PAIR][metric][0] = math.nan
            with self.subTest(metric=metric), self.assertRaises(ValueError):
                summarize_results(log, 2)

    def test_nonbinary_coverage_is_rejected(self):
        for invalid in (0.5, 2, -1, math.inf):
            log = make_log()
            log["first"]["evaluation_results"][PAIR]["coverage"][0] = invalid
            with self.subTest(coverage=invalid), self.assertRaises(ValueError):
                summarize_results(log, 2)


class ACIResultSummaryTests(unittest.TestCase):
    def test_conditional_efficiency_keeps_complete_coverage_and_rolling_timeline(self):
        log = make_aci_log()
        row = summarize_results(log, 2)[PAIR]
        self.assertAlmostEqual(row["avg_coverage_mean"], 4 / 6)
        self.assertAlmostEqual(row["avg_rolling_coverage_mean"], 0.6)
        self.assertAlmostEqual(row["avg_rolling_undercoverage_mean"], 0.28)
        self.assertEqual(row["avg_interval_width_mean"], 4)
        self.assertAlmostEqual(row["avg_winkler_score_mean"], 32 / 3)
        self.assertEqual(row["num_test_points"], 6)
        self.assertEqual(row["total_points"], 6)
        self.assertEqual(row["evaluated_points"], 6)
        self.assertEqual(row["excluded_points_count"], 0)
        self.assertEqual(row["num_full_sets"], 2)
        self.assertEqual(row["num_empty_sets"], 1)
        self.assertEqual(row["full_test_avg_interval_width_mean"], math.inf)
        self.assertIsNone(row["full_test_avg_interval_width_std"])
        self.assertEqual(log["first"]["evaluation_results"][PAIR]["target_indices"],
                         [100, 101, 102, 103, 104, 105])

    def test_no_finite_sequence_still_contributes_coverage_and_exception_counts(self):
        row = summarize_results(make_aci_log(include_no_finite=True), 2)[PAIR]
        self.assertEqual(row["num_sequences"], 2)
        self.assertEqual(row["num_rolling_sequences"], 2)
        self.assertEqual(row["num_efficiency_sequences"], 1)
        self.assertEqual(row["num_undefined_efficiency_sequences"], 1)
        self.assertAlmostEqual(row["avg_coverage_mean"], (4 / 6 + 0.5) / 2)
        self.assertAlmostEqual(row["avg_rolling_coverage_mean"], 0.55)
        self.assertEqual(row["avg_interval_width_mean"], 4)
        self.assertEqual(row["avg_interval_width_std"], 0)
        self.assertEqual(row["num_test_points"], 8)
        self.assertEqual(row["num_finite_intervals"], 3)
        self.assertEqual(row["full_set_rate"], 3 / 8)
        self.assertEqual(row["empty_set_rate"], 2 / 8)

    def test_all_exceptional_sets_leave_efficiency_undefined(self):
        log = make_aci_log(include_no_finite=True)
        del log["first"]
        row = summarize_results(log, 2)[PAIR]
        self.assertEqual(row["avg_coverage_mean"], 0.5)
        self.assertEqual(row["avg_rolling_coverage_mean"], 0.5)
        for metric in ("interval_width", "winkler_score"):
            self.assertIsNone(row[f"avg_{metric}_mean"])
            self.assertIsNone(row[f"avg_{metric}_std"])
        self.assertEqual(row["num_efficiency_sequences"], 0)
        self.assertEqual(row["num_undefined_efficiency_sequences"], 1)

    def test_invalid_scope_mask_and_exception_diagnostics_are_rejected(self):
        changes = (
            ("finite_interval_mask", [True]),
            ("finite_interval_mask", [1, 0, 0, 0, 1, 1]),
            ("finite_interval_mask", [True, True, False, False, True, True]),
            ("interval_kind", ["regular"] * 6),
            ("num_full_sets", 1),
            ("num_empty_sets", True),
            ("full_set_rate", 0.9),
            ("coverage", [True, False, True, False, False, True]),
            ("interval_width", [2, math.inf, math.inf, -math.inf, 4, 6]),
            ("winkler_score", [2, math.inf, math.inf, math.inf, math.inf, 6]),
        )
        for name, invalid in changes:
            log = make_aci_log()
            log["first"]["evaluation_results"][PAIR][name] = invalid
            with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                summarize_results(log, 2)
        log = make_aci_log()
        del log["first"]["evaluation_results"][PAIR]["metrics_scope"]["avg_winkler_score"]
        with self.assertRaisesRegex(ValueError, "metrics_scope"):
            summarize_results(log, 2)

    def test_mixed_scopes_between_sequences_are_rejected(self):
        log = make_aci_log(include_no_finite=True)
        del log["second"]["evaluation_results"][PAIR]["metrics_scope"]
        with self.assertRaisesRegex(ValueError, "scope differs"):
            summarize_results(log, 2)

    def test_undeclared_efficiency_mask_does_not_change_legacy_behavior(self):
        log = make_log()
        log["first"]["evaluation_results"][PAIR]["finite_interval_mask"] = [False] * 5
        self.assertEqual(summarize_results(log, 2), summarize_results(make_log(), 2))


class RunResultInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.results_dir = Path(self.temp_dir.name) / "results" / "split_cp" / "solar" / "lr" / "run"
        self.results_dir.mkdir(parents=True)
        with (self.results_dir / "log.pkl").open("wb") as stream:
            pickle.dump(make_log(), stream)

    def write_config(self):
        (self.results_dir / "resolved_config.yaml").write_text(
            "data:\n"
            "  data_path: ./data/solar_prediction/lr/lr_nsdb-60m_data.pkl\n"
            "model:\n"
            "  target_quantiles: [[0.1, 0.9]]\n"
            "  prediction_step: 1\n"
            "saving_dir: ./results/split_cp/solar/lr/run/\n",
            encoding="utf-8",
        )

    def test_inspection_prints_run_identity_and_model_hyperparameters(self):
        self.write_config()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            summary = inspect_results(self.results_dir, 2)

        self.assertEqual(summary, summarize_results(make_log(), 2))
        for expected in ("split_cp", "solar", "lr", "prediction_step", "target_quantiles"):
            with self.subTest(expected=expected):
                self.assertIn(expected, output.getvalue())

    def test_missing_config_does_not_prevent_metrics(self):
        with contextlib.redirect_stdout(io.StringIO()):
            summary = inspect_results(self.results_dir, 2)

        self.assertAlmostEqual(summary[PAIR]["avg_coverage_mean"], 0.7)

    def test_cli_reads_results_directory(self):
        self.write_config()

        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.results_dir), "--rolling-window-size", "2"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("split_cp", result.stdout)
        self.assertIn("rolling", result.stdout.lower())
        self.assertIn("Rolling undercoverage", result.stdout)
        self.assertIn("winkler", result.stdout.lower())

    def test_inspection_prints_aci_scope_exception_frequencies_and_full_test_diagnostics(self):
        with (self.results_dir / "log.pkl").open("wb") as stream:
            pickle.dump(make_aci_log(include_no_finite=True), stream)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summary = inspect_results(self.results_dir, 2)
        rendered = output.getvalue()
        for expected in (
            "Coverage and rolling coverage include every test prediction",
            "Width/Winkler scope: finite, nonempty intervals only",
            "evaluated=8, excluded=0, total=8",
            "full sets: 3 (37.500000%)", "empty sets: 2 (25.000000%)",
            "undefined efficiency sequences: 1", "Full-test mean width: inf",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, rendered)
        self.assertEqual(summary[PAIR]["num_sequences"], 2)

    def test_cli_help_and_invalid_window(self):
        help_result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--rolling-window-size", help_result.stdout)

        for arguments in ([], ["--rolling-window-size", "0"]):
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(self.results_dir), *arguments],
                capture_output=True,
                text=True,
                timeout=30,
            )
            with self.subTest(arguments=arguments):
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
