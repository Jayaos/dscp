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
        }
        for metric, (mean, std) in expected.items():
            with self.subTest(metric=metric):
                self.assertAlmostEqual(values[f"avg_{metric}_mean"], mean)
                self.assertAlmostEqual(values[f"avg_{metric}_std"], std)

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

    def test_no_complete_window_has_unavailable_rolling_statistics(self):
        values = summarize_results(make_log(), 6)[PAIR]

        self.assertEqual(values["num_rolling_sequences"], 0)
        self.assertAlmostEqual(values["avg_coverage_mean"], 0.7)
        for metric in ("rolling_coverage", "delta_rolling_coverage"):
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
        self.assertIn("winkler", result.stdout.lower())

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
