"""Inspect evaluator CSVs without requiring model artifacts or dependencies."""

import contextlib
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from report.inspect_conditional_coverage_results import inspect_results


SCRIPT = Path(__file__).resolve().parents[1] / "report" / "inspect_conditional_coverage_results.py"


def summary_row(name="qr_rnn", method="qr_cp", dataset="air"):
    return {
        "name": name, "method": method, "dataset": dataset, "base_predictor": "lstm",
        "k": 2, "nominal_coverage": 0.9, "sequence_count": 1,
        "evaluated_sequence_count": 1, "unavailable_sequence_count": 0,
        "cace_definition": "mean_group_undercoverage",
        "cace": 0.123456, "wuc": 0.234567, "worst_sequence_wuc": 0.345678,
        "mean_sequence_coverage": 0.876543, "pooled_coverage": 0.765432,
        "group_size": 10, "evaluated_count": 8, "missing_count": 2,
        "covered_count": 7, "evaluated_group_count": 1, "empty_group_count": 1,
        "outside_shared_count": 0,
    }


def group_row(name="qr_rnn", method="qr_cp", sequence="station", key_type="str", group=0):
    empty = group == 1
    return {
        "name": name, "method": method, "sequence_key": sequence,
        "sequence_key_type": key_type, "nominal_coverage": 0.9,
        "lower_quantile": 0.05, "upper_quantile": 0.95, "group": group,
        "group_size": 2 if empty else 8,
        "evaluated_count": 0 if empty else 8,
        "missing_count": 2 if empty else 0,
        "covered_count": 0 if empty else 7,
        "coverage": None if empty else 0.875,
        "absolute_coverage_error": None if empty else 0.025,
        "undercoverage": None if empty else 0.025,
    }


class ConditionalCoverageInspectorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def make_report(self, name="evaluation", summaries=None, groups=None):
        directory = self.directory / name
        directory.mkdir()
        self.write_csv(directory / "summary.csv", summaries or [summary_row()])
        self.write_csv(directory / "group_metrics.csv", groups or [group_row(), group_row(group=1)])
        return directory

    def capture_report(self, directory):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = inspect_results(directory)
        return result, output.getvalue()

    def run_cli(self, *directories):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(str(path) for path in directories)],
            cwd=self.directory, capture_output=True, text=True, check=False,
        )

    def test_csv_only_report_preserves_saved_summary_and_missing_group_coverage(self):
        sequence = 'station, "north"'
        directory = self.make_report(groups=[
            group_row(sequence=sequence), group_row(sequence=sequence, group=1),
        ])
        before = {path.name: path.read_bytes() for path in directory.iterdir()}

        result, output = self.capture_report(directory)

        self.assertEqual(result["summary"][0]["cace"], "0.123456")
        self.assertEqual(result["summary"][0]["pooled_coverage"], "0.765432")
        self.assertEqual(len(result["group_metrics"]), 2)
        self.assertEqual(result["group_metrics"][0]["sequence_key"], sequence)
        self.assertEqual(result["group_metrics"][0]["coverage"], "0.875")
        self.assertEqual(result["group_metrics"][1]["coverage"], "")
        self.assertEqual(result["group_metrics"][1]["missing_count"], "2")
        for label in ("air", "lstm", "qr_rnn", "qr_cp", "CACE", "WUC", "0.875", "n/a"):
            with self.subTest(label=label):
                self.assertIn(label, output)
        # A display of the saved summaries must not recalculate them from groups.
        self.assertIn("0.123456", output)
        self.assertIn("0.765432", output)
        self.assertEqual(before, {path.name: path.read_bytes() for path in directory.iterdir()})

    def test_runs_and_sequence_key_types_keep_their_group_rows_separate(self):
        summaries = [summary_row(name="qr_rnn"), summary_row(name="iqn_rnn", method="iqn_cp")]
        for summary in summaries:
            summary.update(sequence_count=2, evaluated_sequence_count=2)
        groups = [
            group_row(name=name, method=method, sequence="7", key_type=key_type, group=group)
            for name, method in (("qr_rnn", "qr_cp"), ("iqn_rnn", "iqn_cp"))
            for key_type in ("int", "str")
            for group in (0, 1)
        ]
        directory = self.make_report(summaries=summaries, groups=groups)

        result, output = self.capture_report(directory)

        identities = {
            (row["name"], row["method"], row["sequence_key"], row["sequence_key_type"], row["group"])
            for row in result["group_metrics"]
        }
        self.assertEqual(len(identities), 8)
        for label in ("qr_rnn", "iqn_rnn", "qr_cp", "iqn_cp", "int", "str"):
            with self.subTest(label=label):
                self.assertIn(label, output)

    def test_cli_accepts_multiple_datasets_from_outside_repository(self):
        air = self.make_report("air_results")
        solar = self.make_report("solar_results", summaries=[summary_row(dataset="solar")])

        completed = self.run_cli(air, solar)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for label in ("air", "solar", "CACE", "WUC", "0.875"):
            with self.subTest(label=label):
                self.assertIn(label, completed.stdout)
        self.assertEqual(completed.stderr, "")

    def test_missing_group_csv_has_clear_cli_error(self):
        directory = self.make_report()
        (directory / "group_metrics.csv").unlink()

        completed = self.run_cli(directory)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("group_metrics.csv", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_required_coverage_column_is_validated(self):
        directory = self.make_report()
        row = group_row()
        del row["coverage"]
        self.write_csv(directory / "group_metrics.csv", [row])

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "coverage"):
                inspect_results(directory)
        completed = self.run_cli(directory)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("coverage", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
