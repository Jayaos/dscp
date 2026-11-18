"""Check signed group box plots against saved evaluator CSV identities and values."""

import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from report.plot_conditional_coverage_results import (
    load_plot_data,
    plot_conditional_coverage_results,
)


SCRIPT = Path(__file__).resolve().parents[1] / "report" / "plot_conditional_coverage_results.py"
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def summary_row(name="qr_rnn", method="qr_cp", dataset="air", predictor="lstm", k=3):
    return {
        "name": name, "method": method, "dataset": dataset, "base_predictor": predictor,
        "k": k, "nominal_coverage": .9, "cace": .01, "wuc": .02,
        "mean_sequence_coverage": .91, "pooled_coverage": .92,
    }


def group_row(group, coverage, name="qr_rnn", method="qr_cp", sequence="station", key_type="str"):
    evaluated = 0 if coverage is None else 100
    return {
        "name": name, "method": method,
        "sequence_key": sequence, "sequence_key_type": key_type,
        "nominal_coverage": .9, "group": group, "coverage": coverage,
        "group_size": 100, "evaluated_count": evaluated,
        "missing_count": 100 - evaluated,
        "covered_count": 0 if coverage is None else round(100 * coverage),
    }


class ConditionalCoveragePlotTests(unittest.TestCase):
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
        self.write_csv(directory / "group_metrics.csv", groups or [
            group_row(0, .85), group_row(1, .95), group_row(2, None),
        ])
        return directory

    def test_signed_values_preserve_undercoverage_overcoverage_zero_and_missing(self):
        groups = [group_row(group, coverage) for group, coverage in enumerate([.85, .95, .9, 0, None])]
        directory = self.make_report(summaries=[summary_row(k=5)], groups=groups)
        before = {path.name: path.read_bytes() for path in directory.iterdir()}

        chart, = load_plot_data(directory)

        self.assertEqual(chart["sequences"], [("str", "station")])
        self.assertEqual(chart["k"], 5)
        for actual, expected in zip(chart["errors"][0][:-1], [-.05, .05, 0., -.9]):
            self.assertAlmostEqual(actual, expected)
        self.assertIsNone(chart["errors"][0][-1])
        self.assertEqual(before, {path.name: path.read_bytes() for path in directory.iterdir()})

    def test_groups_use_numeric_order_and_typed_sequences_remain_distinct(self):
        group_order = [10, 2, 0, 1, 9, 8, 7, 6, 5, 4, 3]
        groups = [
            group_row(group, group / 20, sequence="7", key_type=key_type)
            for key_type in ("int", "str") for group in group_order
        ]
        directory = self.make_report(summaries=[summary_row(k=11)], groups=groups)

        chart, = load_plot_data(directory)

        self.assertEqual(chart["sequences"], [("int", "7"), ("str", "7")])
        self.assertEqual(len(chart["errors"]), 2)
        for values in chart["errors"]:
            self.assertEqual(len(values), 11)
            for group, value in enumerate(values):
                self.assertAlmostEqual(value, group / 20 - .9)

    def test_run_variants_and_datasets_keep_separate_metadata(self):
        summaries = [summary_row(name="qr_rnn"), summary_row(name="qr_transformer")]
        groups = [
            group_row(group, .85 if name == "qr_rnn" else .95, name=name)
            for name in ("qr_rnn", "qr_transformer") for group in range(3)
        ]
        air = self.make_report("air", summaries=summaries, groups=groups)
        solar = self.make_report("solar", summaries=[summary_row(dataset="solar", predictor="chronos")])

        charts = load_plot_data(air)
        other, = load_plot_data(solar)

        self.assertEqual([chart["name"] for chart in charts], ["qr_rnn", "qr_transformer"])
        self.assertTrue(all(chart["dataset"] == "air" for chart in charts))
        self.assertLess(charts[0]["errors"][0][0], 0)
        self.assertGreater(charts[1]["errors"][0][0], 0)
        self.assertEqual((other["dataset"], other["base_predictor"]), ("solar", "chronos"))

    def test_invalid_group_identity_and_incomplete_partitions_are_rejected(self):
        normal = [group_row(group, .9) for group in range(3)]
        cases = {
            "unknown_run": [{**normal[0], "name": "unknown"}, *normal[1:]],
            "wrong_method": [{**normal[0], "method": "iqn_cp"}, *normal[1:]],
            "duplicate": [*normal, normal[0]],
            "missing_group": normal[:-1],
            "negative_group": [{**normal[0], "group": -1}, *normal[1:]],
            "group_past_k": [{**normal[0], "group": 3}, *normal[1:]],
            "different_nominal": [{**normal[0], "nominal_coverage": .8}, *normal[1:]],
        }
        for name, groups in cases.items():
            with self.subTest(case=name):
                directory = self.make_report(name, groups=groups)
                with self.assertRaises(ValueError):
                    load_plot_data(directory)

    def test_invalid_coverage_nominal_and_k_are_rejected(self):
        for index, coverage in enumerate([-.1, 1.1, "nan", "inf", "invalid"]):
            with self.subTest(coverage=coverage):
                groups = [group_row(0, .9), group_row(1, .9), group_row(2, .9)]
                groups[0]["coverage"] = coverage
                directory = self.make_report(f"coverage_{index}", groups=groups)
                with self.assertRaises(ValueError):
                    load_plot_data(directory)
        for index, (field, value) in enumerate([
            ("nominal_coverage", 0), ("nominal_coverage", 1),
            ("nominal_coverage", "nan"), ("k", 0), ("k", "2.5"),
        ]):
            with self.subTest(field=field, value=value):
                summary = {**summary_row(), field: value}
                directory = self.make_report(f"summary_{index}", summaries=[summary])
                with self.assertRaises(ValueError):
                    load_plot_data(directory)

    @unittest.skipUnless(HAS_MATPLOTLIB, "Matplotlib is required for rendering")
    def test_boxes_summarize_each_sequences_available_groups_without_filling_missing_values(self):
        from matplotlib.axes import Axes

        groups = [
            group_row(group, value, sequence=sequence)
            for sequence, values in (("complete", [.8, .9, 1.]),
                                     ("empty", [None, None, None]),
                                     ("partial", [.94, None, None]))
            for group, value in enumerate(values)
        ]
        directory = self.make_report(groups=groups)
        statistics, positions, axes = [], [], []
        original_bxp = Axes.bxp

        def capture_statistics(ax, bxpstats, *args, **kwargs):
            statistics.extend(bxpstats)
            positions.extend(kwargs["positions"])
            axes.append(ax)
            return original_bxp(ax, bxpstats, *args, **kwargs)

        with patch.object(Axes, "bxp", new=capture_statistics):
            paths = plot_conditional_coverage_results(directory)

        self.assertEqual(len(paths), 1)
        self.assertEqual(positions, [0, 2])
        self.assertEqual(len(statistics), 2)
        # The complete sequence has three equally weighted group errors.
        for key, expected in (("med", 0.), ("q1", -.05), ("q3", .05),
                              ("whislo", -.1), ("whishi", .1)):
            self.assertAlmostEqual(statistics[0][key], expected)
        # Missing groups must not introduce zero-valued observations.
        for key in ("med", "q1", "q3", "whislo", "whishi"):
            self.assertAlmostEqual(statistics[1][key], .04)
        self.assertIn("empty", " ".join(label.get_text() for label in axes[0].get_xticklabels()))
        self.assertTrue(any("n/a" in text.get_text() for text in axes[0].texts))

    @unittest.skipUnless(HAS_MATPLOTLIB, "Matplotlib is required for rendering")
    def test_plot_writes_one_pdf_per_run_without_modifying_input_csvs(self):
        names = ("qr_rnn", "qr_transformer")
        groups = [
            group_row(group, coverage, name=name)
            for name in names for group, coverage in enumerate([.85, .95, None])
        ]
        directory = self.make_report(summaries=[summary_row(name=name) for name in names], groups=groups)
        before = {path.name: path.read_bytes() for path in directory.glob("*.csv")}

        paths = plot_conditional_coverage_results(directory)

        self.assertEqual(len(paths), 2)
        self.assertEqual(len(set(paths)), 2)
        for path in paths:
            self.assertEqual(path.parent, directory / "plots")
            self.assertEqual(path.suffix, ".pdf")
            self.assertTrue(path.read_bytes().startswith(b"%PDF"))
            for token in ("air", "lstm", "qr_cp", "k3"):
                self.assertIn(token, path.name)
        for name in names:
            self.assertTrue(any(name in path.name for path in paths))
        self.assertEqual(before, {path.name: path.read_bytes() for path in directory.glob("*.csv")})

    @unittest.skipUnless(HAS_MATPLOTLIB, "Matplotlib is required for rendering")
    def test_cli_renders_png_from_outside_repository(self):
        directory = self.make_report()
        destination = self.directory / "saved plots"

        completed = subprocess.run(
            [sys.executable, str(SCRIPT), str(directory), "--save-dir", str(destination), "--format", "png"],
            cwd=self.directory, capture_output=True, text=True, check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        images = list(destination.glob("*.png"))
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
