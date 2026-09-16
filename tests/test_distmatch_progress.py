"""Check per-sequence ETA reporting without changing DistMatch predictions."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from baselines.distmatch.config import validate_config
from baselines.distmatch.model import DistMatchResidualIntervalEstimator
from baselines.distmatch.progress import SequenceProgress
from baselines.distmatch.run_distmatch import evaluate_sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def _config():
    return {
        "seed": 37,
        "num_cores": 1,
        "threads_per_worker": 1,
        "data": {"train_ratio": .5, "valid_ratio": .2, "test_ratio": .3},
        "model": {
            "past_window_len": 3, "n_trees": 2, "qrf_n_estimators": 2,
            "qrf_max_depth": 2, "beta_bins": 3,
        },
    }


class DistMatchProgressTests(unittest.TestCase):
    def test_matching_callback_counts_actual_work_and_finishes_trees(self):
        residuals = np.sin(np.arange(12, dtype=float))
        options = dict(past_window_len=3, n_trees=2, ks_block_size=2)
        events = []
        model = DistMatchResidualIntervalEstimator(**options).fit(
            residuals, progress=lambda *event: events.append(event),
        )
        matching = [event for event in events if event[0] == "matching"]
        self.assertEqual(matching[0], ("matching", 0, 45))
        self.assertEqual(matching[-1], ("matching", 45, 45))
        self.assertTrue(all(event[2] == 45 for event in matching))
        self.assertTrue(all(
            0 < right[1] - left[1] <= options["ks_block_size"]
            for left, right in zip(matching, matching[1:])
        ))
        self.assertEqual(events[len(matching):], [
            ("trees", 0, 2), ("trees", 1, 2), ("trees", 2, 2),
        ])
        plain = DistMatchResidualIntervalEstimator(**options).fit(residuals)
        self.assertEqual(model.diagnostics(), plain.diagnostics())

    def test_cache_hit_skips_matching_progress(self):
        residuals = np.sin(np.arange(12, dtype=float))
        with tempfile.TemporaryDirectory() as directory:
            options = dict(past_window_len=3, n_trees=2, cache_dir=directory)
            DistMatchResidualIntervalEstimator(**options).fit(residuals)
            events = []
            with patch.object(
                DistMatchResidualIntervalEstimator, "_fill_match_matrix",
                side_effect=AssertionError("Cache hit should not recompute matching"),
            ):
                cached = DistMatchResidualIntervalEstimator(**options).fit(
                    residuals, progress=lambda *event: events.append(event),
                )
        self.assertTrue(cached.diagnostics()["cache"]["hit"])
        self.assertEqual(events, [("trees", 0, 2), ("trees", 1, 2), ("trees", 2, 2)])

    def test_redirected_logs_throttle_and_estimate_remaining_seconds(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch(
            "baselines.distmatch.progress.time.monotonic", side_effect=[100, 109, 110, 120],
        ), SequenceProgress("station") as progress:
            progress.update("test", 0, 4)
            progress.update("test", 1, 4)
            progress.update("test", 1, 4)
            progress.update("test", 4, 4)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("DistMatch 'station' test", lines[1])
        self.assertIn("1/4", lines[1])
        self.assertIn("ETA 00:30", lines[1])
        self.assertIn("4/4", lines[2])
        self.assertIn("ETA 00:00", lines[2])
        self.assertNotIn("\r", output.getvalue())
        self.assertNotIn("\x1b", output.getvalue())

    def test_enabled_and_disabled_progress_produce_identical_intervals(self):
        item = {
            "heldout_y": np.sin(np.arange(30, dtype=float)),
            "heldout_predictions": np.zeros(30),
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            enabled = evaluate_sequence("station", item, _config())
        for stage in ("matching", "trees", "replay", "test"):
            self.assertIn(f"DistMatch 'station' {stage}", output.getvalue())
        self.assertIn("ETA", output.getvalue())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            disabled = evaluate_sequence("station", item, {**_config(), "show_progress": False})
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(enabled["evaluation_results"], disabled["evaluation_results"])
        self.assertEqual(enabled["metadata"]["diagnostics"], disabled["metadata"]["diagnostics"])

    def test_progress_setting_is_boolean_and_defaults_on(self):
        self.assertTrue(validate_config(_config())["show_progress"])
        for value in ("false", 0, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "show_progress"):
                validate_config({**_config(), "show_progress": value})

    def test_real_spawn_and_serial_logs_report_each_sequence_and_agree(self):
        script = textwrap.dedent('''
            import numpy as np
            from baselines.distmatch.run_distmatch import evaluate_sequences

            if __name__ == '__main__':
                config = {
                    'seed': 37, 'num_cores': 1, 'threads_per_worker': 1,
                    'data': {'train_ratio': .5, 'valid_ratio': .2, 'test_ratio': .3},
                    'model': {'past_window_len': 3, 'n_trees': 2,
                              'qrf_n_estimators': 2, 'qrf_max_depth': 2, 'beta_bins': 3},
                }
                data = {
                    key: {'heldout_y': np.sin(np.arange(length, dtype=float)),
                          'heldout_predictions': np.zeros(length)}
                    for key, length in [('alpha', 30), ('beta', 33)]
                }
                print('SERIAL START', flush=True)
                serial = evaluate_sequences(data, config)
                print('PARALLEL START', flush=True)
                parallel = evaluate_sequences(data, config, num_cores=2)
                assert list(parallel) == list(data)
                for key in data:
                    assert serial[key]['evaluation_results'] == parallel[key]['evaluation_results']
                    assert serial[key]['metadata']['sequence_seed'] == parallel[key]['metadata']['sequence_seed']
                print('RESULTS AGREE', flush=True)
        ''')
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=120, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("RESULTS AGREE", completed.stdout)
        serial, parallel = completed.stdout.split("PARALLEL START", maxsplit=1)
        for mode, output in (("serial", serial), ("parallel", parallel)):
            for key in ("alpha", "beta"):
                with self.subTest(mode=mode, key=key):
                    test_lines = [
                        line for line in output.splitlines()
                        if f"DistMatch {key!r} test" in line
                    ]
                    self.assertTrue(test_lines, output)
                    self.assertTrue(any("100%" in line and "ETA 00:00" in line for line in test_lines))
            self.assertNotIn("\x1b", output)


if __name__ == "__main__":
    unittest.main()
