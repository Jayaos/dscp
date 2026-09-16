"""Exercise the actual spawn scheduler without requiring CUDA hardware."""

import importlib
import multiprocessing
import os
import sys
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "sbatch") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "sbatch"))
pool = importlib.import_module("sbatch_run_tuning.gpu_trial_pool")

_state = {}


def _initialize_worker(device, label):
    global _state
    _state = {"device": device, "label": label, "count": 0}


def _record_task(task):
    _state["count"] += 1
    time.sleep(0.02)
    return {"task": task, "pid": os.getpid(), **_state}


def _uneven_task(task):
    time.sleep(0.4 if _state["device"] == "slow" else 0.01)
    return {"task": task, "pid": os.getpid(), **_state}


def _failing_task(task):
    if task == "error":
        raise ValueError("intentional trial failure")
    if task == "exit":
        os._exit(23)
    return _record_task(task)


def _failing_initializer(device):
    raise ValueError(f"intentional initialization failure on {device}")


def _unserializable_result(task):
    return lambda: task


class GPUTrialPoolTests(unittest.TestCase):
    def setUp(self):
        self.existing_children = {child.pid for child in multiprocessing.active_children()}

    def tearDown(self):
        remaining = [
            child for child in multiprocessing.active_children()
            if child.pid not in self.existing_children
        ]
        # Clean up even after an assertion fails, then report leaked workers.
        for child in remaining:
            child.terminate()
            child.join(timeout=5)
        self.assertEqual(remaining, [], "pool left live child processes behind")

    def test_spawn_workers_cover_tasks_once_and_keep_device_affinity(self):
        records = list(pool.iter_parallel_trials(
            list(range(10)), ["device-a", "device-b"], _record_task,
            initializer=_initialize_worker, initargs=("initialized once",),
        ))
        self.assertEqual(sorted(record["task"] for record in records), list(range(10)))
        by_device = {}
        for record in records:
            by_device.setdefault(record["device"], []).append(record)
            self.assertEqual(record["label"], "initialized once")
            self.assertNotEqual(record["pid"], os.getpid())
        self.assertEqual(set(by_device), {"device-a", "device-b"})
        pids = set()
        for device_records in by_device.values():
            device_pids = {record["pid"] for record in device_records}
            self.assertEqual(len(device_pids), 1)
            pids.update(device_pids)
            self.assertEqual(
                sorted(record["count"] for record in device_records),
                list(range(1, len(device_records) + 1)),
            )
        self.assertEqual(len(pids), 2)

    def test_idle_worker_receives_next_trial_without_waiting_for_slow_worker(self):
        records = list(pool.iter_parallel_trials(
            list(range(10)), ["slow", "fast"], _uneven_task,
            initializer=_initialize_worker, initargs=("dynamic scheduling",),
        ))
        self.assertEqual(sorted(record["task"] for record in records), list(range(10)))
        fast_count = sum(record["device"] == "fast" for record in records)
        slow_count = sum(record["device"] == "slow" for record in records)
        self.assertGreater(fast_count, slow_count)
        self.assertGreater(slow_count, 0)

    def test_worker_exception_reaches_parent_and_cleans_up_pool(self):
        with self.assertRaisesRegex(Exception, "intentional trial failure"):
            list(pool.iter_parallel_trials(
                ["error", "other", "pending"], ["a", "b"], _failing_task,
                initializer=_initialize_worker, initargs=("exception",),
            ))

    def test_hard_worker_exit_is_detected_and_cleans_up_pool(self):
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "23|exit|died"):
            list(pool.iter_parallel_trials(
                ["exit", "other", "pending"], ["a", "b"], _failing_task,
                initializer=_initialize_worker, initargs=("hard exit",),
            ))
        self.assertLess(time.monotonic() - started, 20, "worker death detection took too long")

    def test_initializer_exception_reaches_parent_and_cleans_up_pool(self):
        with self.assertRaisesRegex(Exception, "intentional initialization failure"):
            list(pool.iter_parallel_trials(
                [1, 2], ["a", "b"], _record_task,
                initializer=_failing_initializer,
            ))

    def test_closing_generator_early_cleans_up_workers(self):
        results = pool.iter_parallel_trials(
            list(range(20)), ["a", "b"], _record_task,
            initializer=_initialize_worker, initargs=("early close",),
        )
        next(results)
        results.close()

    def test_unserializable_worker_result_fails_instead_of_losing_the_task(self):
        with self.assertRaisesRegex(RuntimeError, "pickle|local object"):
            list(pool.iter_parallel_trials([1], ["a"], _unserializable_result))


if __name__ == "__main__":
    unittest.main()
