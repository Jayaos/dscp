"""Parallel Local-CP trials retain per-sequence calibration and a global ranking."""

import contextlib
import copy
import importlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "sbatch") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "sbatch"))
tuning = importlib.import_module("sbatch_run_tuning.run_local_cp_tuning")
pool = importlib.import_module("sbatch_run_tuning.gpu_trial_pool")

_cpu_data = None
_cpu_keys = None
_cpu_cache = None


def _initialize_cpu_worker(device, data, keys):
    global _cpu_data, _cpu_keys, _cpu_cache
    assert device.startswith("cpu-worker-")
    torch.set_num_threads(1)
    _cpu_data, _cpu_keys, _cpu_cache = data, keys, {}


def _run_cpu_trial(task):
    index, config, values, seed = task
    return tuning._run_grid_trial(
        index, OmegaConf.create(config), values, _cpu_data, _cpu_keys, seed, _cpu_cache,
    )


def _small_config(encoder="rnn", rolling=True):
    config = OmegaConf.load(
        REPO_ROOT / "configs" / "lcp_configs" / f"lcp_{encoder}_chronos_air_config.yaml"
    )
    config.device = "cpu"
    config.model.dim_model = 4
    config.model.num_layers = 1
    config.model.window_size = 5
    config.model.calibration_size = 4
    config.model.rolling_calibration = rolling
    config.model.sampling_num = 32
    config.training.epochs = 2
    config.training.batch_size = 64
    config.tuning = {"delta_threshold": 0.0, "model_selection_valid_ratio": 0.2}
    return config


def _artifact():
    values = np.arange(100, dtype=np.float32)
    return {
        key: {
            "heldout_x": np.stack((np.sin(values / 8), np.cos(values / 11)), axis=1),
            "heldout_y": np.sin(values / 7) + offset,
            "heldout_predictions": 0.2 * np.cos(values / 9),
        }
        for key, offset in (("first", 0.3), ("second", -0.5))
    }


def _fake_records(config):
    return [
        {
            "trial_index": index,
            "sequence_keys": ["a", "b"],
            "grid_values": {"training.learning_rate": index / 1000},
            "resolved_config": tuning.plain_config(config),
            "result": {
                "positive_delta_coverage": index != 4,
                "selection_score": {1: 2.0, 2: 1.0, 3: 1.0, 4: 0.0}[index],
            },
        }
        for index in range(1, 5)
    ]


class LocalMultiGPUTuningTests(unittest.TestCase):
    @contextlib.contextmanager
    def _main_setup(self, *, visible_gpus=2, num_gpus=2, tuning_num_gpus=1):
        config = _small_config()
        config.device = 0
        args = SimpleNamespace(
            base_config=Path("unused_base.yaml"), grid_config=Path("unused_grid.yaml"),
            save_dir=REPO_ROOT / "unused_lcp_multigpu_results", sequence_key=None,
            sequence_index=0, top_k=3, seed=2026, num_gpus=num_gpus,
        )
        with contextlib.ExitStack() as stack:
            def mock(name, **kwargs):
                return stack.enter_context(patch.object(tuning, name, **kwargs))
            mock("parse_args", return_value=args)
            stack.enter_context(patch.object(OmegaConf, "load", return_value=config))
            mock("load_grid", return_value=(
                {"training.learning_rate": [0.001, 0.002, 0.003, 0.004]},
                {"num_sequences": "all", "num_gpus": tuning_num_gpus},
            ))
            load_data = mock("load_data", return_value={"b": {}, "a": {}})
            parallel = mock("iter_parallel_trials", return_value=(
                record for record in reversed(_fake_records(config))
            ))
            serial = mock("_run_grid_trial", side_effect=_fake_records(config))
            write = mock("write_trial_artifacts")
            finalize = mock("finalize_and_save_results")
            mkdir = stack.enter_context(patch.object(Path, "mkdir"))
            available = stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
            count = stack.enter_context(patch.object(torch.cuda, "device_count", return_value=visible_gpus))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield SimpleNamespace(
                config=config, args=args, parallel=parallel, serial=serial, write=write,
                finalize=finalize, load_data=load_data, mkdir=mkdir,
                available=available, count=count,
            )

    def test_unordered_completions_rank_globally_and_preserve_calibration_protocol(self):
        with self._main_setup() as mocks:
            tuning.main()
        mocks.serial.assert_not_called()
        mocks.parallel.assert_called_once()
        tasks, devices, worker = mocks.parallel.call_args.args
        self.assertEqual([task[0] for task in tasks], [1, 2, 3, 4])
        self.assertEqual([task[3] for task in tasks], [2026] * 4)
        self.assertTrue(all(task[1]["tuning"]["num_gpus"] == 2 for task in tasks))
        self.assertEqual(devices, ["cuda:0", "cuda:1"])
        self.assertIs(worker, tuning._run_gpu_trial)
        self.assertIs(mocks.parallel.call_args.kwargs["initializer"], tuning._initialize_gpu_worker)
        self.assertEqual(mocks.parallel.call_args.kwargs["initargs"][1], ["a", "b"])
        self.assertEqual(mocks.write.call_count, 4)
        payload = mocks.finalize.call_args.args[1]
        self.assertEqual([row["trial_index"] for row in payload["all_trials"]], [1, 2, 3, 4])
        self.assertEqual([row["trial_index"] for row in payload["top_trials"]], [2, 3, 1])
        self.assertEqual(payload["num_positive_delta_coverage_trials"], 3)
        self.assertEqual(payload["num_sequences"], 2)
        self.assertEqual(payload["execution"], {
            "mode": "parallel_trials", "num_gpus": 2, "worker_devices": devices,
        })
        self.assertEqual(payload["calibration_split"], "nominal_validation")
        self.assertEqual(payload["evaluation_split"], "nominal_calibration")
        self.assertFalse(payload["tuning_protocol"]["final_test_exposed"])
        self.assertFalse(payload["final_test_evaluated"])

    def test_default_single_worker_keeps_serial_cpu_fallback_and_cli_can_disable_parallel(self):
        for num_gpus in (None, 1):
            with self.subTest(num_gpus=num_gpus), self._main_setup(
                num_gpus=num_gpus, tuning_num_gpus=2 if num_gpus == 1 else 1,
            ) as mocks:
                mocks.available.return_value = False
                tuning.main()
                mocks.parallel.assert_not_called()
                self.assertEqual(mocks.serial.call_count, 4)
                mocks.available.assert_not_called()
                mocks.count.assert_not_called()
                self.assertEqual(mocks.finalize.call_args.args[1]["execution"], {
                    "mode": "serial", "num_gpus": 1, "worker_devices": [],
                })

    def test_insufficient_gpus_fail_before_loading_data_or_writing_outputs(self):
        with self._main_setup(visible_gpus=1) as mocks:
            with self.assertRaises(ValueError):
                tuning.main()
        mocks.load_data.assert_not_called()
        mocks.parallel.assert_not_called()
        mocks.mkdir.assert_not_called()
        mocks.write.assert_not_called()
        mocks.finalize.assert_not_called()

    def test_worker_failure_never_publishes_incomplete_final_ranking(self):
        with self._main_setup() as mocks:
            def incomplete_results():
                yield _fake_records(mocks.config)[0]
                raise RuntimeError("worker failed after one result")
            mocks.parallel.return_value = incomplete_results()
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                tuning.main()
        mocks.finalize.assert_not_called()

    def test_artifact_failure_closes_pending_parallel_results(self):
        closed = []
        with self._main_setup() as mocks:
            def results():
                try:
                    yield from _fake_records(mocks.config)
                finally:
                    closed.append(True)
            mocks.parallel.return_value = results()
            mocks.write.side_effect = OSError("cannot save trial")
            with self.assertRaisesRegex(OSError, "cannot save trial"):
                tuning.main()
        self.assertEqual(closed, [True])
        mocks.finalize.assert_not_called()

    def test_gpu_execution_device_does_not_make_saved_config_device_specific(self):
        config = _small_config()
        config.device = 0
        data, keys, cache = {"first": {}}, ["first"], {}
        observed_devices = []

        def run_grid(index, trial_config, values, selected, sequence_keys, seed, prepared):
            observed_devices.append(trial_config.device)
            self.assertIs(selected, data)
            self.assertIs(sequence_keys, keys)
            self.assertIs(prepared, cache)
            self.assertEqual((index, values, seed), (3, {"x": 1}, 2026))
            return {"resolved_config": tuning.plain_config(trial_config)}

        with (
            patch.object(tuning, "_worker_device", "cuda:1"),
            patch.object(tuning, "_worker_selected_data", data),
            patch.object(tuning, "_worker_sequence_keys", keys),
            patch.object(tuning, "_worker_prepared_data_cache", cache),
            patch.object(tuning, "_run_grid_trial", side_effect=run_grid),
        ):
            record = tuning._run_gpu_trial((3, tuning.plain_config(config), {"x": 1}, 2026))
        self.assertEqual(observed_devices, ["cuda:1"])
        self.assertEqual(record["resolved_config"]["device"], 0)
        self.assertEqual(record["worker_device"], "cuda:1")

    def test_real_spawn_trials_match_serial_with_both_encoders_and_calibration_modes(self):
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            data, keys, tasks = _artifact(), ["first", "second"], []
            for index, (encoder, rolling) in enumerate(
                (("rnn", True), ("transformer", True), ("rnn", False), ("transformer", False)), 1
            ):
                config = _small_config(encoder, rolling)
                tasks.append((index, tuning.plain_config(config),
                              {"model.rolling_calibration": rolling}, 2026))
            serial_cache = {}
            with contextlib.redirect_stdout(io.StringIO()):
                expected = [
                    tuning._run_grid_trial(index, OmegaConf.create(config), values,
                                           data, keys, seed, serial_cache)
                    for index, config, values, seed in tasks
                ]
                actual = list(pool.iter_parallel_trials(
                    list(reversed(tasks)), ["cpu-worker-a", "cpu-worker-b"], _run_cpu_trial,
                    initializer=_initialize_cpu_worker, initargs=(data, keys),
                ))
            self.assertEqual(sorted(actual, key=lambda row: row["trial_index"]), expected)
            self.assertEqual(len(serial_cache), 1)
            for record in actual:
                self.assertEqual(record["sequence_keys"], keys)
                self.assertFalse(record["final_test_evaluated"])
                sequences = record["result"]["sequence_results"]
                self.assertEqual(list(sequences), keys)
                for result in sequences.values():
                    self.assertEqual(result["sample_counts"], {
                        "train": 43, "model_selection_valid": 12, "calibration": 10,
                        "initial_calibration_pool": 4, "tuning_evaluation": 10,
                    })
                    self.assertEqual(result["rolling_calibration"], record["grid_values"]["model.rolling_calibration"])
                    self.assertEqual(result["calibration_split"], "nominal_validation")
                    self.assertEqual(result["evaluation_split"], "nominal_calibration")
                    self.assertFalse(result["final_test_evaluated"])
            changed = copy.deepcopy(data)
            for sequence in changed.values():
                for values in sequence.values():
                    values[80:] += 10_000
            for task, baseline in zip(tasks, expected):
                index, config, values, seed = task
                with contextlib.redirect_stdout(io.StringIO()):
                    changed_result = tuning._run_grid_trial(
                        index, OmegaConf.create(config), values, changed, keys, seed, {}
                    )
                self.assertEqual(changed_result, baseline)
        finally:
            torch.set_num_threads(original_threads)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware is not available")
    def test_real_cuda_worker_executes_complete_rnn_and_transformer_trials(self):
        data, keys, tasks = _artifact(), ["first", "second"], []
        for index, encoder in enumerate(("rnn", "transformer"), 1):
            config = _small_config(encoder, rolling=(index == 1))
            config.device = 0
            tasks.append((index, tuning.plain_config(config), {}, 2026))
        records = list(pool.iter_parallel_trials(
            tasks, ["cuda:0"], tuning._run_gpu_trial,
            initializer=tuning._initialize_gpu_worker, initargs=(data, keys),
        ))
        self.assertEqual([record["trial_index"] for record in records], [1, 2])
        for record in records:
            self.assertEqual(record["worker_device"], "cuda:0")
            self.assertEqual(record["resolved_config"]["device"], 0)
            self.assertEqual(list(record["result"]["sequence_results"]), keys)
            self.assertEqual(record["result"]["num_sequences_evaluated"], 2)
            self.assertTrue(np.isfinite(record["result"]["selection_score"]))
            self.assertFalse(record["final_test_evaluated"])


if __name__ == "__main__":
    unittest.main()
