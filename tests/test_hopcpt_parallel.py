"""Exercise GPU dispatch without requiring CUDA or the training dependencies."""

import contextlib
import importlib.util
import io
import random
import sys
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _device(value):
    if value == "cpu":
        return SimpleNamespace(type="cpu", index=None)
    if value == "cuda":
        return SimpleNamespace(type="cuda", index=None)
    if value.startswith("cuda:") and value[5:].isdigit():
        return SimpleNamespace(type="cuda", index=int(value[5:]))
    raise RuntimeError("Invalid device")


def _load_runner():
    attributes = {
        "torch": ["manual_seed", "set_num_threads"],
        "numpy": [],
        "tqdm": ["tqdm"],
        "baselines.hopcpt.model": ["HopfieldNet"],
        "baselines.hopcpt.loss": ["compute_hopfield_net_loss"],
        "dscp.data": ["ConformalPredictionData", "initialize_valid_dataloader", "initialize_test_dataloader"],
        "utils.utils": ["load_data", "save_data", "read_setup", "generate_feature_hopcpt_training",
                        "generate_feature_hopcpt_test", "estimate_hopcpt_residual_interval"],
        "utils.reporting": ["compute_coverage", "compute_interval_width", "compute_winkler_score",
                            "summarize_evaluation_results"],
        "utils.plotting": ["plot_cp_prediction_intervals"],
    }
    modules = {}
    for name, names in attributes.items():
        module = ModuleType(name)
        for attribute in names:
            setattr(module, attribute, Mock())
        modules[name] = module
    modules["torch"].device = _device
    modules["torch"].cuda = SimpleNamespace(
        device_count=Mock(return_value=2), is_available=Mock(return_value=True), set_device=Mock())
    modules["numpy"].random = SimpleNamespace(seed=Mock())
    modules["tqdm"].tqdm = lambda items, **kwargs: items
    spec = importlib.util.spec_from_file_location(
        "_hopcpt_parallel_test_runner", REPO_ROOT / "baselines" / "hopcpt" / "run_hopcpt.py")
    runner = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(runner)
    return runner


class HopCPTParallelTests(unittest.TestCase):
    def setUp(self):
        self.runner = _load_runner()

    def test_disabled_and_explicit_cpu_execution_do_not_query_cuda(self):
        for config, expected in (({}, None), ({"parallel": {"enabled": True, "devices": ["cpu"]}}, ["cpu"])):
            with self.subTest(config=config):
                self.assertEqual(self.runner._parallel_devices(OmegaConf.create(config)), expected)
        self.runner.torch.cuda.device_count.assert_not_called()
        self.runner.torch.cuda.is_available.assert_not_called()

    def test_visible_cuda_indices_are_validated_before_workers_start(self):
        config = OmegaConf.create({"parallel": {"enabled": True, "devices": ["cuda", "1"]}})
        self.assertEqual(self.runner._parallel_devices(config), ["cuda:0", "cuda:1"])
        for devices in ([], [0, "cuda:0"], [0, 2], [True], [1.5], "cuda:0", 2):
            with self.subTest(devices=devices):
                config.parallel.devices = devices
                with self.assertRaises(ValueError):
                    self.runner._parallel_devices(config)
        config.parallel.devices = [0, 1]
        self.runner.torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(ValueError, "visible"):
            self.runner._parallel_devices(config)

    def test_gpu_count_must_be_positive_integer_with_available_devices(self):
        config = OmegaConf.create({"parallel": {"enabled": True, "num_gpus": 2}})
        self.assertEqual(self.runner._parallel_devices(config), ["cuda:0", "cuda:1"])
        for count in (False, 0, -1, 1.5, "2", 3):
            with self.subTest(count=count):
                config.parallel.num_gpus = count
                with self.assertRaises(ValueError):
                    self.runner._parallel_devices(config)

    def test_workers_reseed_each_process_and_apply_thread_budget(self):
        runner = self.runner
        config = {"seed": 17, "parallel": {"threads_per_worker": 2}}
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        with patch.object(runner, "_run_hopcpt_sequence", side_effect=lambda key, *args: (key, random.random())):
            first = runner._run_hopcpt_sequence_chunk([("a", {}), ("b", {})], config, "cpu")
            random.seed(999)
            second = runner._run_hopcpt_sequence_chunk([("a", {}), ("b", {})], config, "cpu")
            self.assertEqual(first, second)
            self.assertNotEqual(first["a"], first["b"])
            runner._run_hopcpt_sequence_chunk([], config, "cuda:1")
        self.assertEqual([call.args[0] for call in runner.torch.manual_seed.call_args_list], [17, 17, 18])
        self.assertEqual([call.args[0] for call in runner.np.random.seed.call_args_list], [17, 17, 18])
        runner.torch.set_num_threads.assert_called_with(2)
        self.assertEqual(runner.torch.cuda.set_device.call_args.args[0].index, 1)

    def test_invalid_worker_thread_budget_fails_before_training(self):
        for count in (True, 0, -1, 1.5):
            with self.subTest(count=count), patch.object(self.runner, "_run_hopcpt_sequence") as train:
                with self.assertRaisesRegex(ValueError, "threads_per_worker"):
                    self.runner._run_hopcpt_sequence_chunk([("a", {})], {
                        "parallel": {"threads_per_worker": count}}, "cpu")
                train.assert_not_called()

    def test_spawn_dispatch_assigns_every_sequence_once_and_merges_worker_logs(self):
        runner = self.runner
        config = OmegaConf.create({
            "device": 0, "seed": 17, "saving_dir": "unused_hopcpt_results",
            "parallel": {"enabled": True, "devices": [0, 1], "threads_per_worker": 2},
            "data": {"data_path": "lr_solar_data.pkl", "train_ratio": 0.33,
                     "valid_ratio": 0.33, "normalize": True},
            "model": {"prediction_step": 1, "y_lags": 1, "target_quantiles": [[0.05, 0.95]]},
        })
        data = {key: {"value": index} for index, key in enumerate("abcde")}
        runner.ConformalPredictionData.return_value.data = data
        runner.read_setup.return_value = ("lr", "solar")
        runner.summarize_evaluation_results.return_value = {}
        assignments = []

        def submit(worker, items, payload, device):
            self.assertIs(worker, runner._run_hopcpt_sequence_chunk)
            self.assertEqual(payload["seed"], 17)
            self.assertEqual(payload["parallel"]["threads_per_worker"], 2)
            assignments.append((device, [key for key, _ in items]))
            future = Future()
            future.set_result({key: {"device": device, **value} for key, value in items})
            return future

        with (
            patch.object(runner.OmegaConf, "load", return_value=config),
            patch.object(runner.os, "makedirs"),
            patch.object(runner, "ProcessPoolExecutor") as pool,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            pool.return_value.__enter__.return_value.submit.side_effect = submit
            runner.run_hopcpt("unused.yaml")

        self.assertEqual(pool.call_args.kwargs["max_workers"], 2)
        self.assertEqual(pool.call_args.kwargs["mp_context"].get_start_method(), "spawn")
        self.assertEqual(assignments, [("cuda:0", ["a", "c", "e"]), ("cuda:1", ["b", "d"])])
        merged = runner.summarize_evaluation_results.call_args.args[0]
        self.assertEqual(set(merged), set(data))
        self.assertEqual(merged["e"], {"device": "cuda:0", "value": 4})
        self.assertEqual(merged["d"], {"device": "cuda:1", "value": 3})


if __name__ == "__main__":
    unittest.main()
