"""CPU workers preserve per-sequence grouping results and artifact atomicity."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import copy
import io
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from conditional_coverage import prepare_groups
from conditional_coverage.artifacts import load_group_artifact
from conditional_coverage.config import validate_config


def _runtime_thread_counts():
    """Inspect actual thread settings inside a fresh, initialized CPU worker."""
    from threadpoolctl import threadpool_info
    return {
        "torch": torch.get_num_threads(),
        "torch_interop": torch.get_num_interop_threads(),
        "native": [library["num_threads"] for library in threadpool_info()],
    }


class ConditionalCoverageParallelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source_path = self.directory / "source.pkl"
        t = np.arange(64, dtype=np.float64)
        self.data = {
            key: {
                "heldout_x": np.column_stack((np.cos(t / 7 + phase), t / 64)),
                "heldout_y": np.sin(t / 5 + phase) + t / 100,
                "heldout_predictions": {"unused": "by grouping workers"},
            }
            for key, phase in (("7", 0.0), (7, 0.5))
        }
        self.write_source()
        method = {
            "data": {"data_path": str(self.source_path), "train_ratio": .5, "valid_ratio": .16},
            "model": {"prediction_step": 1},
        }
        (self.directory / "qr.yaml").write_text(yaml.safe_dump(method), encoding="utf-8")
        self.config = {
            "path_base": str(self.directory), "data_path": "source.pkl", "output_dir": "groups",
            "device": "cpu", "num_threads": 1, "seed": 17,
            "model": {"window_size": 4, "dim_model": 4, "num_layers": 1},
            "training": {"epochs": 2, "batch_size": 32, "early_stop": 2},
            "clustering": {"k": 3, "n_init": 1},
            "comparisons": [{"name": "qr", "method": "qr_cp", "config_path": "qr.yaml"}],
        }
        # The test runner may itself be launched from a Slurm allocation.
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("SLURM_CPUS_PER_TASK", None)

    def write_source(self):
        with self.source_path.open("wb") as stream:
            pickle.dump(self.data, stream)

    def build(self, **changes):
        return prepare_groups.build_shared_groups(
            {**self.config, **changes}, progress=lambda _: None,
        )

    @staticmethod
    def checkpoints(path, manifest):
        return {
            reference["sequence"]["value"]: torch.load(
                path / reference["checkpoint"], map_location="cpu", weights_only=True,
            )
            for reference in manifest["sequences"]
        }

    def test_spawned_workers_match_serial_groups_and_checkpoints_independent_of_order(self):
        serial_path = self.build(output_dir="serial", cpu_workers=1)
        parallel_path = self.build(output_dir="parallel", cpu_workers=2, sequences=[7, "7"])
        serial_manifest, serial_sequences = load_group_artifact(serial_path)
        parallel_manifest, parallel_sequences = load_group_artifact(parallel_path)
        self.assertEqual(list(serial_sequences), ["7", 7])
        self.assertEqual(list(parallel_sequences), [7, "7"])
        self.assertEqual(serial_manifest["grouping_id"], parallel_manifest["grouping_id"])
        self.assertEqual(parallel_manifest["execution"]["cpu_workers"], 2)
        self.assertEqual(parallel_manifest["execution"]["threads_per_worker"], 1)
        self.assertEqual(parallel_manifest["execution"]["backend"], "spawn_processes")
        self.assertEqual(serial_manifest["execution"]["backend"], "serial")
        serial_checkpoints = self.checkpoints(serial_path, serial_manifest)
        parallel_checkpoints = self.checkpoints(parallel_path, parallel_manifest)
        for key in self.data:
            first, second = serial_sequences[key], parallel_sequences[key]
            for field in ("calibration_indices", "test_indices", "fit_indices", "validation_indices",
                          "calibration_embeddings", "test_embeddings"):
                np.testing.assert_array_equal(first[field], second[field])
            for field in ("centroids", "calibration_labels", "test_labels",
                          "calibration_counts", "test_counts"):
                np.testing.assert_array_equal(first["groups"][3][field], second["groups"][3][field])
            for field in ("fit_loss", "validation_loss", "seed", "best_epoch"):
                self.assertEqual(first["training"][field], second["training"][field])
            for name, value in serial_checkpoints[key]["model_state_dict"].items():
                torch.testing.assert_close(value, parallel_checkpoints[key]["model_state_dict"][name],
                                           rtol=0, atol=0)
        self.assertEqual(len(list((parallel_path / "encoders").glob("*.pt"))), 2)
        resolved = yaml.safe_load((parallel_path / "resolved_config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(resolved["cpu_workers"], 2)
        self.assertEqual(resolved["num_threads"], 1)

    def test_dry_run_caps_workers_to_sequences_and_slurm_allocation_without_training(self):
        before = set(self.directory.rglob("*"))
        with patch.object(prepare_groups, "ProcessPoolExecutor") as executor:
            with patch("conditional_coverage.encoder.train_encoder") as training:
                for allocation, expected in ((None, 2), ("1", 1), ("8", 2)):
                    if allocation is None:
                        os.environ.pop("SLURM_CPUS_PER_TASK", None)
                    else:
                        os.environ["SLURM_CPUS_PER_TASK"] = allocation
                    with self.subTest(allocation=allocation):
                        report = prepare_groups.build_shared_groups(
                            {**self.config, "cpu_workers": 12}, dry_run=True, progress=lambda _: None,
                        )
                        self.assertEqual(report["execution"]["requested_cpu_workers"], 12)
                        self.assertEqual(report["execution"]["cpu_workers"], expected)
                        self.assertEqual(report["execution"]["threads_per_worker"], 1)
                executor.assert_not_called()
                training.assert_not_called()
        self.assertEqual(set(self.directory.rglob("*")), before)

    def test_cpu_worker_initializer_limits_torch_and_native_libraries_to_one_thread(self):
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn"),
            initializer=prepare_groups._initialize_cpu_worker,
        ) as executor:
            counts = executor.submit(_runtime_thread_counts).result(timeout=60)
        self.assertEqual(counts["torch"], 1)
        self.assertEqual(counts["torch_interop"], 1)
        self.assertTrue(counts["native"])
        self.assertEqual(set(counts["native"]), {1})

    def test_worker_failure_does_not_publish_a_partial_artifact(self):
        self.data[7]["heldout_x"] = np.zeros((64, 2))
        self.data[7]["heldout_y"] = np.zeros(64)
        self.write_source()
        before = set(self.directory.rglob("*"))
        with self.assertRaisesRegex(ValueError, "distinct calibration"):
            self.build(cpu_workers=2)
        self.assertFalse((self.directory / "groups").exists())
        self.assertEqual(set(self.directory.rglob("*")), before)

    def test_invalid_worker_counts_and_oversubscribed_or_gpu_combinations_are_rejected(self):
        self.assertEqual(validate_config(self.config)["cpu_workers"], 1)
        for value in (0, -1, True, 2.5, "2"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "cpu_workers"):
                validate_config({**self.config, "cpu_workers": value})
        for changes in ({"cpu_workers": 2, "num_threads": 2},
                        {"cpu_workers": 2, "device": "cuda:0"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config({**self.config, **changes})

    def test_cli_cpu_workers_selects_cpu_and_one_thread_before_dry_run(self):
        config = copy.deepcopy(self.config)
        config.update(device="cuda:0", num_threads=3)
        path = self.directory / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        output = io.StringIO()
        with redirect_stdout(output):
            result = prepare_groups.main(["--config", str(path), "--cpu-workers", "2", "--dry-run"])
        self.assertEqual(json.loads(output.getvalue())["execution"]["cpu_workers"], 2)
        self.assertEqual(result["config"]["device"], "cpu")
        self.assertEqual(result["config"]["num_threads"], 1)
        self.assertEqual(result["config"]["cpu_workers"], 2)
        self.assertFalse((self.directory / "groups").exists())
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            prepare_groups.main(["--config", str(path), "--cpu-workers", "2", "--device", "cuda:0", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
