"""CPU sequence workers preserve Local-CP outputs across process counts."""

import contextlib
import io
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dscp import run_local_cp
from dscp.local_cp_parallel import run_cpu_local_cp


class LocalCPCPUParallelTests(unittest.TestCase):
    @staticmethod
    def artifact():
        values = np.linspace(-1.0, 1.0, 24, dtype=np.float32)
        return {
            name: {
                "heldout_x": np.column_stack([values, values**2]),
                "heldout_y": np.sin(values + index * 0.1),
                "heldout_predictions": values * (0.1 + index * 0.05),
            }
            for index, name in enumerate(("series_b", "series_a", "series_c"))
        }

    @staticmethod
    def configuration(artifact_path, output_dir, encoder):
        return OmegaConf.create({
            "device": "cpu",
            "seed": 47,
            "saving_dir": str(output_dir),
            "data": {
                "data_path": str(artifact_path),
                "train_ratio": 0.5,
                "valid_ratio": 0.125,
                "calibration_ratio": 0.125,
                "test_ratio": 0.25,
                "normalize": False,
                "strided_features": "xr",
            },
            "model": {
                "rnn_type": "lstm",
                "dim_model": 4,
                "num_heads": 2,
                "num_layers": 1,
                "dropout": 0.0,
                "use_current_feature": False,
                "training_quantiles": [0.2, 0.5, 0.8],
                "target_quantiles": [[0.1, 0.9]],
                "prediction_step": 1,
                "window_size": 3,
                "calibration_size": 3,
                "rolling_calibration": True,
                "kernel": "epanechnikov",
                "bandwidth": 1.0,
                "similarity_fn": "cos_similarity",
                "temperature": 10.0,
                "sampling_num": 16,
            },
            "training": {
                "batch_size": 64,
                "learning_rate": 0.001,
                "epochs": 1,
                "early_stop": 1,
            },
            "plotting": {"plotting": False},
        })

    def assert_nested_equal(self, first, second):
        if isinstance(first, dict):
            self.assertEqual(list(first), list(second))
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for first_item, second_item in zip(first, second):
                self.assert_nested_equal(first_item, second_item)
        elif isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, (np.ndarray, np.generic)):
            np.testing.assert_array_equal(first, second)
        else:
            self.assertEqual(first, second)

    def test_real_workers_preserve_every_sequence_and_reproducible_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path = root / "lr_air_data.pkl"
            artifact = self.artifact()
            with artifact_path.open("wb") as stream:
                pickle.dump(artifact, stream)

            for encoder in ("rnn", "transformer"):
                outputs = []
                for workers in (1, 2):
                    with self.subTest(encoder=encoder, workers=workers):
                        output_dir = root / f"{encoder}_{workers}"
                        config = self.configuration(artifact_path, output_dir, encoder)
                        config_path = root / f"{encoder}_{workers}.yaml"
                        OmegaConf.save(config, config_path)
                        with contextlib.redirect_stdout(io.StringIO()):
                            run_cpu_local_cp(str(config_path), encoder=encoder, workers=workers)

                        with (output_dir / "log.pkl").open("rb") as stream:
                            log = pickle.load(stream)
                        self.assertEqual(list(log), list(artifact))
                        with (output_dir / "summary_results.pkl").open("rb") as stream:
                            summary = pickle.load(stream)
                        self.assertEqual(set(summary), {(0.1, 0.9)})
                        models = {}
                        for index, key in enumerate(artifact):
                            models[key] = torch.load(
                                output_dir / f"{key}_model.pt",
                                map_location="cpu", weights_only=True,
                            )
                            result = log[key]["evaluation_results"][(0.1, 0.9)]
                            self.assertEqual(result["target_indices"], list(range(18, 24)))
                            np.testing.assert_array_equal(
                                result["target_y"], artifact[key]["heldout_y"][18:],
                            )
                            self.assertEqual(len(log[key]["train_loss"]), 1)
                            self.assertTrue(log[key]["rolling_calibration"])
                            self.assertEqual(log[key]["sequence_seed"], 47 + index)
                        self.assertEqual(
                            {path.name for path in output_dir.glob("*_model.pt")},
                            {f"{key}_model.pt" for key in artifact},
                        )
                        outputs.append((log, summary, models))
                self.assert_nested_equal(outputs[0], outputs[1])

    def test_sequence_subset_skips_artifact_reload_and_shared_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = {"series_a": self.artifact()["series_a"]}
            for encoder in ("rnn", "transformer"):
                with self.subTest(encoder=encoder):
                    output_dir = root / encoder
                    config_path = root / f"{encoder}.yaml"
                    OmegaConf.save(
                        self.configuration(root / "missing_air_data.pkl", output_dir, encoder),
                        config_path,
                    )
                    runner = getattr(run_local_cp, f"run_{encoder}_local_cp")
                    with (
                        patch.object(run_local_cp, "load_data") as load_data,
                        patch.object(run_local_cp, "save_data") as save_data,
                        patch.object(run_local_cp, "_save_local_cp_summary") as summary,
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        log = runner(
                            str(config_path), sequence_data=sequence, save_aggregate=False,
                        )
                    load_data.assert_not_called()
                    save_data.assert_not_called()
                    summary.assert_not_called()
                    self.assertEqual(list(log), ["series_a"])
                    self.assertEqual(
                        {path.name for path in output_dir.iterdir()}, {"series_a_model.pt"},
                    )

    def test_invalid_worker_counts_and_non_cpu_device_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.configuration(root / "missing_air_data.pkl", root / "output", "rnn")
            config_path = root / "config.yaml"
            OmegaConf.save(config, config_path)
            for workers in (0, -1, True, 1.5):
                with self.subTest(workers=workers), self.assertRaises(ValueError):
                    run_cpu_local_cp(str(config_path), encoder="rnn", workers=workers)
            config.device = 0
            OmegaConf.save(config, config_path)
            with self.assertRaisesRegex(ValueError, "cpu"):
                run_cpu_local_cp(str(config_path), encoder="rnn", workers=1)
            self.assertFalse((root / "output").exists())

    def test_worker_failure_is_propagated_without_success_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path = root / "lr_air_data.pkl"
            artifact = self.artifact()
            del artifact["series_a"]["heldout_predictions"]
            with artifact_path.open("wb") as stream:
                pickle.dump(artifact, stream)
            output_dir = root / "output"
            config_path = root / "config.yaml"
            OmegaConf.save(self.configuration(artifact_path, output_dir, "rnn"), config_path)
            with (
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "sequence 'series_a' failed"),
            ):
                run_cpu_local_cp(str(config_path), encoder="rnn", workers=2)
            self.assertFalse((output_dir / "summary_results.pkl").exists())


if __name__ == "__main__":
    unittest.main()
