"""End-to-end checks for building reusable shared-group artifacts."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from conditional_coverage.artifacts import load_group_artifact
from conditional_coverage.config import file_sha256, validate_config
from conditional_coverage.prepare_groups import build_shared_groups, main


class ConditionalCoveragePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source_path = self.directory / "observed.pkl"
        index = np.arange(101, dtype=np.float64)
        self.data = {"station": {
            "heldout_x": np.column_stack((np.cos(index / 11), index / 101)),
            "heldout_y": np.sin(index / 5) + index / 100,
            # Neither structural validation nor encoder fitting may inspect this.
            "heldout_predictions": {"deliberately": "not a forecast array"},
        }}
        self.write_source(self.source_path, self.data)
        self.qr_path = self.directory / "qr.yaml"
        self.split_path = self.directory / "split.yaml"
        self.qr = {"data": {"data_path": str(self.source_path),
                            "train_ratio": .5, "valid_ratio": .16},
                   "model": {"prediction_step": 1, "target_quantiles": [[.05, .95]]}}
        self.split = {"data": {"data_path": str(self.source_path),
                               "calibration_ratio": .66, "test_ratio": .34},
                      "model": {"prediction_step": 1}}
        self.write_yaml(self.qr_path, self.qr)
        self.write_yaml(self.split_path, self.split)
        self.config = {
            "path_base": str(self.directory),
            "data_path": "observed.pkl", "output_dir": "groups", "seed": 7,
            "device": "cpu", "num_threads": 1,
            "model": {"window_size": 4, "dim_model": 6, "num_layers": 1},
            "training": {"epochs": 2, "batch_size": 32, "early_stop": 2},
            "clustering": {"k": 5, "n_init": 2},
            "comparisons": [
                {"name": "qr", "method": "qr_cp", "config_path": "qr.yaml"},
                {"name": "split", "method": "split_cp", "config_path": "split.yaml"},
            ],
        }

    @staticmethod
    def write_source(path, data):
        with Path(path).open("wb") as stream:
            pickle.dump(data, stream)

    @staticmethod
    def write_yaml(path, data):
        with Path(path).open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream)

    def build(self, config=None, dry_run=False):
        return build_shared_groups(self.config if config is None else config,
                                   dry_run=dry_run, progress=lambda message: None)

    def checkpoint(self, path, manifest):
        reference = manifest["sequences"][0]["checkpoint"]
        return torch.load(path / reference, map_location="cpu", weights_only=True)

    def configure_logged_qr(self):
        run_dir = self.directory / "qr_run"
        run_dir.mkdir()
        self.write_yaml(run_dir / "resolved_config.yaml", self.qr)
        complete = np.arange(67, 101)
        valid = np.ones(len(complete), dtype=bool)
        valid[[0, 7, 33]] = False
        self.qr_log = {"station": {
            "metadata": {"split": "test", "target_indices": complete.tolist(),
                         "valid_prediction_mask": valid.tolist(), "total_points": len(complete)},
            "evaluation_results": {(.05, .95): {
                "target_indices": complete[valid].tolist(),
                "target_y": self.data["station"]["heldout_y"][complete[valid]].tolist(),
                "coverage": [True] * int(valid.sum()),
            }},
        }}
        self.write_source(run_dir / "log.pkl", self.qr_log)
        self.config["comparisons"][0] = {"name": "qr", "method": "qr_cp", "run_dir": "qr_run"}
        return run_dir

    def test_build_save_reload_has_shared_groups_and_independent_median_checkpoint(self):
        originals = {path: file_sha256(path) for path in (self.source_path, self.qr_path, self.split_path)}
        path = self.build()
        self.assertEqual(path, self.directory / "groups" / "k5")
        manifest, sequences = load_group_artifact(path)
        record = sequences["station"]
        np.testing.assert_array_equal(record["calibration_indices"], np.arange(66))
        np.testing.assert_array_equal(record["test_indices"], np.arange(67, 101))
        np.testing.assert_array_equal(record["excluded_boundary_indices"], [66])
        np.testing.assert_array_equal(record["calibration_target_indices"], np.arange(4, 66))
        self.assertEqual(set(record["groups"]), {5})
        self.assertEqual(manifest["k"], 5)
        for k, group in record["groups"].items():
            self.assertEqual(group["test_labels"].shape, (34,))
            self.assertEqual(int(group["test_counts"].sum()), 34)
            self.assertEqual(int(group["calibration_counts"].sum()), 62)
            self.assertEqual(group["centroids"].shape, (k, 6))
            distances = ((record["test_embeddings"][:, None, :] - group["centroids"][None, :, :]) ** 2).sum(axis=2)
            np.testing.assert_array_equal(group["test_labels"], distances.argmin(axis=1))
        np.testing.assert_allclose(np.linalg.norm(record["test_embeddings"], axis=1), 1, atol=1e-6)
        checkpoint = self.checkpoint(path, manifest)
        self.assertEqual(checkpoint["target"], "observed_y")
        self.assertEqual(checkpoint["training_quantiles"], [0.5])
        self.assertEqual(checkpoint["model_state_dict"]["training_quantiles"].tolist(), [0.5])
        self.assertEqual(checkpoint["model_state_dict"]["output_linear.weight"].shape, (1, 6))
        self.assertEqual(manifest["source"]["sha256"], originals[self.source_path])
        self.assertEqual(manifest["test_policy"], "intersection_of_original_test_indices_before_prediction_exclusions")
        for original, digest in originals.items():
            self.assertEqual(file_sha256(original), digest)
        saved_config = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(saved_config["output_dir"], str(self.directory / "groups"))
        with patch("conditional_coverage.encoder.train_encoder") as train:
            with self.assertRaises(FileExistsError):
                self.build(saved_config)
            train.assert_not_called()
        self.assertFalse((path / "k5").exists())
        repeated_path = self.build(dict(self.config, output_dir="groups_repeated"))
        repeated_manifest, repeated_sequences = load_group_artifact(repeated_path)
        self.assertEqual(manifest["grouping_id"], repeated_manifest["grouping_id"])
        for k in record["groups"]:
            np.testing.assert_array_equal(record["groups"][k]["test_labels"],
                                          repeated_sequences["station"]["groups"][k]["test_labels"])

    def test_forecasts_and_last_test_outcome_cannot_change_encoder_or_groups(self):
        first_path = self.build()
        first_manifest, first_sequences = load_group_artifact(first_path)
        first_checkpoint = self.checkpoint(first_path, first_manifest)
        self.data["station"]["heldout_predictions"] = np.full(101, np.nan)
        self.data["station"]["heldout_y"][-1] = np.nan
        self.write_source(self.source_path, self.data)
        config = dict(self.config, output_dir="groups_changed")
        second_path = self.build(config)
        second_manifest, second_sequences = load_group_artifact(second_path)
        second_checkpoint = self.checkpoint(second_path, second_manifest)
        for name, weights in first_checkpoint["model_state_dict"].items():
            torch.testing.assert_close(weights, second_checkpoint["model_state_dict"][name], rtol=0, atol=0)
        for field in ("calibration_embeddings", "test_embeddings"):
            np.testing.assert_array_equal(first_sequences["station"][field], second_sequences["station"][field])
        for k in first_sequences["station"]["groups"]:
            for field in ("test_labels", "calibration_labels", "centroids"):
                np.testing.assert_array_equal(first_sequences["station"]["groups"][k][field],
                                              second_sequences["station"]["groups"][k][field])

    def test_dry_run_audits_inputs_without_creating_any_files(self):
        before = set(self.directory.rglob("*"))
        result = self.build(dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["output_dir"], str(self.directory / "groups" / "k5"))
        self.assertEqual(result["config"]["output_dir"], str(self.directory / "groups"))
        self.assertEqual(result["sequences"][0]["calibration_end"], 66)
        self.assertEqual(result["sequences"][0]["test_start"], 67)
        self.assertEqual(result["sequences"][0]["boundary_gap_count"], 1)
        self.assertEqual(before, set(self.directory.rglob("*")))

    def test_cli_output_override_and_relative_configuration_paths(self):
        config_path = self.directory / "grouping.yaml"
        config = copy.deepcopy(self.config)
        config["path_base"] = "."
        self.write_yaml(config_path, config)
        destination = self.directory / "cli_output"
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--config", str(config_path), "--output-dir", str(destination), "--dry-run"])
        self.assertEqual(result["output_dir"], str(destination / "k5"))
        self.assertEqual(json.loads(output.getvalue())["output_dir"], str(destination / "k5"))
        self.assertEqual(result["config"]["output_dir"], str(destination))
        self.assertFalse(destination.exists())
        self.assertFalse((self.directory / "groups").exists())

    def test_different_source_timeline_is_rejected_before_output(self):
        alternate = self.directory / "other_source.pkl"
        changed = copy.deepcopy(self.data)
        changed["station"]["heldout_y"][10] += 100
        self.write_source(alternate, changed)
        self.split["data"]["data_path"] = str(alternate)
        self.write_yaml(self.split_path, self.split)
        with self.assertRaisesRegex(ValueError, "cannot mix timelines"):
            self.build(dry_run=True)
        self.assertFalse((self.directory / "groups").exists())

    def test_invalid_k_and_unknown_clustering_or_model_settings_are_rejected(self):
        for changes in ({"clustering": {"k": [3, 5]}},
                        {"clustering": {"k": 0}},
                        {"clustering": {"k": -1}},
                        {"clustering": {"k": 2.5}},
                        {"clustering": {"k": "5"}},
                        {"clustering": {"k": True}},
                        {"clustering": {"k": 3, "metric": "cosine"}},
                        {"model": {"target": "residual"}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config({**self.config, **changes})

    def test_k_defaults_to_five_and_old_list_setting_has_migration_message(self):
        config = copy.deepcopy(self.config)
        del config["clustering"]
        self.assertEqual(validate_config(config)["clustering"], {"k": 5, "n_init": 10})
        for clustering in ({"ks": [3, 5, 10]}, {"k": 5, "ks": [5]}):
            config["clustering"] = clustering
            with self.subTest(clustering=clustering), self.assertRaisesRegex(ValueError, "set clustering.k"):
                validate_config(config)

    def test_custom_k_saves_only_the_selected_partition(self):
        config = copy.deepcopy(self.config)
        config["clustering"]["k"] = 7
        path = self.build(config)
        self.assertEqual(path, self.directory / "groups" / "k7")
        manifest, sequences = load_group_artifact(path)
        self.assertEqual(manifest["k"], 7)
        self.assertEqual(set(sequences["station"]["groups"]), {7})
        group = sequences["station"]["groups"][7]
        self.assertEqual(group["centroids"].shape, (7, 6))
        self.assertEqual(group["test_counts"].shape, (7,))
        self.assertEqual(int(group["test_counts"].sum()), 34)
        saved_config = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(saved_config["clustering"]["k"], 7)

    def test_different_k_artifacts_coexist_without_overwriting_existing_base_files(self):
        base = self.directory / "groups"
        base.mkdir()
        legacy_manifest = base / "manifest.json"
        legacy_contents = b'{"legacy": "existing flat artifact"}\n'
        legacy_manifest.write_bytes(legacy_contents)
        first_path = self.build()
        originals = {path: file_sha256(path) for path in first_path.rglob("*") if path.is_file()}

        changed = copy.deepcopy(self.config)
        changed["clustering"]["k"] = 3
        second_path = self.build(changed)
        self.assertEqual(first_path, base / "k5")
        self.assertEqual(second_path, base / "k3")
        self.assertEqual(load_group_artifact(first_path)[0]["k"], 5)
        self.assertEqual(load_group_artifact(second_path)[0]["k"], 3)
        self.assertEqual(legacy_manifest.read_bytes(), legacy_contents)
        self.assertEqual(originals, {path: file_sha256(path) for path in originals})
        with patch("conditional_coverage.encoder.train_encoder") as train:
            with self.assertRaises(FileExistsError):
                self.build(changed)
            train.assert_not_called()

    def test_k_larger_than_available_calibration_contexts_fails_before_training(self):
        config = copy.deepcopy(self.config)
        config["clustering"]["k"] = 63  # Only 62 eligible calibration contexts.
        with self.assertRaisesRegex(ValueError, "fewer calibration contexts"):
            self.build(config)
        self.assertFalse((self.directory / "groups").exists())

    def test_run_logs_preserve_original_test_intersection_after_exclusions(self):
        self.configure_logged_qr()
        result = self.build(dry_run=True)
        sequence = result["sequences"][0]
        self.assertEqual(sequence["test_contexts"], 34)
        self.assertEqual(sequence["calibration_end"], 66)
        self.assertEqual(sequence["test_start"], 67)

    def test_run_log_targets_must_match_the_selected_source_timeline(self):
        run_dir = self.configure_logged_qr()
        self.qr_log["station"]["evaluation_results"][(.05, .95)]["target_y"][0] += 100
        self.write_source(run_dir / "log.pkl", self.qr_log)
        with self.assertRaisesRegex(ValueError, "logged targets disagree"):
            self.build(dry_run=True)

    def test_run_log_full_timeline_must_match_the_configuration(self):
        run_dir = self.configure_logged_qr()
        self.qr_log["station"]["metadata"]["target_indices"][0] = 66
        self.write_source(run_dir / "log.pkl", self.qr_log)
        with self.assertRaisesRegex(ValueError, "original logged test indices"):
            self.build(dry_run=True)


if __name__ == "__main__":
    unittest.main()
