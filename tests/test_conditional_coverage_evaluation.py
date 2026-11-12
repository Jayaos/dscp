"""Evaluate saved explicit and legacy logs without training any models."""

from contextlib import redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import yaml

from conditional_coverage.artifacts import save_group_artifact
from conditional_coverage.config import file_sha256
from conditional_coverage.evaluate import evaluate_saved_runs, main
from conditional_coverage.evaluation_config import load_evaluation_config, validate_evaluation_config


class SavedConditionalCoverageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source_path = self.directory / "source.pkl"
        self.source = {
            key: {"heldout_x": np.arange(15)[:, None],
                  "heldout_y": np.arange(15, dtype=np.float32) + offset}
            for key, offset in (("station", 0), (7, 100))
        }
        self.dump(self.source_path, self.source)
        record = {
            "calibration_indices": np.arange(9),
            "calibration_target_indices": np.arange(2, 9),
            "test_indices": np.arange(10, 15),
            "excluded_boundary_indices": np.array([9]),
            "scaler": {}, "member_boundaries": {}, "training": {},
            "groups": {2: {
                "centroids": np.array([[-1.], [1.]]),
                "calibration_labels": np.array([0, 0, 0, 1, 1, 1, 1]),
                "test_labels": np.array([0, 0, 1, 1, 1]),
                "calibration_counts": np.array([3, 4]), "test_counts": np.array([2, 3]),
                "inertia": 0., "seed": 7, "n_init": 1,
            }},
        }
        self.groups_dir = self.directory / "groups"
        save_group_artifact(
            self.groups_dir,
            {"k": 2, "grouping_id": "fixture", "dataset": "air", "base_predictor": "lstm",
             "horizon": 1, "source": {"path": str(self.source_path),
                                      "sha256": file_sha256(self.source_path)}},
            {key: copy.deepcopy(record) for key in self.source},
            {key: {"model_state_dict": {}} for key in self.source}, {},
        )
        self.config = {
            "path_base": str(self.directory), "groups_dir": "groups", "output_dir": "evaluation",
            "nominal_coverage": .9, "runs": [],
        }
        self.logs = {}

    @staticmethod
    def dump(path, value):
        with Path(path).open("wb") as stream:
            pickle.dump(value, stream)

    @staticmethod
    def yaml(path, value):
        Path(path).write_text(yaml.safe_dump(value), encoding="utf-8")

    def add_run(self, name, method, *, legacy=False, saved_config=True, indices=None):
        directory = self.directory / name
        directory.mkdir()
        start = 9 if method == "local_cp" else 10
        ids = np.arange(start, 15) if indices is None else np.asarray(indices, dtype=np.int64)
        log = {}
        for key, source in self.source.items():
            result = {
                "target_y": source["heldout_y"][ids].tolist(),
                "coverage": (ids % 2 == 0).tolist(),
            }
            if not legacy:
                result["target_indices"] = ids.tolist()
            log[key] = {"evaluation_results": {(.95, .05): result}}
        self.dump(directory / "log.pkl", log)
        if saved_config:
            data = {"data_path": str(self.source_path), "train_ratio": .5, "valid_ratio": .16}
            if method == "local_cp":
                data.update(train_ratio=.4, valid_ratio=.1, calibration_ratio=.16, test_ratio=.34)
            self.yaml(directory / "resolved_config.yaml", {"data": data, "model": {"prediction_step": 1}})
        self.config["runs"].append({"name": name, "method": method, "run_dir": name})
        self.logs[name] = log
        return directory

    def evaluate(self, **kwargs):
        return evaluate_saved_runs(self.config, progress=lambda _: None, **kwargs)

    def test_three_methods_match_groups_and_publish_reports_without_touching_inputs(self):
        self.add_run("qr", "qr_cp", indices=[14, 10, 13, 11, 12])
        self.add_run("iqn", "iqn_cp")
        self.add_run("local", "local_cp")
        inputs = [self.source_path, *self.groups_dir.rglob("*"),
                  *(self.directory / name / "log.pkl" for name in self.logs)]
        fingerprints = {path: file_sha256(path) for path in inputs if path.is_file()}
        report = self.evaluate()
        self.assertEqual(len(report["summary"]), 3)
        self.assertEqual(len(report["sequence_metrics"]), 6)
        self.assertEqual(len(report["group_metrics"]), 12)
        for row in report["summary"]:
            self.assertEqual(row["cace_definition"], "mean_group_undercoverage")
            self.assertEqual(row["sequence_count"], 2)
            self.assertEqual(row["evaluated_count"], 10)
            self.assertEqual(row["missing_count"], 0)
            self.assertAlmostEqual(row["cace"], (.4 + (.9 - 2 / 3)) / 2)
            self.assertAlmostEqual(row["wuc"], .4)
            self.assertEqual(row["outside_shared_count"], 2 if row["method"] == "local_cp" else 0)
        self.assertTrue(all(row["alignment"] == "result_indices" for row in report["alignment"]))
        self.assertEqual(report["policies"]["cace_definition"], "mean_group_undercoverage")
        self.assertTrue(all(row["cace_definition"] == "mean_group_undercoverage"
                            for row in report["sequence_metrics"]))
        output = self.directory / "evaluation"
        self.assertEqual({p.name for p in output.iterdir()}, {
            "summary.csv", "sequence_metrics.csv", "group_metrics.csv", "evaluation.json",
            "resolved_config.yaml",
        })
        loaded = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(loaded, report)
        self.assertEqual(loaded["sequences"], ["station", 7])
        with (output / "group_metrics.csv").open(newline="", encoding="utf-8") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 12)
        with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
            self.assertTrue(all(row["cace_definition"] == "mean_group_undercoverage"
                                for row in csv.DictReader(stream)))
        self.assertEqual(fingerprints, {path: file_sha256(path) for path in fingerprints})
        with self.assertRaises(FileExistsError):
            self.evaluate()

    def test_legacy_chronology_uses_original_method_split_before_shared_crop(self):
        self.add_run("local", "local_cp", legacy=True)
        with self.assertRaisesRegex(ValueError, "allow_chronological_fallback"):
            self.evaluate(dry_run=True)
        self.config["allow_chronological_fallback"] = True
        report = self.evaluate(dry_run=True)
        self.assertEqual(report["alignment"][0]["alignment"], "chronological_config")
        self.assertEqual(report["alignment"][0]["full_test_count"], 6)
        self.assertEqual(report["sequence_metrics"][0]["outside_shared_count"], 1)
        self.assertEqual(report["sequence_metrics"][0]["covered_count"], 3)
        self.assertFalse((self.directory / "evaluation").exists())

    def test_log_only_legacy_runs_support_per_run_fallback(self):
        self.add_run("qr", "qr_cp", legacy=True, saved_config=False)
        self.config["runs"][0]["allow_chronological_fallback"] = True
        before = set(self.directory.rglob("*"))
        report = self.evaluate(dry_run=True)
        self.assertEqual(report["alignment"][0]["alignment"], "chronological_suffix")
        self.assertEqual(report["runs"][0]["source_validation"], "logged_targets")
        self.assertEqual(before, set(self.directory.rglob("*")))

    def test_config_path_supports_old_runs_without_saved_resolved_config(self):
        directory = self.add_run("qr", "qr_cp", legacy=True)
        original_config = directory / "original.yaml"
        (directory / "resolved_config.yaml").rename(original_config)
        self.config["runs"][0].update(config_path="qr/original.yaml", allow_chronological_fallback=True)
        report = self.evaluate(dry_run=True)
        self.assertEqual(report["alignment"][0]["alignment"], "chronological_config")
        self.assertEqual(report["runs"][0]["config_path"], str(original_config))

    def test_sparse_explicit_ids_report_group_missing_counts_without_inventing_original_total(self):
        self.add_run("qr", "qr_cp", saved_config=False, indices=[14, 10])
        report = self.evaluate()
        self.assertIsNone(report["alignment"][0]["full_test_count"])
        self.assertIsNone(report["alignment"][0]["excluded_count"])
        self.assertEqual(report["sequence_metrics"][0]["missing_count"], 3)
        self.assertEqual(report["sequence_metrics"][0]["evaluated_count"], 2)

    def test_masks_preserve_excluded_positions_and_all_excluded_is_null(self):
        directory = self.add_run("qr", "qr_cp")
        log = self.logs["qr"]
        for key, mask in (("station", [False, False, True, False, True]), (7, [False] * 5)):
            ids = np.arange(10, 15)
            log[key]["metadata"] = {"target_indices": ids.tolist(), "valid_prediction_mask": mask}
            log[key]["evaluation_results"][(.95, .05)] = {
                "coverage": [True] * sum(mask),
                "target_y": self.source[key]["heldout_y"][ids[mask]].tolist(),
            }
        self.dump(directory / "log.pkl", log)
        report = self.evaluate()
        first, second = report["sequence_metrics"]
        self.assertEqual(first["evaluated_count"], 2)
        self.assertEqual(first["missing_count"], 3)
        self.assertEqual(first["evaluated_group_count"], 1)
        self.assertEqual(first["cace"], 0)
        self.assertEqual(first["wuc"], 0)
        self.assertIsNone(second["cace"])
        self.assertEqual(second["status"], "no_valid_predictions")
        self.assertEqual(report["summary"][0]["unavailable_sequence_count"], 1)
        self.assertEqual(report["summary"][0]["missing_count"], 8)
        self.assertIsNone(report["group_metrics"][0]["coverage"])
        json.dumps(report, allow_nan=False)

    def test_unknown_sequence_defaults_to_intersection_but_explicit_requests_are_strict(self):
        directory = self.add_run("qr", "qr_cp")
        self.add_run("iqn", "iqn_cp")
        del self.logs["qr"][7]
        self.dump(directory / "log.pkl", self.logs["qr"])
        report = self.evaluate(dry_run=True)
        self.assertEqual(report["sequences"], ["station"])
        self.assertEqual(report["skipped_sequences"], [{"sequence_key": 7, "missing_from": ["qr"]}])
        self.config["sequences"] = [7]
        with self.assertRaisesRegex(ValueError, "missing"):
            self.evaluate(dry_run=True)

    def test_target_mismatch_and_wrong_nominal_level_fail_without_output(self):
        directory = self.add_run("qr", "qr_cp")
        log = self.logs["qr"]
        log["station"]["evaluation_results"][(.95, .05)]["target_y"][0] += 1
        self.dump(directory / "log.pkl", log)
        with self.assertRaisesRegex(ValueError, "target_y disagrees"):
            self.evaluate()
        log["station"]["evaluation_results"][(.95, .05)]["target_y"][0] -= 1
        self.dump(directory / "log.pkl", log)
        self.config["nominal_coverage"] = .8
        with self.assertRaisesRegex(ValueError, "No saved interval"):
            self.evaluate()
        self.assertFalse((self.directory / "evaluation").exists())

    def test_same_source_can_be_relocated_but_different_data_is_rejected(self):
        directory = self.add_run("qr", "qr_cp")
        relocated = self.directory / "moved.pkl"
        self.source_path.rename(relocated)
        self.config["data_path"] = str(relocated)
        report = self.evaluate(dry_run=True)
        self.assertTrue(report["runs"][0]["configured_source_unavailable"])
        changed = copy.deepcopy(self.source)
        changed["station"]["heldout_y"][0] += 1
        self.dump(relocated, changed)
        with self.assertRaisesRegex(ValueError, "source artifact differs"):
            self.evaluate(dry_run=True)

    def test_run_with_different_available_source_or_later_test_boundary_fails(self):
        directory = self.add_run("qr", "qr_cp")
        path = directory / "resolved_config.yaml"
        settings = yaml.safe_load(path.read_text())
        other = self.directory / "other.pkl"
        self.dump(other, {"different": {}})
        settings["data"]["data_path"] = str(other)
        self.yaml(path, settings)
        with self.assertRaisesRegex(ValueError, "method source artifact differs"):
            self.evaluate(dry_run=True)
        settings["data"].update(data_path=str(self.source_path), train_ratio=.7, valid_ratio=.1)
        self.yaml(path, settings)
        with self.assertRaisesRegex(ValueError, "does not cover the shared groups"):
            self.evaluate(dry_run=True)

    def test_cli_resolves_paths_and_enables_fallback_without_writing_on_dry_run(self):
        self.add_run("qr", "qr_cp", legacy=True, saved_config=False)
        config = copy.deepcopy(self.config)
        config["path_base"] = "."
        path = self.directory / "evaluate.yaml"
        self.yaml(path, config)
        output = io.StringIO()
        destination = self.directory / "cli_report"
        with redirect_stdout(output):
            report = main(["--config", str(path), "--allow-chronological-fallback",
                           "--output-dir", str(destination), "--dry-run"])
        self.assertEqual(json.loads(output.getvalue()), report)
        self.assertEqual(report["output_dir"], str(destination))
        self.assertFalse(destination.exists())

    def test_config_rejects_unsupported_methods_and_invalid_options(self):
        self.add_run("qr", "qr_cp")
        for field, value in (("nominal_coverage", 90), ("nominal_coverage", True),
                             ("allow_chronological_fallback", "true"), ("runs", []),
                             ("unknown", 1), ("sequences", [True])):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_evaluation_config({**self.config, field: value})
        config = copy.deepcopy(self.config)
        config["runs"][0]["method"] = "spci"
        with self.assertRaisesRegex(ValueError, "method must be"):
            validate_evaluation_config(config)
        config["runs"][0].update(method="qr_cp", quantile_pair=[.1, .9])
        with self.assertRaisesRegex(ValueError, "must match nominal"):
            validate_evaluation_config(config)

    def test_example_config_has_only_supported_methods_and_enables_legacy_fallback(self):
        path = Path(__file__).resolve().parents[1] / "configs/conditional_coverage_configs/evaluate_example.yaml"
        config = load_evaluation_config(path)
        self.assertGreater(len(config["runs"]), 0)
        self.assertTrue(config["allow_chronological_fallback"])
        self.assertLessEqual({run["method"] for run in config["runs"]}, {"qr_cp", "iqn_cp", "local_cp"})


if __name__ == "__main__":
    unittest.main()
