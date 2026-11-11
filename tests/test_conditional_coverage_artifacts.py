import copy
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import torch

from conditional_coverage.artifacts import (
    file_sha256,
    load_group_artifact,
    save_group_artifact,
    validate_sequences,
)
from conditional_coverage.grouping import fit_groups


def _embeddings():
    angles = np.array([0.0, 0.1, 0.2, 2.0, 2.1, 2.2, 4.0, 4.1, 4.2])
    return np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)


def _record():
    calibration = _embeddings()
    test = np.repeat(calibration[:1], 3, axis=0)
    return {
        "calibration_indices": np.arange(11),
        "calibration_target_indices": np.arange(2, 11),
        "test_indices": np.arange(12, 15),
        "excluded_boundary_indices": np.array([11]),
        "scaler": {"feature_mean": np.zeros(2), "feature_std": np.ones(2)},
        "member_boundaries": {"first": {"test_start": 11}, "second": {"test_start": 12}},
        "training": {"best_epoch": 2, "seed": 42},
        "groups": fit_groups(calibration, test, k=3, seed=42, n_init=2),
        "calibration_embeddings": calibration,
        "test_embeddings": test,
    }


def _checkpoint():
    return {"model_state_dict": {"weight": torch.ones(2, 2)}, "model": {"dim_model": 2}}


class ConditionalCoverageGroupingTests(unittest.TestCase):
    def test_assignments_are_nearest_fitted_centroids_and_reproducible(self):
        calibration = _embeddings()
        test = calibration[[0, 4, 8]]
        first = fit_groups(calibration, test, k=3, seed=123, n_init=2)
        second = fit_groups(calibration, test, k=3, seed=123, n_init=2)
        self.assertEqual(set(first), {3})
        for k, group in first.items():
            distances = np.sum((test[:, None] - group["centroids"][None]) ** 2, axis=-1)
            np.testing.assert_array_equal(group["test_labels"], np.argmin(distances, axis=1))
            np.testing.assert_array_equal(group["centroids"], second[k]["centroids"])
            np.testing.assert_array_equal(group["calibration_labels"], second[k]["calibration_labels"])
            self.assertEqual(group["calibration_counts"].sum(), len(calibration))
            self.assertEqual(group["test_counts"].sum(), len(test))
        # Euclidean K-means arithmetic-mean centroids are not renormalized.
        self.assertTrue(np.all(np.linalg.norm(first[3]["centroids"], axis=1) < 0.999))

    def test_test_embeddings_cannot_change_calibration_partition(self):
        calibration = _embeddings()
        first = fit_groups(calibration, calibration[:2], k=3, seed=5, n_init=2)[3]
        changed = fit_groups(calibration, -calibration[:2], k=3, seed=5, n_init=2)[3]
        np.testing.assert_array_equal(first["centroids"], changed["centroids"])
        np.testing.assert_array_equal(first["calibration_labels"], changed["calibration_labels"])

    def test_empty_test_groups_keep_zero_counts(self):
        record = _record()
        counts = record["groups"][3]["test_counts"]
        self.assertEqual(np.count_nonzero(counts), 1)
        self.assertEqual(len(counts), 3)
        validate_sequences({"station": record})

    def test_invalid_or_collapsed_embeddings_and_settings_fail(self):
        embeddings = _embeddings()
        for calibration, test, kwargs in (
            (embeddings * 2, embeddings, {}),
            (np.zeros_like(embeddings), embeddings, {}),
            (embeddings, embeddings[:, :1], {}),
            (np.full_like(embeddings, np.nan), embeddings, {}),
            (embeddings[:2], embeddings, {}),
            (np.repeat(embeddings[:1], 9, axis=0), embeddings, {}),
            (embeddings, embeddings, {"k": [3, 5]}),
            (embeddings, embeddings, {"k": 0}),
            (embeddings, embeddings, {"k": True}),
            (embeddings, embeddings, {"k": 2.5}),
            (embeddings, embeddings, {"n_init": 0}),
            (embeddings, embeddings, {"seed": -1}),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit_groups(calibration, test, **({"k": 3, "n_init": 2} | kwargs))


class ConditionalCoverageArtifactTests(unittest.TestCase):
    def test_roundtrip_preserves_indices_groups_checkpoints_and_provenance(self):
        record = _record()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "groups"
            save_group_artifact(destination, {"source_sha256": "example", "grouping_id": "test"},
                                {"station": record}, {"station": _checkpoint()}, {"seed": 42})
            manifest, sequences = load_group_artifact(destination)
            self.assertEqual(manifest["source_sha256"], "example")
            self.assertEqual(manifest["grouping_id"], "test")
            self.assertEqual(manifest["schema_version"], 1)
            restored = sequences["station"]
            np.testing.assert_array_equal(restored["test_indices"], record["test_indices"])
            np.testing.assert_array_equal(restored["groups"][3]["test_labels"], record["groups"][3]["test_labels"])
            np.testing.assert_array_equal(restored["calibration_embeddings"], record["calibration_embeddings"])
            checkpoint = destination / manifest["sequences"][0]["checkpoint"]
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            torch.testing.assert_close(saved["model_state_dict"]["weight"], torch.ones(2, 2))

    def test_identifiers_are_typed_and_cannot_escape_checkpoint_directory(self):
        keys = (1, "1", "../../elsewhere")
        sequences = {key: _record() for key in keys}
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "groups"
            save_group_artifact(destination, {}, sequences, {key: _checkpoint() for key in keys}, {})
            manifest, loaded = load_group_artifact(destination)
            self.assertEqual(set(loaded), set(keys))
            references = manifest["sequences"]
            self.assertEqual(len({entry["checkpoint"] for entry in references}), len(keys))
            for reference in references:
                self.assertEqual((destination / reference["checkpoint"]).parent, destination / "encoders")

    def test_disabling_cache_keeps_source_record_unchanged(self):
        record = _record()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "groups"
            save_group_artifact(destination, {}, {"s": record}, {"s": _checkpoint()}, {}, cache_embeddings=False)
            manifest, loaded = load_group_artifact(destination)
            self.assertFalse(manifest["cache_embeddings"])
            self.assertNotIn("calibration_embeddings", loaded["s"])
            self.assertNotIn("test_embeddings", loaded["s"])
            self.assertIn("calibration_embeddings", record)

    def test_existing_artifact_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "groups"
            args = (destination, {}, {"s": _record()}, {"s": _checkpoint()}, {})
            save_group_artifact(*args)
            digest = file_sha256(destination / "groups.pkl")
            with self.assertRaises(FileExistsError):
                save_group_artifact(*args)
            self.assertEqual(file_sha256(destination / "groups.pkl"), digest)

    def test_failed_checkpoint_does_not_publish_a_partial_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "groups"
            with self.assertRaises(ValueError):
                save_group_artifact(destination, {}, {"s": _record()}, {"s": {}}, {})
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_checksum_rejects_modified_groups_or_checkpoint(self):
        for changed_file in ("groups", "checkpoint"):
            with self.subTest(changed_file=changed_file), tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "groups"
                save_group_artifact(destination, {}, {"s": _record()}, {"s": _checkpoint()}, {})
                manifest, _ = load_group_artifact(destination)
                relative = "groups.pkl" if changed_file == "groups" else manifest["sequences"][0]["checkpoint"]
                with (destination / relative).open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    load_group_artifact(destination)

    def test_invalid_membership_alignment_and_counts_are_rejected(self):
        record = _record()
        mutations = [
            lambda r: r.update(test_indices=np.array([12, 12, 14])),
            lambda r: r.update(test_indices=np.array([10, 12, 13])),
            lambda r: r.update(excluded_boundary_indices=np.array([], dtype=int)),
            lambda r: r["groups"][3].update(test_labels=np.array([0, 1])),
            lambda r: r["groups"][3].update(test_labels=np.array([0, 1, 3])),
            lambda r: r["groups"][3].update(test_counts=np.array([1, 1, 1])),
            lambda r: r.update(calibration_embeddings=np.zeros((9, 2))),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(record)
            mutate(changed)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                validate_sequences({"station": changed})

    def test_loading_checks_schema_and_membership_even_with_matching_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "groups"
            save_group_artifact(destination, {}, {"s": _record()}, {"s": _checkpoint()}, {})
            with (destination / "groups.pkl").open("rb") as stream:
                payload = pickle.load(stream)
            payload["sequences"]["s"]["groups"][3]["test_counts"] = np.array([1, 1, 1])
            with (destination / "groups.pkl").open("wb") as stream:
                pickle.dump(payload, stream)
            manifest_path = destination / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["groups.pkl"] = file_sha256(destination / "groups.pkl")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "counts do not match"):
                load_group_artifact(destination)
            manifest["schema_version"] = 999
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema version"):
                load_group_artifact(destination)


if __name__ == "__main__":
    unittest.main()
