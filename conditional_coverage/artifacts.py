"""Persist shared groups and independent encoder checkpoints as one artifact."""

from collections.abc import Mapping
import hashlib
import json
from numbers import Integral
from pathlib import Path, PurePosixPath
import pickle
import tempfile

import numpy as np


SCHEMA_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_identity(key):
    if isinstance(key, str):
        return {"type": "str", "value": key}
    if isinstance(key, Integral) and not isinstance(key, (bool, np.bool_)):
        return {"type": "int", "value": int(key)}
    raise ValueError("Sequence identifiers must be strings or integers.")


def _identity_text(key):
    return json.dumps(_sequence_identity(key), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _indices(values, name, *, allow_empty=False):
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    if not allow_empty and not len(result):
        raise ValueError(f"{name} cannot be empty.")
    if np.any(result < 0) or np.any(result[1:] <= result[:-1]):
        raise ValueError(f"{name} must contain increasing, unique, nonnegative indices.")
    return result


def validate_sequences(sequences):
    """Check index identity, label alignment, and counts without loading models."""
    if not isinstance(sequences, Mapping) or not sequences:
        raise ValueError("A grouping artifact requires a nonempty sequence mapping.")
    for key, record in sequences.items():
        _sequence_identity(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"Sequence {key!r} must be a mapping.")
        required = {
            "calibration_indices", "calibration_target_indices", "test_indices",
            "excluded_boundary_indices", "scaler", "member_boundaries", "groups", "training",
        }
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Sequence {key!r} is missing fields: {sorted(missing)}.")
        calibration = _indices(record["calibration_indices"], "calibration_indices")
        targets = _indices(record["calibration_target_indices"], "calibration_target_indices")
        test = _indices(record["test_indices"], "test_indices")
        excluded = _indices(record["excluded_boundary_indices"], "excluded_boundary_indices", allow_empty=True)
        if not np.array_equal(calibration, np.arange(int(calibration[-1]) + 1)):
            raise ValueError("calibration_indices must identify the full shared prefix starting at zero.")
        if not np.isin(targets, calibration).all() or calibration[-1] >= test[0]:
            raise ValueError("Calibration targets must be inside the prefix, strictly before test.")
        expected_gap = np.arange(int(calibration[-1]) + 1, int(test[0]))
        if not np.array_equal(excluded, expected_gap):
            raise ValueError("excluded_boundary_indices must identify the gap between shared partitions.")
        if not isinstance(record["groups"], Mapping) or not record["groups"]:
            raise ValueError("Each sequence requires at least one clustering result.")
        dimensions = set()
        for k, group in record["groups"].items():
            if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 1:
                raise ValueError("Group keys must be positive integer K values.")
            if not isinstance(group, Mapping):
                raise ValueError("Each clustering result must be a mapping.")
            missing = {"centroids", "calibration_labels", "test_labels", "calibration_counts", "test_counts", "inertia", "seed", "n_init"} - group.keys()
            if missing:
                raise ValueError(f"Clustering K={k} is missing fields: {sorted(missing)}.")
            centroids = np.asarray(group["centroids"])
            if (centroids.ndim != 2 or centroids.shape[0] != k or centroids.shape[1] < 1
                    or centroids.dtype.kind not in "iuf" or not np.isfinite(centroids).all()):
                raise ValueError(f"K={k} requires {k} finite centroid vectors.")
            dimensions.add(centroids.shape[1])
            for split, size in (("calibration", len(targets)), ("test", len(test))):
                labels = np.asarray(group[f"{split}_labels"])
                counts = np.asarray(group[f"{split}_counts"])
                if (labels.shape != (size,) or labels.dtype.kind not in "iu"
                        or np.any(labels < 0) or np.any(labels >= k)):
                    raise ValueError(f"K={k} {split} labels must align with target indices and be in [0, K).")
                if (counts.shape != (k,) or counts.dtype.kind not in "iu"
                        or not np.array_equal(counts, np.bincount(labels.astype(np.int64), minlength=k))):
                    raise ValueError(f"K={k} {split} counts do not match labels.")
                if split == "calibration" and np.any(counts == 0):
                    raise ValueError("Calibration groups must all be nonempty.")
            inertia = group["inertia"]
            if not np.isscalar(inertia) or not np.isfinite(inertia) or inertia < 0:
                raise ValueError("K-means inertia must be finite and nonnegative.")
        if len(dimensions) != 1:
            raise ValueError("All K values must use the same embedding dimension.")
        dimension = next(iter(dimensions))
        for field, size in (("calibration_embeddings", len(targets)), ("test_embeddings", len(test))):
            if field in record:
                values = np.asarray(record[field])
                if (values.shape != (size, dimension) or values.dtype.kind not in "iuf"
                        or not np.isfinite(values).all()
                        or not np.allclose(np.linalg.norm(values, axis=1), 1.0, rtol=1e-5, atol=1e-6)):
                    raise ValueError(f"{field} must contain aligned L2-normalized embeddings.")


def save_group_artifact(output_dir, manifest, sequences, checkpoints, resolved_config, cache_embeddings=True):
    """Publish a complete artifact atomically; refuse an existing destination.

    Checkpoint paths are derived from typed sequence identifiers, so sequence
    names cannot escape the artifact and integer/string names cannot collide.
    Pickle and checkpoint files are intended for trusted local experiment data.
    """
    import torch
    import yaml

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Grouping output already exists: {destination}")
    if not isinstance(manifest, Mapping) or not isinstance(resolved_config, Mapping):
        raise ValueError("manifest and resolved_config must be mappings.")
    if not isinstance(cache_embeddings, bool):
        raise ValueError("cache_embeddings must be a boolean.")
    validate_sequences(sequences)
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(sequences):
        raise ValueError("Every sequence must have exactly one encoder checkpoint.")
    stored_sequences = {
        key: {field: value for field, value in record.items()
              if cache_embeddings or field not in {"calibration_embeddings", "test_embeddings"}}
        for key, record in sequences.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temporary:
        staging = Path(temporary).resolve()
        # TemporaryDirectory only removes this newly allocated sibling on failure.
        if staging.parent != destination.parent:
            raise ValueError("Artifact staging directory escaped its intended parent.")
        (staging / "encoders").mkdir()
        with (staging / "groups.pkl").open("wb") as stream:
            pickle.dump({"schema_version": SCHEMA_VERSION, "sequences": stored_sequences}, stream,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with (staging / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(dict(resolved_config), stream, sort_keys=False)
        references = []
        for key in sorted(sequences, key=_identity_text):
            identity = _sequence_identity(key)
            digest = hashlib.sha256(_identity_text(key).encode("utf-8")).hexdigest()
            checkpoint_path = f"encoders/{digest}.pt"
            if not isinstance(checkpoints[key], Mapping) or "model_state_dict" not in checkpoints[key]:
                raise ValueError("Encoder checkpoints must contain a model_state_dict.")
            torch.save(dict(checkpoints[key]), staging / checkpoint_path)
            references.append({"sequence": identity, "checkpoint": checkpoint_path})
        files = ["groups.pkl", "resolved_config.yaml"] + [entry["checkpoint"] for entry in references]
        saved_manifest = {
            **dict(manifest),
            "schema_version": SCHEMA_VERSION,
            "cache_embeddings": cache_embeddings,
            "sequences": references,
            "files": {name: file_sha256(staging / name) for name in files},
        }
        with (staging / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(saved_manifest, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        if destination.exists():
            raise FileExistsError(f"Grouping output already exists: {destination}")
        staging.rename(destination)
    return destination


def _artifact_path(directory, name):
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("Manifest file paths must be relative POSIX paths.")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Manifest file paths must stay inside the artifact.")
    path = (directory / Path(*relative.parts)).resolve()
    if directory not in path.parents:
        raise ValueError("Manifest file path escaped the artifact.")
    return path


def load_group_artifact(output_dir):
    """Verify checksums and membership contracts, then return manifest/groups."""
    directory = Path(output_dir).resolve()
    with (directory / "manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported grouping artifact schema version.")
    files = manifest.get("files")
    references = manifest.get("sequences")
    if not isinstance(files, dict) or not {"groups.pkl", "resolved_config.yaml"}.issubset(files):
        raise ValueError("Grouping manifest is missing required checksums.")
    if not isinstance(references, list) or not references:
        raise ValueError("Grouping manifest is missing sequence checkpoint references.")
    for name, checksum in files.items():
        path = _artifact_path(directory, name)
        if not path.is_file() or file_sha256(path) != checksum:
            raise ValueError(f"Grouping artifact checksum mismatch: {name}")
    with (directory / "groups.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported groups payload schema version.")
    sequences = payload.get("sequences")
    validate_sequences(sequences)
    expected = {_identity_text(key) for key in sequences}
    actual = set()
    checkpoint_paths = set()
    for reference in references:
        if not isinstance(reference, dict) or not isinstance(reference.get("sequence"), dict):
            raise ValueError("Invalid sequence checkpoint reference.")
        identity = reference["sequence"]
        if set(identity) != {"type", "value"}:
            raise ValueError("Invalid typed sequence identifier.")
        actual_identity = _sequence_identity(identity["value"])
        if identity != actual_identity:
            raise ValueError("Invalid typed sequence identifier.")
        text = _identity_text(identity["value"])
        checkpoint = reference.get("checkpoint")
        if text in actual or checkpoint in checkpoint_paths or checkpoint not in files:
            raise ValueError("Duplicate or missing sequence checkpoint reference.")
        if not isinstance(checkpoint, str) or not checkpoint.startswith("encoders/"):
            raise ValueError("Encoder checkpoints must be inside encoders/.")
        actual.add(text)
        checkpoint_paths.add(checkpoint)
    if actual != expected:
        raise ValueError("Manifest sequence identifiers do not match saved groups.")
    return manifest, sequences
