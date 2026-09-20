"""Prepare, execute, and merge independent sequence shards on shared storage."""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import pickle
import socket
import tempfile
import time
import uuid

from omegaconf import OmegaConf

from baselines.distmatch.config import positive_integer
from baselines.distmatch.run_distmatch import (
    _resolve_paths,
    _run_metadata,
    _write_run_results,
    evaluate_sequences,
)


SCHEMA_VERSION = 1


def _worker_identity():
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm": {name: os.environ[name] for name in (
            "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_PROCID", "SLURM_NODEID",
            "SLURM_LOCALID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME",
        ) if name in os.environ},
    }


def _config(config):
    config = _resolve_paths(config)
    if not config["data"].get("data_path") or not config.get("saving_dir"):
        raise ValueError("Distributed DistMatch requires data.data_path and saving_dir.")
    return config


def _read_pickle(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        with path.open("rb") as stream:
            return pickle.load(stream)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(f"Cannot read {label} at {path}: {exc}") from exc


def _atomic_pickle(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _artifact_identity(path):
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _load_artifact(config, expected=None):
    path = Path(config["data"]["data_path"])
    identity = _artifact_identity(path)
    if expected is not None and identity != expected:
        raise ValueError("Prediction artifact changed since shard preparation; prepare a new run.")
    data = _read_pickle(path, "prediction artifact")
    if not isinstance(data, dict) or not data:
        raise ValueError("The prediction artifact must be a nonempty dictionary of series.")
    if _artifact_identity(path) != identity:
        raise ValueError("Prediction artifact changed while being read; prepare a new run.")
    return data, identity


def _partitions(keys, num_shards):
    quotient, remainder = divmod(len(keys), num_shards)
    groups, start = [], 0
    for index in range(num_shards):
        end = start + quotient + (index < remainder)
        groups.append(keys[start:end])
        start = end
    return groups


def _load_run(config):
    config = _config(config)
    root = Path(config["saving_dir"])
    manifest = _read_pickle(root / "manifest.pkl", "shard manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Invalid or unsupported DistMatch shard manifest.")
    if manifest.get("config") != config:
        raise ValueError("Configuration does not match the prepared shard manifest.")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
        raise ValueError("Shard manifest is missing its run_id.")
    prepared_at = manifest.get("prepared_at_unix")
    if (isinstance(prepared_at, bool) or not isinstance(prepared_at, (int, float))
            or not math.isfinite(prepared_at) or prepared_at < 0):
        raise ValueError("Shard manifest has an invalid preparation timestamp.")
    count = positive_integer(manifest.get("num_shards"), "manifest.num_shards", minimum=2)
    keys = manifest.get("sequence_keys")
    if (not isinstance(keys, list) or len(keys) < count
            or manifest.get("num_sequences") != len(keys)):
        raise ValueError("Invalid sequence counts in shard manifest.")
    if manifest.get("shard_keys") != _partitions(keys, count):
        raise ValueError("Shard manifest must contain the balanced, ordered sequence partitions.")
    if not isinstance(manifest.get("artifact"), dict):
        raise ValueError("Shard manifest is missing its artifact identity.")
    data, _ = _load_artifact(config, expected=manifest["artifact"])
    if list(data) != keys:
        raise ValueError("Prediction artifact sequence keys do not match the shard manifest.")
    return config, root, manifest, data


@contextmanager
def _claim(path):
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Run step is already claimed: {path}. A previous process may still be running; "
            "inspect an interrupted run before removing its lock."
        ) from exc
    try:
        with stream:
            stream.write(json.dumps(_worker_identity()) + "\n")
        yield
    finally:
        path.unlink(missing_ok=True)


def prepare_shards(config, num_shards):
    """Create a fresh run directory and deterministic contiguous sequence groups.

    ``num_cores`` remains the worker count per shard/node. The manifest is the
    final preparation commit: workers must not run before it exists.
    """
    config = _config(config)
    num_shards = positive_integer(num_shards, "num_shards", minimum=2)
    data, identity = _load_artifact(config)
    if num_shards > len(data):
        raise ValueError(f"num_shards={num_shards} exceeds the {len(data)} available sequences.")
    keys = list(data)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex,
        "config": config,
        "artifact": identity,
        "num_sequences": len(keys),
        "num_shards": num_shards,
        "sequence_keys": keys,
        "shard_keys": _partitions(keys, num_shards),
        "prepared_at_unix": time.time(),
    }
    root = Path(config["saving_dir"])
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Distributed output directory already exists: {root}. Choose a fresh saving_dir."
        ) from exc
    (root / "shards").mkdir()
    for name in ("launch_config.yaml", "resolved_config.yaml"):
        OmegaConf.save(config=OmegaConf.create(config), f=root / name, resolve=True)
    _atomic_pickle(root / "manifest.pkl", manifest)
    print(f"DistMatch: prepared {num_shards} shards with sequence counts "
          f"{[len(group) for group in manifest['shard_keys']]}", flush=True)
    return manifest


def run_shard(config, shard_index):
    """Evaluate one assigned group and atomically publish its completion envelope."""
    shard_index = positive_integer(shard_index, "shard_index", minimum=0)
    config, root, manifest, data = _load_run(config)
    if shard_index >= manifest["num_shards"]:
        raise ValueError(f"shard_index must be smaller than {manifest['num_shards']}.")
    path = root / "shards" / f"shard_{shard_index:04d}.pkl"
    with _claim(path.with_suffix(".lock")):
        if path.exists():
            raise FileExistsError(f"Shard {shard_index} is already complete: {path}")
        keys = manifest["shard_keys"][shard_index]
        selected = {key: data[key] for key in keys}
        del data
        started = time.perf_counter()
        log = evaluate_sequences(selected, config, split="test")
        elapsed = time.perf_counter() - started
        if not isinstance(log, dict) or list(log) != keys:
            raise ValueError(f"Shard {shard_index} returned missing, unexpected, or reordered sequences.")
        if _artifact_identity(Path(config["data"]["data_path"])) != manifest["artifact"]:
            raise ValueError("Prediction artifact changed during shard evaluation; prepare a new run.")
        metadata = _run_metadata(config, len(keys), elapsed)
        metadata["worker"] = _worker_identity()
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "shard_index": shard_index,
            "sequence_keys": keys,
            "config": config,
            "artifact": manifest["artifact"],
            "log": log,
            "metadata": metadata,
        }
        _atomic_pickle(path, envelope)
    print(f"DistMatch: completed shard {shard_index} ({len(keys)} sequences)", flush=True)
    return envelope


def merge_shards(config):
    """Validate every shard before writing ordinary DSCP results in original order."""
    config, root, manifest, data = _load_run(config)
    del data
    with _claim(root / "merge.lock"):
        outputs = ("log.pkl", "summary_results.pkl", "run_metadata.yaml")
        if any((root / name).exists() for name in outputs):
            raise FileExistsError("Merged outputs already exist; refusing to overwrite this run.")
        expected = {f"shard_{index:04d}.pkl" for index in range(manifest["num_shards"])}
        actual = {path.name for path in (root / "shards").glob("*.pkl")}
        missing, unexpected = sorted(expected - actual), sorted(actual - expected)
        if missing or unexpected:
            raise ValueError(f"Incomplete or unexpected shard results: missing={missing}, unexpected={unexpected}")
        log, shard_metadata = {}, []
        for index, keys in enumerate(manifest["shard_keys"]):
            path = root / "shards" / f"shard_{index:04d}.pkl"
            result = _read_pickle(path, f"shard {index} result")
            if not isinstance(result, dict):
                raise ValueError(f"Invalid completion envelope for shard {index}.")
            required = {
                "schema_version": SCHEMA_VERSION, "run_id": manifest["run_id"],
                "shard_index": index, "sequence_keys": keys, "config": config,
                "artifact": manifest["artifact"],
            }
            for field, value in required.items():
                if result.get(field) != value:
                    raise ValueError(f"Shard {index} {field} does not match the prepared run.")
            shard_log = result.get("log")
            if not isinstance(shard_log, dict) or list(shard_log) != keys:
                raise ValueError(f"Shard {index} has missing, duplicate, unexpected, or reordered sequences.")
            if set(log).intersection(shard_log):
                raise ValueError(f"Duplicate sequences in shard {index}.")
            if not isinstance(result.get("metadata"), dict):
                raise ValueError(f"Shard {index} has invalid run metadata.")
            log.update(shard_log)
            shard_metadata.append(result["metadata"])
        if list(log) != manifest["sequence_keys"]:
            raise ValueError("Merged sequence membership or order differs from the prepared run.")
        if _artifact_identity(Path(config["data"]["data_path"])) != manifest["artifact"]:
            raise ValueError("Prediction artifact changed during shard merge; prepare a new run.")
        elapsed = max(0.0, time.time() - manifest["prepared_at_unix"])
        metadata = _run_metadata(config, len(log), elapsed)
        workers = [min(config["num_cores"], len(keys)) for keys in manifest["shard_keys"]]
        metadata["configured_num_cores"] = config["num_cores"] * manifest["num_shards"]
        metadata["effective_num_cores"] = sum(workers)
        metadata["distributed"] = {
            "run_id": manifest["run_id"], "num_shards": manifest["num_shards"],
            "num_cores_per_shard": config["num_cores"],
            "sequence_counts": [len(keys) for keys in manifest["shard_keys"]],
            "effective_workers_per_shard": workers,
            "shard_metadata": shard_metadata,
            "elapsed_includes_preparation_and_scheduling": True,
        }
        return _write_run_results(config, log, metadata)
