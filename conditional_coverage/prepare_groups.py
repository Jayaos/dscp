"""Build shared contextual groups: python -m conditional_coverage.prepare_groups."""

import argparse
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import platform

import numpy as np

from .config import file_sha256, load_comparisons, load_config, validate_config
from .data import observed_sequence, prepare_contexts, shared_indices


def sequence_seed(seed, key, purpose):
    payload = json.dumps([seed, type(key).__name__, key, purpose], ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little")


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _verify_logged_targets(comparison, key, y, indices):
    if "log" not in comparison:
        return
    record = comparison["log"][key]
    metadata = record.get("metadata", {})
    for result in record.get("evaluation_results", {}).values():
        if "target_y" not in result:
            continue
        targets = np.asarray(result["target_y"])
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if targets.ndim != 1:
            raise ValueError(f"{comparison['name']}/{key}: expected scalar logged targets.")
        logged_indices = result.get("target_indices")
        if logged_indices is None:
            logged_indices = indices
            if "valid_prediction_mask" in metadata:
                logged_indices = np.asarray(metadata.get("target_indices", indices))[
                    np.asarray(metadata["valid_prediction_mask"], dtype=bool)
                ]
        logged_indices = np.asarray(logged_indices)
        if logged_indices.size == 0:
            logged_indices = logged_indices.astype(np.int64)
        if (logged_indices.ndim != 1 or logged_indices.dtype.kind not in "iu"
                or len(logged_indices) != len(targets)
                or np.any(logged_indices < 0) or np.any(logged_indices >= len(y))):
            raise ValueError(f"{comparison['name']}/{key}: cannot align logged targets.")
        if not np.allclose(targets, y[logged_indices], rtol=1e-5, atol=1e-6, equal_nan=True):
            raise ValueError(f"{comparison['name']}/{key}: logged targets disagree with source timeline.")


def inspect_inputs(config):
    """Load and audit inputs, without fitting models or writing output files."""
    comparisons = load_comparisons(config)
    source_sha256 = file_sha256(config["data_path"])
    with Path(config["data_path"]).open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, Mapping) or not data:
        raise ValueError("Source artifact must be a nonempty mapping of sequences.")
    if any(isinstance(key, bool) or not isinstance(key, (str, int)) for key in data):
        raise ValueError("Source sequence keys must be strings or integers.")
    requested = config.get("sequences")
    keys = list(data) if requested is None else list(requested)
    unknown = set(keys) - set(data)
    if unknown:
        raise ValueError(f"Requested sequences not in source: {unknown!r}.")
    skipped = []
    available = []
    for key in keys:
        missing = [c["name"] for c in comparisons if "log" in c and key not in c["log"]]
        if missing:
            if requested is not None:
                raise ValueError(f"Requested sequence {key!r} absent from runs: {missing}.")
            skipped.append({"sequence_key": key, "missing_from": missing})
        else:
            available.append(key)
    if not available:
        raise ValueError("No sequences are shared by all comparison runs.")
    audit = []
    for key in available:
        x, y = observed_sequence(data[key])
        shared = shared_indices(len(y), comparisons, key)
        prepared = prepare_contexts(
            x, y, shared["calibration_indices"], shared["test_indices"],
            config["model"]["window_size"], config["training"]["validation_ratio"],
        )
        if len(prepared["calibration_target_indices"]) < config["clustering"]["k"]:
            raise ValueError(f"{key!r}: fewer calibration contexts than requested clusters.")
        for comparison in comparisons:
            original = shared_indices(len(y), [comparison], key)["test_indices"]
            _verify_logged_targets(comparison, key, y, original)
        audit.append({
            "sequence_key": key, "length": len(y),
            "calibration_end": int(shared["calibration_indices"][-1]) + 1,
            "test_start": int(shared["test_indices"][0]),
            "calibration_contexts": len(prepared["calibration_target_indices"]),
            "fit_contexts": len(prepared["fit_indices"]),
            "validation_contexts": len(prepared["validation_indices"]),
            "test_contexts": len(prepared["test_indices"]),
            "boundary_gap_count": len(shared["excluded_boundary_indices"]),
            "member_boundaries": shared["member_boundaries"],
        })
    return data, comparisons, available, {
        "sequences": audit, "skipped_sequences": skipped, "source_sha256": source_sha256,
    }


def _execution_settings(config, sequence_count):
    """Cap sequence concurrency by the selected data and the Slurm allocation."""
    requested = config.get("cpu_workers", 1)
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated is not None:
        try:
            allocated = int(allocated)
        except ValueError as exc:
            raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer.") from exc
        if allocated < 1:
            raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer.")
    workers = min(requested, sequence_count)
    if allocated is not None:
        workers = min(workers, allocated)
    return {
        "requested_cpu_workers": requested, "cpu_workers": workers,
        "threads_per_worker": config["num_threads"],
        "slurm_cpus_per_task": allocated,
        "backend": "spawn_processes" if workers > 1 else "serial",
        "device": str(config["device"]),
    }


_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)


@contextmanager
def _single_thread_worker_environment():
    # Spawned Python imports may initialize BLAS before the pool initializer.
    original = {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT}
    try:
        for name in _THREAD_ENVIRONMENT:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _initialize_cpu_worker():
    """Limit each fresh spawned process to one compute thread per library."""
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    import sklearn  # Load its OpenMP library before applying threadpoolctl.
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    global _CPU_THREAD_LIMITS
    _CPU_THREAD_LIMITS = threadpool_limits(limits=1)


def _prepare_sequence(task):
    """Fit one sequence independently; tasks contain no other sequence's data."""
    from .encoder import extract_embeddings, train_encoder
    from .grouping import fit_groups

    key, x, y, shared, config = task
    prepared = prepare_contexts(
        x, y, shared["calibration_indices"], shared["test_indices"],
        config["model"]["window_size"], config["training"]["validation_ratio"],
    )
    encoder_seed = sequence_seed(config["seed"], key, "encoder")
    clustering_seed = sequence_seed(config["seed"], key, "clustering")
    training_settings = {k: v for k, v in config["training"].items() if k != "validation_ratio"}
    model, training = train_encoder(
        prepared, config["model"], training_settings, encoder_seed, device=config["device"],
    )
    calibration_embeddings = extract_embeddings(
        model, prepared, prepared["calibration_target_indices"], device=config["device"],
    )
    test_embeddings = extract_embeddings(
        model, prepared, prepared["test_indices"], device=config["device"],
    )
    groups = fit_groups(
        calibration_embeddings, test_embeddings, k=config["clustering"]["k"],
        seed=clustering_seed, n_init=config["clustering"]["n_init"],
    )
    record = {
        **shared, "calibration_target_indices": prepared["calibration_target_indices"],
        "fit_indices": prepared["fit_indices"], "validation_indices": prepared["validation_indices"],
        "scaler": prepared["scaler"], "training": training, "groups": groups,
    }
    if config["cache_embeddings"]:
        record.update(calibration_embeddings=calibration_embeddings, test_embeddings=test_embeddings)
    checkpoint = {
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "model_config": config["model"], "dim_feature": prepared["features"].shape[1],
        "training_quantiles": [0.5], "target": "observed_y",
        "scaler": {name: value.tolist() if isinstance(value, np.ndarray) else value
                   for name, value in prepared["scaler"].items()},
        "seed": encoder_seed, "best_epoch": training["best_epoch"],
    }
    return record, checkpoint


def _cpu_group_task(task):
    key = task[0]
    try:
        record, checkpoint = _prepare_sequence(task)
    except Exception as exc:
        raise ValueError(f"Sequence {key!r} failed while preparing shared groups: {exc}") from exc
    # Ordinary pickle embeds tensor storage in bytes instead of passing Torch
    # shared-memory handles whose lifetime depends on the worker process.
    return record, pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL)


def _parallel_sequences(tasks, workers, completed):
    """Keep only one submitted sequence per worker and collect in the parent."""
    tasks = iter(tasks)
    with _single_thread_worker_environment(), ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_cpu_worker,
    ) as executor:
        pending = {}

        def submit_next():
            try:
                task = next(tasks)
            except StopIteration:
                return
            pending[executor.submit(_cpu_group_task, task)] = task[0]

        for _ in range(workers):
            submit_next()
        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    key = pending.pop(future)
                    record, checkpoint_bytes = future.result()
                    completed(key, record, pickle.loads(checkpoint_bytes))
                    submit_next()
        except BaseException:
            for future in pending:
                future.cancel()
            raise


def build_shared_groups(config, *, dry_run=False, progress=print):
    """Prepare one immutable grouping artifact for the selected comparisons.

    Accept a configuration path or a mapping (mapping paths are relative to
    cwd unless path_base is explicit). Existing destinations are never replaced.
    """
    config = (load_config(config) if isinstance(config, (str, Path))
              else validate_config(config))
    output = Path(config["output_dir"])
    if not dry_run and output.exists():
        raise FileExistsError(f"Grouping artifact already exists: {output}. Choose a new output_dir.")
    data, comparisons, keys, audit = inspect_inputs(config)
    execution = _execution_settings(config, len(keys))
    if dry_run:
        return {"dry_run": True, "output_dir": str(output), "config": config,
                "execution": execution, **audit}

    import sklearn
    import torch
    from threadpoolctl import threadpool_limits
    from .artifacts import save_group_artifact
    from .encoder import resolve_device

    device = resolve_device(config["device"])
    execution["device"] = str(device)
    threads_before = torch.get_num_threads()
    sequences, checkpoints = {}, {}
    thread_limits = threadpool_limits(limits=config["num_threads"])
    settings = {name: config[name] for name in
                ("seed", "model", "training", "clustering", "cache_embeddings")}
    settings["device"] = str(device)

    def tasks():
        for key in keys:
            x, y = observed_sequence(data[key])
            shared = shared_indices(len(y), comparisons, key)
            yield key, x, y, shared, settings

    def completed(key, record, checkpoint):
        sequences[key], checkpoints[key] = record, checkpoint
        counts = {str(k): group["test_counts"].tolist() for k, group in record["groups"].items()}
        progress(f"[{len(sequences)}/{len(keys)}] Finished {key!r}; "
                 f"selected epoch {record['training']['best_epoch']}; test group sizes: {counts}")

    progress(f"Preparing {len(keys)} sequences with {execution['cpu_workers']} worker(s), "
             f"{execution['threads_per_worker']} thread(s) per worker on {device}.")
    try:
        torch.set_num_threads(config["num_threads"])
        if execution["cpu_workers"] > 1:
            _parallel_sequences(tasks(), execution["cpu_workers"], completed)
        else:
            for task in tasks():
                record, checkpoint = _prepare_sequence(task)
                completed(task[0], record, checkpoint)
    finally:
        torch.set_num_threads(threads_before)
        thread_limits.restore_original_limits()

    # Completion order does not change stored sequence ordering or identities.
    sequences = {key: sequences[key] for key in keys}
    checkpoints = {key: checkpoints[key] for key in keys}

    # Exact per-sequence boundaries are in the audit. Keep configuration
    # fingerprints here rather than serializing method-specific mappings whose
    # integer and string sequence keys cannot both be represented by JSON keys.
    comparison_metadata = [{k: v for k, v in c.items() if k not in {"log", "config"}}
                           for c in comparisons]
    if file_sha256(config["data_path"]) != audit["source_sha256"]:
        raise ValueError("Source artifact changed while groups were being prepared; output was not saved.")
    manifest = {
        "source": {"path": config["data_path"], "sha256": audit["source_sha256"]},
        "dataset": config.get("dataset"), "base_predictor": config.get("base_predictor"),
        "target": "observed_y", "horizon": 1,
        "context": "concat(heldout_x[j-W:j], heldout_y[j-W:j]); target=heldout_y[j]",
        "calibration_policy": "intersection_of_entire_pre_test_heldout_prefixes",
        "test_policy": "intersection_of_original_test_indices_before_prediction_exclusions",
        "preprocessing_policy": "fit_on_encoder_fitting_prefix_then_freeze",
        "embedding": "last_hidden_state_l2_normalized",
        "clustering": "euclidean_kmeans_on_calibration_embeddings_only",
        "k": config["clustering"]["k"],
        "comparisons": comparison_metadata, "audit": audit,
        "execution": execution,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__, "scikit_learn": sklearn.__version__},
    }
    # Storage paths and comparison names do not change the learned partition.
    # The identity follows data content, the actual shared splits and fitting
    # settings, so a reproducibility run can use a different output directory.
    sequence_specification = [
        {field: entry[field] for field in ("sequence_key", "length", "calibration_end", "test_start")}
        for entry in audit["sequences"]
    ]
    sequence_specification.sort(key=lambda entry: json.dumps(
        [type(entry["sequence_key"]).__name__, entry["sequence_key"]], ensure_ascii=False,
    ))
    identity_settings = {
        "schema_version": 1, "source": manifest["source"]["sha256"],
        "model": config["model"], "training": config["training"],
        "clustering": config["clustering"], "seed": config["seed"],
        "device": str(device), "num_threads": config["num_threads"],
        "context": manifest["context"], "versions": manifest["versions"],
        "sequences": sequence_specification,
    }
    identity = json.dumps(identity_settings, sort_keys=True, default=_serializable)
    manifest["grouping_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    destination = save_group_artifact(
        output, manifest, sequences, checkpoints, config,
        cache_embeddings=config["cache_embeddings"],
    )
    progress(f"Saved shared groups to {destination}")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Evaluator grouping YAML.")
    parser.add_argument("--output-dir", type=Path, help="New output directory; never overwritten.")
    parser.add_argument("--device", help="cpu, cuda, or cuda:N.")
    parser.add_argument("--cpu-workers", type=int,
                        help="Train independent sequences concurrently on CPUs, one thread per worker.")
    parser.add_argument("--dry-run", action="store_true", help="Audit sources and shared splits without training or writing.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.output_dir is not None:
            config["output_dir"] = str(args.output_dir.expanduser().resolve())
        if args.device is not None:
            config["device"] = args.device
        if args.cpu_workers is not None:
            if args.device is not None and args.device != "cpu":
                raise ValueError("--cpu-workers requires --device cpu when a device is specified.")
            config.update(cpu_workers=args.cpu_workers, device="cpu", num_threads=1)
        result = build_shared_groups(config, dry_run=args.dry_run)
        if args.dry_run:
            print(json.dumps(result, indent=2, default=_serializable))
        return result
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
