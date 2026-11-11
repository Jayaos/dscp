"""Run independent Local-CP sequences in single-threaded CPU processes."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import multiprocessing
import os
from pathlib import Path
import random


CPU_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "MPLBACKEND": "Agg",
}


@contextmanager
def _cpu_environment():
    previous = {name: os.environ.get(name) for name in CPU_ENVIRONMENT}
    os.environ.update(CPU_ENVIRONMENT)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _initialize_cpu_worker():
    os.environ.update(CPU_ENVIRONMENT)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _run_sequence(config_path, encoder, key, record, sequence_seed):
    import numpy as np
    import torch

    from dscp.run_local_cp import run_rnn_local_cp, run_transformer_local_cp

    random.seed(sequence_seed)
    np.random.seed(sequence_seed)
    torch.manual_seed(sequence_seed)
    print(f"Starting sequence {key!r} on CPU (seed={sequence_seed})", flush=True)
    runner = run_rnn_local_cp if encoder == "rnn" else run_transformer_local_cp
    log = runner(config_path, sequence_data={key: record}, save_aggregate=False)
    log[key]["sequence_seed"] = sequence_seed
    return log[key]


def run_cpu_local_cp(config_path, *, encoder, workers):
    """Train one sequence per worker and merge results in artifact order.

    Workers prepare only their assigned sequence and write distinct model files.
    The parent alone writes the aggregate logs, summary, and plots. Sequence
    seeds depend on their original position, never on process scheduling.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if encoder not in ("rnn", "transformer"):
        raise ValueError("encoder must be 'rnn' or 'transformer'")

    from utils.experiment_config import load_experiment_config

    config = load_experiment_config(config_path)
    if config.device != "cpu":
        raise ValueError("CPU sequence workers require device: cpu")

    with _cpu_environment():
        from utils.utils import load_data, save_data

        data = load_data(config.data.data_path)
        if not data:
            raise ValueError("No sequences found in the prediction artifact")
        keys = list(data)
        worker_count = min(workers, len(keys))
        output_dir = Path(config.saving_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_seed = int(config.get("seed", 2026))
        print(
            f"Local-CP CPU: {len(keys)} sequences, {worker_count} workers, "
            "one thread per worker",
            flush=True,
        )
        records = iter(enumerate(data.items()))
        completed = {}
        # Spawn avoids inheriting initialized PyTorch/OpenMP thread pools.
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_cpu_worker,
        ) as executor:
            pending = {}

            def submit_next():
                try:
                    index, (key, record) = next(records)
                except StopIteration:
                    return
                sequence_seed = (base_seed + index) % (2**32)
                future = executor.submit(
                    _run_sequence, str(config_path), encoder, key, record, sequence_seed,
                )
                pending[future] = key

            for _ in range(worker_count):
                submit_next()
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    key = pending.pop(future)
                    try:
                        completed[key] = future.result()
                    except Exception as exc:
                        for remaining in pending:
                            remaining.cancel()
                        raise RuntimeError(f"Local-CP sequence {key!r} failed") from exc
                    print(f"Completed sequence {key!r} ({len(completed)}/{len(keys)})", flush=True)
                    submit_next()

        log = {key: completed[key] for key in keys}
        save_data(output_dir / "log.pkl", log)
        from dscp.run_local_cp import _save_local_cp_summary

        _save_local_cp_summary(config, log)
        return log
