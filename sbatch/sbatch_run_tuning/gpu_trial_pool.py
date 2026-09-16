"""Run independent trials in persistent, explicitly assigned GPU processes.

The scheduler itself is CUDA-independent so its process lifecycle can also be
tested on CPU. Device initialization belongs to the caller's initializer.
"""

import multiprocessing
from multiprocessing.connection import wait
import traceback


def _worker_entry(connection, device, worker_fn, initializer, initargs):
    try:
        if initializer is not None:
            initializer(device, *initargs)
        while True:
            command, task = connection.recv()
            if command == "stop":
                return
            # Pipe.send serializes synchronously: serialization errors are
            # reported just like training errors instead of losing a result.
            connection.send(("result", worker_fn(task)))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def iter_parallel_trials(tasks, devices, worker_fn, *, initializer=None, initargs=()):
    """Yield completed trials, assigning the next task to each free GPU.

    Functions and task payloads must be picklable by the spawn start method.
    There is at most one worker and one active trial per device. A worker
    failure aborts the iterator and stops the other workers; the caller should
    write the final ranking only after exhausting this iterator successfully.
    """
    devices = list(devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Trial workers require a nonempty list of distinct devices.")
    task_iterator = iter(tasks)
    exhausted = object()
    context = multiprocessing.get_context("spawn")
    workers = []
    pending = {}
    completed = False

    try:
        for device in devices:
            task = next(task_iterator, exhausted)
            if task is exhausted:
                break
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker_entry,
                args=(child, device, worker_fn, initializer, initargs),
                name=f"qr-trial-{device}",
            )
            try:
                process.start()
            except BaseException:
                parent.close()
                child.close()
                raise
            child.close()
            workers.append((process, parent))
            pending[parent] = (process, device)
            parent.send(("task", task))

        while pending:
            ready = wait(
                list(pending) + [process.sentinel for process, _ in pending.values()]
            )
            # Drain results before checking exit signals, so a child's error
            # message is preferred over the less specific exit-code report.
            for connection in list(pending):
                if connection not in ready:
                    continue
                process, device = pending[connection]
                try:
                    status, result = connection.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError(
                        f"Trial worker on {device} exited without returning its result "
                        f"(exit code {process.exitcode})."
                    ) from exc
                if status == "error":
                    raise RuntimeError(f"Trial worker on {device} failed:\n{result}")
                if status != "result":
                    raise RuntimeError(f"Unexpected trial worker response: {status!r}")
                yield result
                task = next(task_iterator, exhausted)
                if task is exhausted:
                    connection.send(("stop", None))
                    del pending[connection]
                else:
                    connection.send(("task", task))

            for connection, (process, device) in pending.items():
                if process.sentinel in ready and not connection.poll():
                    raise RuntimeError(
                        f"Trial worker on {device} exited unexpectedly "
                        f"(exit code {process.exitcode})."
                    )
        completed = True
    finally:
        # Also runs when the caller fails while saving an intermediate result.
        # Never wait for a long-running training trial after another fails.
        for process, connection in workers:
            connection.close()
            if not completed and process.is_alive():
                process.terminate()
        for process, _ in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            process.close()
