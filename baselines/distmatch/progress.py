"""Per-sequence stage progress for terminals and redirected batch logs."""

import sys
import time

from tqdm import tqdm


_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
    "[{elapsed} elapsed, ETA {remaining}, {rate_fmt}]"
)
_TERMINAL_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}, ETA {remaining}]"
)


class SequenceProgress:
    """Estimate remaining time within each stage using completed work units.

    Interactive workers get separate terminal rows. Redirected output uses
    flushed, newline-delimited snapshots at most once every ten seconds, plus
    stage boundaries, so SLURM logs contain no cursor-control sequences.
    """

    def __init__(self, key, enabled=True, position=0):
        self.key = key
        self.enabled = enabled
        self.position = position
        self.stream = sys.stdout
        self.interactive = self.stream.isatty()
        self.stage = None
        self.bar = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def update(self, stage, completed, total):
        if not self.enabled:
            return
        now = time.monotonic()
        new_stage = stage != self.stage
        if new_stage:
            self.close()
            self.stage = stage
            self.started = now
            self.last_print = now
            self.description = f"DistMatch {self.key!r} {stage}"
            self.unit = {"matching": "pair", "trees": "tree"}.get(stage, "step")
            if self.interactive:
                self.bar = tqdm(
                    total=total, desc=self.description, unit=self.unit,
                    file=self.stream, position=self.position, leave=False,
                    mininterval=0.5, miniters=1, dynamic_ncols=True,
                    bar_format=_TERMINAL_BAR_FORMAT,
                )
        if self.bar is not None:
            self.bar.update(completed - self.bar.n)
        elif new_stage or completed == total or now - self.last_print >= 10.0:
            line = tqdm.format_meter(
                completed, total, now - self.started, prefix=self.description,
                unit=self.unit, ascii=True, bar_format=_BAR_FORMAT,
            )
            # Shared across spawned workers by the runner initializer.
            with tqdm.get_lock():
                print(line, file=self.stream, flush=True)
            self.last_print = now
