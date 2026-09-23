"""Durable diagnostics for timestamps excluded by DistMatch evaluation."""

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile


EXCLUSION_FIELDS = (
    "dataset", "predictor", "sequence_key", "seed", "sequence_seed", "split",
    "test_offset", "target_index", "target_y", "target_prediction", "target_residual",
    "triggering_quantile_pair", "excluded_quantile_pairs", "tree_index", "beta",
    "lower_quantile", "upper_quantile", "lower_bound", "upper_bound", "bound_scale",
    "residual_scale", "reason", "action",
)


def _json_default(value):
    """Handle array-library values without importing its numerical runtime."""
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize exclusion value of type {type(value).__name__}.")


def _json(value):
    return json.dumps(value, default=_json_default, allow_nan=False, ensure_ascii=False)


class ExclusionJournal:
    """Write one JSON line per failure before continuing online evaluation.

    Each evaluation invocation owns a distinct file, including retries and
    evaluations of the same key on different nodes. Files are created lazily,
    flushed and synced after each event, and retained if later evaluation fails.
    Calls without a saving directory do not write files.
    """

    def __init__(self, config, key, split):
        output = config.get("saving_dir")
        self.directory = Path(output) / "exclusions" if output else None
        identity = f"{config.get('seed')}\0{type(key).__name__}\0{key}\0{split}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        self.prefix = f"sequence_{digest}_"
        self.path = None
        self._stream = None

    def __enter__(self):
        return self

    def record(self, event):
        if self.directory is None:
            return
        # Serialize before creating a file, so invalid diagnostic data cannot
        # create a misleading empty journal or a partly written JSON object.
        line = _json(event) + "\n"
        if self._stream is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._stream = tempfile.NamedTemporaryFile(
                mode="a", encoding="utf-8", newline="", dir=self.directory,
                prefix=self.prefix, suffix=".jsonl", delete=False,
            )
            self.path = Path(self._stream.name)
        self._stream.write(line)
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def __exit__(self, exc_type, exc, traceback):
        if self._stream is not None:
            self._stream.close()
        return False


def write_excluded_points(log, saving_dir):
    """Publish exclusions from the completed log, in sequence/timestamp order.

    The parent process calls this once, after sequence workers or distributed
    shards finish. Journals from interrupted attempts are deliberately not
    merged into the completed run's CSV. An empty run still gets a header.
    """
    output = Path(saving_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "excluded_points.csv"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output,
            prefix=".excluded_points.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=EXCLUSION_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for key, sequence in log.items():
                for event in sequence.get("metadata", {}).get("excluded_points", []):
                    # The JSON round trip normalizes scalar/array types and
                    # rejects nonfinite values before writing a public record.
                    row = json.loads(_json(event))
                    row.setdefault("sequence_key", str(key))
                    for name, value in row.items():
                        if isinstance(value, (list, dict)):
                            row[name] = _json(value)
                    writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
