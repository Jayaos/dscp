"""Causal residual streams and the exact NexCP calibration/test boundary."""

from collections.abc import Mapping
import numpy as np

from baselines.aci.data import sequence_array, finite_sequence, split_boundaries


def score_streams(targets, predictions, score_type):
    with np.errstate(over="ignore", invalid="ignore"):
        residual = targets - predictions
    if not np.isfinite(residual).all():
        raise ValueError("Residual scores overflowed.")
    if score_type == "signed":
        return {"lower": -residual, "upper": residual}
    if score_type == "absolute":
        return {"absolute": np.abs(residual)}
    raise ValueError("score_type must be 'signed' or 'absolute'.")


def prepare_sequence(item, config, key=None, split="test"):
    if split not in {"test", "calibration"}:
        raise ValueError("split must be 'test' or 'calibration'.")
    targets = sequence_array(item["heldout_y"], "heldout_y")
    predictions = sequence_array(item["heldout_predictions"], "heldout_predictions")
    if len(targets) != len(predictions):
        raise ValueError("heldout_y and heldout_predictions must have matching lengths.")
    settings = config["data"]
    start = settings.get("test_start")
    if isinstance(start, Mapping):
        if key not in start:
            raise ValueError(f"data.test_start has no boundary for sequence {key!r}.")
        start = start[key]
    bounds = split_boundaries(len(targets), settings["calibration_ratio"], settings.get("test_ratio"), start)
    start, end = (0, bounds["test_start"]) if split == "calibration" else (bounds["test_start"], len(targets))
    stream_start = start
    if split == "test":
        initialization = config["model"]["initialization"]
        if initialization == "calibration_replay":
            stream_start = 0
        elif initialization == "calibration_tail":
            warmup = config["model"]["window_size"] + 1
            if start < warmup:
                raise ValueError(
                    f"Sequence {key!r} has {start} calibration observations, but "
                    f"calibration_tail requires at least {warmup} (window_size + 1). "
                    "Reduce the window size or choose another initialization mode."
                )
            # The update excludes its current score from the CDF window. One
            # extra observation produces a refined threshold before testing.
            stream_start = start - warmup
    # Calibration selection does not inspect any test values, including finiteness.
    y = finite_sequence(targets[stream_start:end], "available heldout_y")
    point = finite_sequence(predictions[stream_start:end], "available heldout_predictions")
    offset = start - stream_start
    return {"scores": score_streams(y, point, config["model"]["score_type"]),
            "prefix_length": offset, "y": y[offset:], "predictions": point[offset:],
            "target_indices": np.arange(start, end, dtype=np.int64),
            "boundaries": {**bounds, "stream_start": stream_start,
                           "evaluation_start": start, "evaluation_end": end}}


def interval_endpoints(predictions, thresholds):
    """Signed thresholds may be negative; only an empty intersection is empty."""
    if "absolute" in thresholds:
        lower_radius = upper_radius = thresholds["absolute"]
        empty = lower_radius < 0
    else:
        lower_radius, upper_radius = thresholds["lower"], thresholds["upper"]
        # Comparing avoids overflow when adding two finite signed thresholds.
        empty = lower_radius < -upper_radius
    with np.errstate(over="ignore", invalid="ignore"):
        lower = np.asarray(predictions) - lower_radius
        upper = np.asarray(predictions) + upper_radius
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise RuntimeError("Finite COP thresholds overflowed response-scale endpoints.")
    raw_lower, raw_upper = lower.copy(), upper.copy()
    lower[empty], upper[empty] = np.inf, -np.inf
    return lower, upper, raw_lower, raw_upper, empty
