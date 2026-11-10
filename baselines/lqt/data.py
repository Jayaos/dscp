"""NexCP boundaries with calibration-only tuning and fresh test streams."""

from collections.abc import Mapping
import numpy as np
from baselines.aci.data import sequence_array, finite_sequence, split_boundaries, make_scores


def prepare_sequence(item, config, key=None, split="test"):
    if split not in {"test", "calibration"}:
        raise ValueError("split must be 'test' or 'calibration'.")
    targets = sequence_array(item["heldout_y"], "heldout_y")
    predictions = sequence_array(item["heldout_predictions"], "heldout_predictions")
    if len(targets) != len(predictions):
        raise ValueError("heldout_y and heldout_predictions must have matching lengths.")
    settings = config["data"]
    boundary = settings.get("test_start")
    if isinstance(boundary, Mapping):
        if key not in boundary:
            raise ValueError(f"data.test_start has no boundary for sequence {key!r}.")
        boundary = boundary[key]
    boundaries = split_boundaries(len(targets), settings["calibration_ratio"],
                                  settings.get("test_ratio"), boundary)
    start, end = ((0, boundaries["test_start"]) if split == "calibration"
                  else (boundaries["test_start"], len(targets)))
    # Only the selected stream's values are inspected or scored.
    y = finite_sequence(targets[start:end], "available heldout_y")
    predictions = finite_sequence(predictions[start:end], "available heldout_predictions")
    scores, _ = make_scores(y, predictions, "absolute")
    return {"y": y, "predictions": predictions, "scores": scores,
            "target_indices": np.arange(start, end, dtype=np.int64),
            "boundaries": {**boundaries, "evaluation_start": start, "evaluation_end": end}}
