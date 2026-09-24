"""Shared final-test crossing exclusions for QR-CP and direct cosine IQN-CP."""

import csv
import json
from pathlib import Path

import torch

from utils.reporting import summarize_evaluation_results
from utils.utils import get_sorted_unique_quantiles


EXCLUSION_POLICY = "skip_crossed_quantiles_timestamp_all_coverage_levels"


def _counts(total, evaluated):
    excluded = total - evaluated
    return {
        "total_points": total,
        "evaluated_points": evaluated,
        "excluded_points_count": excluded,
        "exclusion_rate": excluded / total if total else 0.0,
        "evaluation_status": (
            "no_valid_predictions" if not evaluated else
            "completed_with_exclusions" if excluded else "complete"
        ),
    }


class QuantileCrossingTracker:
    """Keep the original test timeline while filtering every pair identically.

    Outputs are already ordered by quantile level. Any descending adjacent
    values imply a crossing, including between levels from different interval
    pairs. Equal values remain valid. This filter is only for final reporting;
    training, checkpoint validation, and tuning keep their original data.
    """

    def __init__(
        self, config, key, heldout_size, test_size, *,
        method="qr_cp", head_metadata=None,
    ):
        self.key = str(key)
        self.method = method
        self.method_label = {"qr_cp": "QR-CP", "iqn_cp": "IQN-CP"}.get(method, method)
        self.head_metadata = (
            {"head_type": "independent"} if head_metadata is None else dict(head_metadata)
        )
        self.quantiles = get_sorted_unique_quantiles(config.model.target_quantiles)
        self.pairs = [list(pair) for pair in config.model.target_quantiles]
        self.target_indices = list(range(heldout_size - test_size, heldout_size))
        self.bound_scale = "normalized_residual" if config.data.normalize else "original_residual"
        self.valid_mask = []
        self.excluded_points = []

    def filter_batch(self, quantile_values, target_residual, target_y, target_predictions):
        quantile_values = quantile_values.detach().cpu()
        if quantile_values.ndim != 2 or quantile_values.shape[1] != len(self.quantiles):
            raise ValueError(f"{self.method_label} crossing exclusions require one-step quantile predictions.")
        if not torch.isfinite(quantile_values).all():
            raise ValueError(f"Nonfinite predicted quantiles for sequence {self.key!r}.")
        crossings = quantile_values[:, :-1] > quantile_values[:, 1:]
        valid = ~crossings.any(dim=1)
        offset = len(self.valid_mask)
        for row in torch.nonzero(~valid, as_tuple=False).flatten().tolist():
            crossed_levels = torch.nonzero(crossings[row], as_tuple=False).flatten().tolist()
            self.excluded_points.append({
                "sequence_key": self.key,
                "split": "test",
                "test_offset": offset + row,
                "target_index": self.target_indices[offset + row],
                "target_y": float(target_y[row].item()),
                "target_prediction": float(target_predictions[row].item()),
                "target_residual": float(target_y[row].item() - target_predictions[row].item()),
                "crossed_quantile_pairs": [
                    [self.quantiles[index], self.quantiles[index + 1]] for index in crossed_levels
                ],
                "excluded_quantile_pairs": self.pairs,
                "quantile_levels": self.quantiles,
                "predicted_quantiles": quantile_values[row].tolist(),
                "bound_scale": self.bound_scale,
                "reason": "quantile_crossing",
                "action": "excluded_from_metrics",
            })
        self.valid_mask.extend(valid.tolist())
        return (
            quantile_values[valid], target_residual[valid],
            target_y[valid], target_predictions[valid],
        )

    def finalize(self, evaluation_results):
        if len(self.valid_mask) != len(self.target_indices):
            raise ValueError(f"{self.method_label} prediction mask does not cover the full test timeline.")
        counts = _counts(len(self.valid_mask), sum(self.valid_mask))
        retained_indices = [index for index, valid in zip(self.target_indices, self.valid_mask) if valid]
        for result in evaluation_results.values():
            result.update(counts)
            result["target_indices"] = retained_indices
        print(
            f"Test points: evaluated={counts['evaluated_points']}, "
            f"excluded={counts['excluded_points_count']}, total={counts['total_points']}"
        )
        return {
            "method": self.method,
            **self.head_metadata,
            "split": "test",
            "target_indices": self.target_indices,
            "valid_prediction_mask": self.valid_mask,
            "excluded_points": self.excluded_points,
            "exclusion_policy": EXCLUSION_POLICY,
            "reporting_timeline": "successful_predictions_only",
            **counts,
        }


def summarize_crossing_results(log, pairs):
    """Match DistMatch's equal sequence weighting and unavailable-score policy."""
    summaries = {}
    for configured_pair in pairs:
        pair = tuple(configured_pair)
        eligible = {key: item for key, item in log.items()
                    if item["evaluation_results"][pair]["coverage"]}
        if eligible:
            result = summarize_evaluation_results(eligible, [pair])[pair]
        else:
            result = {f"avg_{metric}_{stat}": None
                      for metric in ("coverage", "delta_coverage", "interval_width", "winkler_score")
                      for stat in ("mean", "std")}
        total = sum(item["metadata"]["total_points"] for item in log.values())
        evaluated = sum(item["metadata"]["evaluated_points"] for item in log.values())
        result.update({
            "num_sequences": len(eligible),
            "num_total_sequences": len(log),
            "num_no_valid_sequences": len(log) - len(eligible),
            **_counts(total, evaluated),
        })
        summaries[pair] = result
    return summaries


# Preserve the existing QR runner API while sharing the identical policy with IQN.
summarize_qr_results = summarize_crossing_results


def write_excluded_points(log, saving_dir):
    """Publish a diagnostic row for each excluded timestamp, even for empty runs."""
    fields = (
        "sequence_key", "split", "test_offset", "target_index", "target_y",
        "target_prediction", "target_residual", "crossed_quantile_pairs",
        "excluded_quantile_pairs", "quantile_levels", "predicted_quantiles",
        "bound_scale", "reason", "action",
    )
    destination = Path(saving_dir) / "excluded_points.csv"
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sequence in log.values():
            for event in sequence["metadata"]["excluded_points"]:
                writer.writerow({name: json.dumps(value, allow_nan=False) if isinstance(value, list) else value
                                 for name, value in event.items()})
    return destination
