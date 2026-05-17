import argparse
import itertools
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from utils.utils import save_data


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(method_name: str, include_num_cores: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Run hyperparameter tuning for {method_name}. "
            "Candidates are filtered by positive avg_delta_coverage and ranked by avg_winkler_score."
        )
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="Base experiment config YAML used as the template for each trial.",
    )
    parser.add_argument(
        "--grid-config",
        type=Path,
        required=True,
        help=(
            "Grid config YAML. Expected format: "
            "grid: {model.dim_model: [32, 64], training.learning_rate: [0.001, 0.0003]}"
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="Directory where tuning artifacts will be written.",
    )
    parser.add_argument(
        "--sequence-key",
        type=str,
        default=None,
        help="Explicit sequence key to tune on. If omitted, --sequence-index is used.",
    )
    parser.add_argument(
        "--sequence-index",
        type=int,
        default=0,
        help="Sorted sequence index used when --sequence-key is not provided.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top-ranked positive-coverage configs to keep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed. Trial index is added to this value.",
    )
    if include_num_cores:
        parser.add_argument(
            "--num-cores",
            type=int,
            default=1,
            help=(
                "Number of parallel sequence workers to use inside each grid trial. "
                "Grid trials are still evaluated sequentially."
            ),
        )
    return parser.parse_args()


def load_grid(grid_path: Path) -> Tuple[dict, dict]:
    grid_cfg = OmegaConf.load(grid_path)
    grid = OmegaConf.to_container(grid_cfg.get("grid", {}), resolve=True)
    if not isinstance(grid, dict) or len(grid) == 0:
        raise ValueError("Grid config must define a non-empty `grid:` mapping.")

    normalized_grid = {}
    for dotted_key, values in grid.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Grid entry `{dotted_key}` must map to a non-empty list.")
        normalized_grid[dotted_key] = values
    tuning_cfg = OmegaConf.to_container(grid_cfg.get("tuning", {}), resolve=True)
    if tuning_cfg is None:
        tuning_cfg = {}
    if not isinstance(tuning_cfg, dict):
        raise ValueError("Grid config `tuning:` section must be a mapping when provided.")

    return normalized_grid, tuning_cfg


def iter_grid_configs(base_config, grid: dict):
    keys = list(grid.keys())
    value_lists = [grid[key] for key in keys]

    for values in itertools.product(*value_lists):
        cfg = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
        updates = {}
        for dotted_key, value in zip(keys, values):
            _set_dotted_key(cfg, dotted_key, value)
            updates[dotted_key] = value
        yield cfg, updates


def _set_dotted_key(cfg, dotted_key: str, value):
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node:
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def choose_sequence_key(dataset: dict, sequence_key: Optional[str], sequence_index: int) -> str:
    keys = sorted(dataset.keys())
    if not keys:
        raise ValueError("No sequence keys were found in the prepared dataset.")

    if sequence_key is not None:
        if sequence_key not in dataset:
            raise ValueError(
                f"Sequence key `{sequence_key}` was not found. Available keys: {keys}"
            )
        return sequence_key

    if not (0 <= sequence_index < len(keys)):
        raise ValueError(
            f"sequence_index={sequence_index} is out of range for {len(keys)} sequences."
        )
    return keys[sequence_index]


def resolve_num_sequences(tuning_cfg) -> int:
    if tuning_cfg is None:
        tuning_cfg = {}
    num_sequences = int(tuning_cfg.get("num_sequences", 1))
    if num_sequences <= 0:
        raise ValueError("tuning.num_sequences must be a positive integer.")
    return num_sequences


def resolve_delta_threshold(tuning_cfg) -> float:
    if tuning_cfg is None:
        tuning_cfg = {}
    return float(tuning_cfg.get("delta_threshold", 0.0))


def choose_sequence_keys(
    dataset: dict,
    sequence_key: Optional[str],
    sequence_index: int,
    num_sequences: int,
) -> List[str]:
    keys = sorted(dataset.keys())
    if not keys:
        raise ValueError("No sequence keys were found in the prepared dataset.")

    if sequence_key is not None:
        if sequence_key not in dataset:
            raise ValueError(
                f"Sequence key `{sequence_key}` was not found. Available keys: {keys}"
            )
        return [sequence_key]

    if not (0 <= sequence_index < len(keys)):
        raise ValueError(
            f"sequence_index={sequence_index} is out of range for {len(keys)} sequences."
        )

    end_index = sequence_index + num_sequences
    if end_index > len(keys):
        raise ValueError(
            f"Requested {num_sequences} sequences starting at index {sequence_index}, "
            f"but only {len(keys) - sequence_index} are available."
        )

    return keys[sequence_index:end_index]


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw_device):
    if isinstance(raw_device, str):
        lowered = raw_device.lower()
        if lowered == "cpu":
            return torch.device("cpu")
        if lowered.startswith("cuda"):
            return torch.device(raw_device)
        if lowered.isdigit():
            raw_device = int(lowered)
        else:
            return torch.device(raw_device)

    if isinstance(raw_device, int):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{raw_device}")
        return torch.device("cpu")

    return torch.device("cpu")


def summarize_evaluation_results(
    evaluation_results: dict, target_quantiles: list, delta_threshold: float = 0.0
) -> Tuple[Dict, float, bool]:
    pair_summaries = {}
    winkler_scores = []
    all_positive = True

    for confidence_pair in target_quantiles:
        pair_key = tuple(confidence_pair)
        result = evaluation_results[pair_key]
        avg_coverage = float(np.mean(result["coverage"]))
        target_coverage = float(max(pair_key) - min(pair_key))
        avg_delta_coverage = avg_coverage - target_coverage
        avg_interval_width = float(np.mean(result["interval_width"]))
        avg_winkler_score = float(np.mean(result["winkler_score"]))

        pair_summaries[str(pair_key)] = {
            "avg_coverage": avg_coverage,
            "target_coverage": target_coverage,
            "avg_delta_coverage": float(avg_delta_coverage),
            "avg_interval_width": avg_interval_width,
            "avg_winkler_score": avg_winkler_score,
        }

        winkler_scores.append(avg_winkler_score)
        all_positive = all_positive and (avg_delta_coverage > delta_threshold)

    selection_score = float(np.mean(winkler_scores))
    return pair_summaries, selection_score, all_positive


def write_trial_artifacts(
    save_dir: Path,
    trial_index: int,
    config,
    record: dict,
):
    trial_dir = save_dir / f"trial_{trial_index:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=trial_dir / "resolved_config.yaml")
    save_data(trial_dir / "result.pkl", record)


def finalize_and_save_results(save_dir: Path, payload: dict):
    save_dir.mkdir(parents=True, exist_ok=True)
    save_data(save_dir / "tuning_results.pkl", payload)


def plain_config(config) -> dict:
    return OmegaConf.to_container(config, resolve=True)


def aggregate_sequence_results(sequence_results: dict, target_quantiles: list) -> dict:
    if not sequence_results:
        raise ValueError("No sequence results were provided for aggregation.")

    ordered_results = list(sequence_results.values())
    pair_metrics = {}
    for confidence_pair in target_quantiles:
        pair_key = str(tuple(confidence_pair))
        pair_metrics[pair_key] = {
            "avg_coverage": float(np.mean([item["pair_metrics"][pair_key]["avg_coverage"] for item in ordered_results])),
            "target_coverage": float(
                np.mean([item["pair_metrics"][pair_key]["target_coverage"] for item in ordered_results])
            ),
            "avg_delta_coverage": float(
                np.mean([item["pair_metrics"][pair_key]["avg_delta_coverage"] for item in ordered_results])
            ),
            "avg_interval_width": float(
                np.mean([item["pair_metrics"][pair_key]["avg_interval_width"] for item in ordered_results])
            ),
            "avg_winkler_score": float(
                np.mean([item["pair_metrics"][pair_key]["avg_winkler_score"] for item in ordered_results])
            ),
        }

    return {
        "num_sequences_evaluated": len(sequence_results),
        "sequence_results": sequence_results,
        "mean_best_valid_loss": float(np.mean([item["best_valid_loss"] for item in ordered_results])),
        "mean_best_epoch": float(np.mean([item["best_epoch"] for item in ordered_results])),
        "pair_metrics": pair_metrics,
        "selection_score": float(np.mean([item["selection_score"] for item in ordered_results])),
        "positive_delta_coverage": all(item["positive_delta_coverage"] for item in ordered_results),
    }
