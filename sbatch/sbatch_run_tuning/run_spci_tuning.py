"""Tune SPCI on a held-out fraction of its nominal training prefix."""

from numbers import Integral
from pathlib import Path
import math
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from baselines.spci.model import build_quantile_forest
from baselines.spci.tuning_data import prepare_spci_tuning_data
from sbatch.sbatch_run_tuning.common import (
    aggregate_sequence_results,
    choose_sequence_keys,
    finalize_and_save_results,
    iter_grid_configs,
    load_grid,
    parse_args,
    plain_config,
    resolve_delta_threshold,
    resolve_num_sequences,
    set_global_seed,
    summarize_evaluation_results,
    write_trial_artifacts,
)
from utils.reporting import compute_coverage, compute_interval_width, compute_winkler_score
from utils.utils import get_interval_quantile_indices, load_data


ALLOWED_GRID_KEYS = frozenset({
    "model.window_size", "model.n_estimators", "model.max_depth", "model.criterion",
})
EVALUATION_SPLIT = "model_selection_valid"


def resolve_tuning_inputs(base_config_path, tuning_cfg, save_dir):
    """Resolve one predictor's configuration, artifact, and output directory."""
    if "base_predictor" in tuning_cfg:
        raise ValueError(
            "Move tuning.base_predictor to the top-level base_predictor field "
            "in the base experiment config supplied via --base-config."
        )
    base_config_path = Path(base_config_path).expanduser().resolve()
    config = OmegaConf.load(base_config_path)
    predictor = config.get("base_predictor")
    if predictor is not None:
        if not isinstance(predictor, str) or predictor.strip().lower() not in (
            "lr", "lstm", "chronos"
        ):
            raise ValueError("base_predictor must be one of 'lr', 'lstm', or 'chronos'.")
        predictor = predictor.strip().lower()
        config.base_predictor = predictor

    artifact_path = Path(str(config.data.data_path)).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = REPO_ROOT / artifact_path
    config.data.data_path = str(artifact_path.resolve())
    save_dir_text = str(save_dir)
    if "{base_predictor}" in save_dir_text:
        if predictor is None:
            predictor = artifact_path.stem.split("_", 1)[0]
        save_dir_text = save_dir_text.replace("{base_predictor}", predictor)
    return config, base_config_path, Path(save_dir_text).expanduser().resolve()


def _normalization_params(config, sequence_data):
    if not config.data.normalize:
        return None
    return sequence_data["train_residuals_mu"], sequence_data["train_residuals_std"]


def _run_single_trial(config, sequence_item, normalization_params):
    """Fit on the inner training prefix and score its later validation suffix."""
    train_dataset = sequence_item["train_dataset"]
    evaluation_dataset = sequence_item["model_selection_valid_dataset"]
    sorted_quantiles, pair_to_indices = get_interval_quantile_indices(
        config.model.target_quantiles
    )
    if len(sorted_quantiles) < 2 or any(not 0 < q < 1 for q in sorted_quantiles):
        raise ValueError("SPCI tuning requires interval quantiles strictly between 0 and 1.")
    train = train_dataset[:]
    evaluation = evaluation_dataset[:]
    forest = build_quantile_forest(config, len(train_dataset), np.asarray(sorted_quantiles))
    forest.fit(train[1].numpy(), train[4].flatten().numpy())
    quantiles = torch.as_tensor(
        forest.predict(evaluation[1].numpy()), dtype=torch.float32
    )
    if quantiles.shape != (len(sorted_quantiles), len(evaluation_dataset)):
        raise ValueError("The quantile forest returned an unexpected prediction shape.")
    if not torch.isfinite(quantiles).all():
        raise ValueError("The quantile forest returned nonfinite validation predictions.")

    residual_std = normalization_params[1] if normalization_params is not None else None
    evaluation_results = {}
    for pair in config.model.target_quantiles:
        pair_key = tuple(pair)
        hi_idx, lo_idx = pair_to_indices[pair_key]
        hi, lo = quantiles[hi_idx], quantiles[lo_idx]
        metrics = {
            "coverage": compute_coverage(hi, lo, evaluation[4]),
            "interval_width": compute_interval_width(hi, lo, normalized_std=residual_std),
            "winkler_score": compute_winkler_score(
                hi, lo, evaluation[5], evaluation[6], pair_key,
                normalized_params=normalization_params,
            ),
        }
        if any(not np.isfinite(values).all() for values in metrics.values()):
            raise ValueError("SPCI validation metrics must be finite.")
        evaluation_results[pair_key] = metrics
    pair_metrics, selection_score, eligible = summarize_evaluation_results(
        evaluation_results, config.model.target_quantiles,
        delta_threshold=float(config.tuning.delta_threshold),
    )
    return {
        "estimator": type(forest).__name__,
        "seed": int(config.seed),
        "num_train_samples": len(train_dataset),
        "num_model_selection_valid_samples": len(evaluation_dataset),
        "num_tuning_evaluation_samples": len(evaluation_dataset),
        "evaluation_split": EVALUATION_SPLIT,
        "nominal_validation_evaluated": False,
        "final_test_evaluated": False,
        "pair_metrics": pair_metrics,
        "selection_score": selection_score,
        "positive_delta_coverage": eligible,
    }


def _run_grid_trial(
    trial_index, trial_config, grid_values, selected_data, sequence_keys, seed,
    prepared_data_cache,
):
    effective_seed = seed + trial_index
    set_global_seed(effective_seed)
    trial_config.seed = effective_seed
    print(f"[spci] trial {trial_index}: seed={effective_seed}, grid={grid_values}", flush=True)
    data_cache_key = (
        trial_config.model.window_size,
        trial_config.model.prediction_step,
        trial_config.data.train_ratio,
        trial_config.data.valid_ratio,
        trial_config.tuning.model_selection_valid_ratio,
        trial_config.data.normalize,
    )
    prepared = prepared_data_cache.get(data_cache_key)
    if prepared is None:
        prepared = prepare_spci_tuning_data(selected_data, trial_config)
        prepared_data_cache[data_cache_key] = prepared
    sequence_results = {
        key: _run_single_trial(
            trial_config, prepared.dataset[key],
            _normalization_params(trial_config, prepared.data[key]),
        )
        for key in sequence_keys
    }
    result = aggregate_sequence_results(sequence_results, trial_config.model.target_quantiles)
    result.update({
        "evaluation_split": EVALUATION_SPLIT,
        "nominal_validation_evaluated": False,
        "final_test_evaluated": False,
    })
    return {
        "trial_index": trial_index,
        "seed": effective_seed,
        "sequence_keys": sequence_keys,
        "grid_values": grid_values,
        "evaluation_split": EVALUATION_SPLIT,
        "nominal_validation_evaluated": False,
        "final_test_evaluated": False,
        "result": result,
        "resolved_config": plain_config(trial_config),
    }


def run_tuning(
    base_config_path, grid_config_path, save_dir, *, sequence_key=None,
    sequence_index=0, top_k=3, seed=42,
):
    """Evaluate the grid and save QR-CP-compatible trial records and rankings."""
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    grid_config_path = Path(grid_config_path).expanduser().resolve()
    grid, tuning_cfg = load_grid(grid_config_path)
    unexpected = sorted(set(grid) - ALLOWED_GRID_KEYS)
    if unexpected:
        raise ValueError(
            f"Unsupported SPCI grid keys: {unexpected}. Allowed keys: {sorted(ALLOWED_GRID_KEYS)}. "
            "Keep the data, splits, target quantiles, and horizon fixed during tuning."
        )
    num_trials = math.prod(len(values) for values in grid.values())
    if seed + num_trials >= 2**32:
        raise ValueError("seed plus the number of grid trials must be less than 2**32.")
    config, base_config_path, save_dir = resolve_tuning_inputs(
        base_config_path, tuning_cfg, save_dir
    )
    num_sequences = resolve_num_sequences(tuning_cfg)
    delta_threshold = resolve_delta_threshold(tuning_cfg)
    if not math.isfinite(delta_threshold):
        raise ValueError("tuning.delta_threshold must be finite.")
    ratio = tuning_cfg.get("model_selection_valid_ratio", 0.2)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 < ratio < 1:
        raise ValueError("tuning.model_selection_valid_ratio must be finite and strictly between 0 and 1.")
    config.tuning = dict(tuning_cfg)
    config.tuning.model_selection_valid_ratio = float(ratio)
    config.tuning.delta_threshold = delta_threshold

    data = load_data(config.data.data_path)
    sequence_keys = choose_sequence_keys(data, sequence_key, sequence_index, num_sequences)
    selected_data = {key: data[key] for key in sequence_keys}
    print(f"[spci] base configuration: {base_config_path}", flush=True)
    print(f"[spci] prediction artifact: {config.data.data_path}", flush=True)
    print(f"[spci] tuning on the last {ratio:.1%} of the nominal training prefix", flush=True)
    print(f"[spci] {num_trials} trials on {len(sequence_keys)} sequences; output: {save_dir}", flush=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    prepared_data_cache = {}
    trials = []
    for trial_index, (trial_config, grid_values) in enumerate(iter_grid_configs(config, grid), start=1):
        # Each exported configuration can be passed directly to the ordinary runner.
        trial_config.saving_dir = str(save_dir / f"trial_{trial_index:04d}" / "final_run")
        record = _run_grid_trial(
            trial_index, trial_config, grid_values, selected_data, sequence_keys,
            int(seed), prepared_data_cache,
        )
        trials.append(record)
        write_trial_artifacts(
            save_dir, trial_index, OmegaConf.create(record["resolved_config"]), record
        )
    eligible_trials = [trial for trial in trials if trial["result"]["positive_delta_coverage"]]
    ranked_trials = sorted(
        eligible_trials, key=lambda trial: (trial["result"]["selection_score"], trial["trial_index"])
    )
    payload = {
        "method": "spci",
        "base_config_path": str(base_config_path),
        "grid_config_path": str(grid_config_path),
        "sequence_keys": sequence_keys,
        "num_sequences": len(sequence_keys),
        "seed": int(seed),
        "delta_threshold": delta_threshold,
        "num_trials": len(trials),
        "num_positive_delta_coverage_trials": len(eligible_trials),
        "top_k": top_k,
        "evaluation_split": EVALUATION_SPLIT,
        "nominal_validation_evaluated": False,
        "final_test_evaluated": False,
        "execution": {"mode": "serial", "forest_n_jobs": config.model.get("n_jobs", -1)},
        "tuning_protocol": {
            "model_fit_dataset": "train_dataset",
            "hyperparameter_evaluation_dataset": "model_selection_valid_dataset",
            "evaluation_split": EVALUATION_SPLIT,
            "model_selection_valid_ratio": float(ratio),
            "nominal_validation_evaluated": False,
            "final_test_evaluated": False,
        },
        "top_trials": ranked_trials[:top_k],
        "all_trials": trials,
    }
    finalize_and_save_results(save_dir, payload)
    print(f"[spci] {len(eligible_trials)}/{len(trials)} trials passed the coverage filter.", flush=True)
    for rank, trial in enumerate(payload["top_trials"], start=1):
        print(
            f"[spci] rank {rank}: trial {trial['trial_index']:04d}, "
            f"Winkler={trial['result']['selection_score']:.6g}", flush=True,
        )
    return payload


def main():
    args = parse_args("spci")
    run_tuning(
        args.base_config, args.grid_config, args.save_dir,
        sequence_key=args.sequence_key, sequence_index=args.sequence_index,
        top_k=args.top_k, seed=args.seed,
    )


if __name__ == "__main__":
    main()
