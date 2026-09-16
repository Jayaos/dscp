import copy
from contextlib import closing
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dscp.data import ConformalPredictionData
from dscp.loss import compute_loss_rnn_predictor, compute_loss_transformer_predictor
from dscp.models.local_cp import LocalConformalPrediction
from dscp.models.rnn_predictor import RNNPredictor
from dscp.models.transformer_predictor import TransformerPredictor
from sbatch_run_tuning.common import (
    aggregate_sequence_results,
    choose_sequence_keys,
    finalize_and_save_results,
    iter_grid_configs,
    load_grid,
    parse_args,
    plain_config,
    resolve_delta_threshold,
    resolve_device,
    resolve_num_gpus,
    resolve_num_sequences,
    resolve_worker_devices,
    set_global_seed,
    summarize_evaluation_results,
    write_trial_artifacts,
)
from sbatch_run_tuning.gpu_trial_pool import iter_parallel_trials
from utils.reporting import compute_coverage, compute_interval_width, compute_winkler_score
from utils.utils import (
    generate_strided_feature, get_interval_quantile_indices, load_data,
    validate_rolling_calibration,
)


def resolve_tuning_inputs(base_config_path, tuning_cfg, save_dir):
    """Load the selected experiment and resolve its tuning output directory."""
    if "base_predictor" in tuning_cfg:
        raise ValueError(
            "Move tuning.base_predictor to the top-level base_predictor field in "
            "the base experiment config supplied via --base-config."
        )

    resolved_base_config_path = Path(base_config_path).resolve()
    base_config = OmegaConf.load(base_config_path)
    predictor = None

    if "base_predictor" in base_config:
        predictor = base_config.base_predictor
        if not isinstance(predictor, str) or predictor.strip().lower() not in (
            "chronos", "lr", "lstm"
        ):
            raise ValueError(
                "base_predictor in the base experiment config must be one scalar "
                "string: 'chronos', 'lr', or 'lstm'."
            )
        predictor = predictor.strip().lower()
        base_config.base_predictor = predictor

    save_dir_text = str(save_dir)
    if "{base_predictor}" in save_dir_text:
        if predictor is None:
            predictor = Path(str(base_config.data.data_path)).stem.split("_", 1)[0]
        save_dir_text = save_dir_text.replace("{base_predictor}", predictor)
    return base_config, resolved_base_config_path, Path(save_dir_text).resolve()


def _build_model(config, dim_feature: int, dim_x: int):
    use_current_feature = bool(config.model.use_current_feature)
    current_feature_dim = dim_x if use_current_feature else 0

    if "rnn_type" in config.model:
        model = RNNPredictor(
            rnn_type=config.model.rnn_type,
            dim_feature=dim_feature,
            dim_model=config.model.dim_model,
            num_layer=config.model.num_layers,
            prediction_step=config.model.prediction_step,
            current_feature_dim=current_feature_dim,
            dropout=config.model.dropout,
            training_quantiles=OmegaConf.select(config, "model.training_quantiles"),
        )
        loss_fn = compute_loss_rnn_predictor
        model_type = "rnn"
    else:
        model = TransformerPredictor(
            dim_feature=dim_feature,
            dim_model=config.model.dim_model,
            num_head=config.model.num_heads,
            dim_ff=config.model.dim_model * 4,
            num_layer=config.model.num_layers,
            prediction_step=config.model.prediction_step,
            current_feature_dim=current_feature_dim,
            dropout=config.model.dropout,
            training_quantiles=OmegaConf.select(config, "model.training_quantiles"),
        )
        loss_fn = compute_loss_transformer_predictor
        model_type = "transformer"
    return model, loss_fn, model_type, use_current_feature


def _model_selection_valid_ratio(config) -> float:
    return float(config.tuning.get("model_selection_valid_ratio", 0.2))


def _normalization_params(config, sequence_data):
    if not bool(config.data.normalize):
        return None
    return (
        sequence_data["train_residuals_mu"],
        sequence_data["train_residuals_std"],
    )


def _prepared_data_cache_key(config):
    return (
        int(config.model.window_size),
        int(config.model.prediction_step),
        float(config.data.train_ratio),
        float(config.data.valid_ratio),
        float(config.data.calibration_ratio),
        float(config.data.test_ratio),
        bool(config.data.normalize),
        _model_selection_valid_ratio(config),
    )


def _prepare_trial_data(raw_data, config):
    cpd = ConformalPredictionData(copy.deepcopy(raw_data))
    cpd.prepare_quantile_regression_datasets(
        config.model.window_size,
        config.model.prediction_step,
        config.data.train_ratio,
        config.data.valid_ratio,
        normalize=config.data.normalize,
        calibration_ratio=config.data.calibration_ratio,
        test_ratio=config.data.test_ratio,
        model_selection_valid_ratio=_model_selection_valid_ratio(config),
    )
    return cpd


def _run_single_trial(config, sequence_item, normalization_params):
    rolling_calibration = validate_rolling_calibration(
        OmegaConf.select(config, "model.rolling_calibration", default=True)
    )
    device = resolve_device(config.device)
    sorted_quantiles, pair_to_indices = get_interval_quantile_indices(config.model.target_quantiles)
    delta_threshold = float(config.tuning.get("delta_threshold", 0.0))

    train_dataset = sequence_item["train_dataset"]
    model_selection_valid_dataset = sequence_item["model_selection_valid_dataset"]
    calibration_dataset = sequence_item["calibration_dataset"]
    tuning_evaluation_dataset = sequence_item["tuning_evaluation_dataset"]

    train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    model_selection_valid_dataloader = DataLoader(
        model_selection_valid_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
    )
    tuning_evaluation_dataloader = DataLoader(
        tuning_evaluation_dataset,
        batch_size=1,
        shuffle=False,
    )

    dim_x = train_dataset.strided_x.shape[-1]
    if config.data.strided_features == "xry":
        dim_feature = dim_x + 2
    elif config.data.strided_features == "xr":
        dim_feature = dim_x + 1
    elif config.data.strided_features == "r":
        dim_feature = 1
    else:
        raise ValueError("wrong strided features specified")

    model, loss_fn, model_type, use_current_feature = _build_model(config, dim_feature, dim_x)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    train_loss = []
    model_selection_valid_loss = []
    best_loss = np.inf
    best_epoch = 0
    best_model = copy.deepcopy(model.state_dict())

    for epoch_idx in range(config.training.epochs):
        model.train()
        loss_sum = 0.0
        for strided_x, strided_residual, strided_y, target_x, target_residual, _, _ in train_dataloader:
            optimizer.zero_grad()
            strided_feature = generate_strided_feature(
                strided_x,
                strided_residual,
                strided_y,
                config.data.strided_features,
            ).to(device)
            target_residual = target_residual.to(device)
            target_x = target_x.to(device)
            if use_current_feature:
                loss = loss_fn(model, strided_feature, target_residual, target_x)
            else:
                loss = loss_fn(model, strided_feature, target_residual)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        epoch_train_loss = loss_sum / len(train_dataloader)
        train_loss.append(float(epoch_train_loss))

        model.eval()
        loss_sum = 0.0
        for strided_x, strided_residual, strided_y, target_x, target_residual, _, _ in model_selection_valid_dataloader:
            strided_feature = generate_strided_feature(
                strided_x,
                strided_residual,
                strided_y,
                config.data.strided_features,
            ).to(device)
            target_residual = target_residual.to(device)
            target_x = target_x.to(device)
            with torch.no_grad():
                if use_current_feature:
                    loss = loss_fn(model, strided_feature, target_residual, target_x)
                else:
                    loss = loss_fn(model, strided_feature, target_residual)
            loss_sum += loss.item()

        epoch_valid_loss = loss_sum / len(model_selection_valid_dataloader)
        model_selection_valid_loss.append(float(epoch_valid_loss))

        if epoch_valid_loss < best_loss:
            best_loss = epoch_valid_loss
            best_epoch = epoch_idx + 1
            best_model = copy.deepcopy(model.state_dict())

        if config.training.early_stop and (epoch_idx + 1 - best_epoch) >= config.training.early_stop:
            break

    model.load_state_dict(best_model)
    evaluation_results = {
        tuple(confidence_pair): {
            "coverage": [],
            "interval_width": [],
            "winkler_score": [],
        }
        for confidence_pair in config.model.target_quantiles
    }

    if normalization_params is not None:
        residual_mu, residual_std = normalization_params

    calibration_dataloader = DataLoader(
        calibration_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    model.eval()
    with torch.no_grad():
        calibration_repr, calibration_residual = model.encode_dataloader(
            model,
            calibration_dataloader,
            config.data.strided_features,
            device,
            use_current_feature=use_current_feature,
        )

    configured_calibration_size = OmegaConf.select(config, "model.calibration_size", default=None)
    if configured_calibration_size is None:
        calibration_pool_size = len(calibration_dataset)
    else:
        calibration_pool_size = min(int(configured_calibration_size), len(calibration_dataset))
        if calibration_pool_size <= 0:
            raise ValueError("model.calibration_size must be positive when it is provided.")

    # Keep the calibration state on CPU. LocalConformalPrediction
    # transfers it to the configured device for each similarity computation.
    calibration_repr = calibration_repr[-calibration_pool_size:].detach().cpu()
    calibration_residual = calibration_residual[-calibration_pool_size:].detach().cpu()

    for strided_x, strided_residual, strided_y, target_x, target_residual, target_y, target_predictions in tuning_evaluation_dataloader:
        with torch.no_grad():
            strided_feature = generate_strided_feature(
                strided_x,
                strided_residual,
                strided_y,
                config.data.strided_features,
            ).to(device)
            target_x = target_x.to(device)
            encoded = model.encode(
                model,
                strided_feature,
                current_feature=target_x if use_current_feature else None,
            )
            query_repr = encoded[:, -1, :]

        local_cp = LocalConformalPrediction(
            calibration_repr,
            calibration_residual,
            config.model.similarity_fn,
            config.model.temperature,
            device,
        )
        pred_quantile_values = local_cp.approximate_quantile(query_repr, sorted_quantiles, config.model.sampling_num)

        if device.type != "cpu":
            pred_quantile_values = pred_quantile_values.cpu().detach()

        for confidence_pair in config.model.target_quantiles:
            pair_key = tuple(confidence_pair)
            hi_idx, lo_idx = pair_to_indices[pair_key]
            hi = pred_quantile_values[hi_idx, :]
            lo = pred_quantile_values[lo_idx, :]
            evaluation_results[pair_key]["coverage"].extend(compute_coverage(hi, lo, target_residual))
            evaluation_results[pair_key]["interval_width"].extend(
                compute_interval_width(
                    hi,
                    lo,
                    normalized_std=residual_std if normalization_params is not None else None,
                )
            )
            evaluation_results[pair_key]["winkler_score"].extend(
                compute_winkler_score(
                    hi,
                    lo,
                    target_y,
                    target_predictions,
                    pair_key,
                    normalized_params=(residual_mu, residual_std)
                    if normalization_params is not None
                    else None,
                )
            )

        # Optional updates happen after scoring; otherwise the initial
        # calibration pairs remain fixed throughout tuning evaluation.
        if rolling_calibration:
            calibration_repr = torch.vstack(
                [calibration_repr, query_repr.detach().cpu()]
            )[-calibration_pool_size:]
            calibration_residual = torch.vstack(
                [calibration_residual, target_residual.detach().cpu().reshape(-1, 1)]
            )[-calibration_pool_size:]

    pair_metrics, selection_score, positive_delta_coverage = summarize_evaluation_results(
        evaluation_results,
        config.model.target_quantiles,
        delta_threshold=delta_threshold,
    )

    return {
        "model_type": model_type,
        "training_quantiles": model.training_quantiles.detach().cpu().tolist(),
        "rolling_calibration": rolling_calibration,
        "train_loss": train_loss,
        "model_fit_train_loss": train_loss,
        "model_selection_valid_loss": model_selection_valid_loss,
        # Backward-compatible aliases consumed by shared aggregation/reporting.
        "valid_loss": model_selection_valid_loss,
        "best_model_selection_valid_loss": float(best_loss),
        "best_valid_loss": float(best_loss),
        "best_epoch": best_epoch,
        "calibration_split": "nominal_validation",
        "evaluation_split": "nominal_calibration",
        "final_test_evaluated": False,
        "num_train_samples": len(train_dataset),
        "num_model_selection_valid_samples": len(model_selection_valid_dataset),
        "num_calibration_samples": len(calibration_dataset),
        "num_initial_calibration_pool_samples": calibration_pool_size,
        "num_tuning_evaluation_samples": len(tuning_evaluation_dataset),
        "sample_counts": {
            "train": len(train_dataset),
            "model_selection_valid": len(model_selection_valid_dataset),
            "calibration": len(calibration_dataset),
            "initial_calibration_pool": calibration_pool_size,
            "tuning_evaluation": len(tuning_evaluation_dataset),
        },
        "dataset_roles": {
            "model_fit": "train_dataset",
            "checkpoint_selection": "model_selection_valid_dataset",
            "conformal_calibration": "calibration_dataset",
            "hyperparameter_evaluation": "tuning_evaluation_dataset",
        },
        "pair_metrics": pair_metrics,
        "selection_score": selection_score,
        "positive_delta_coverage": positive_delta_coverage,
    }


def _run_grid_trial(
    trial_index, trial_config, grid_values, selected_data, sequence_keys, seed, prepared_data_cache
):
    """Evaluate one configuration on every selected sequence in the same order."""
    print(
        f"[local_cp] starting trial {trial_index} on device={trial_config.get('device', 'default')} "
        f"with grid_values={grid_values}",
        flush=True,
    )
    set_global_seed(seed + trial_index)
    data_cache_key = _prepared_data_cache_key(trial_config)
    trial_cpd = prepared_data_cache.get(data_cache_key)
    if trial_cpd is None:
        trial_cpd = _prepare_trial_data(selected_data, trial_config)
        prepared_data_cache[data_cache_key] = trial_cpd
    sequence_results = {
        sequence_key: _run_single_trial(
            trial_config,
            trial_cpd.dataset[sequence_key],
            _normalization_params(trial_config, trial_cpd.data[sequence_key]),
        )
        for sequence_key in sequence_keys
    }
    result = aggregate_sequence_results(sequence_results, trial_config.model.target_quantiles)
    result["mean_best_model_selection_valid_loss"] = result["mean_best_valid_loss"]
    result["calibration_split"] = "nominal_validation"
    result["evaluation_split"] = "nominal_calibration"
    result["final_test_evaluated"] = False
    result["dataset_roles"] = {
        "model_fit": "train_dataset",
        "checkpoint_selection": "model_selection_valid_dataset",
        "conformal_calibration": "calibration_dataset",
        "hyperparameter_evaluation": "tuning_evaluation_dataset",
    }
    return {
        "trial_index": trial_index,
        "sequence_keys": sequence_keys,
        "grid_values": grid_values,
        "calibration_split": "nominal_validation",
        "evaluation_split": "nominal_calibration",
        "final_test_evaluated": False,
        "result": result,
        "resolved_config": plain_config(trial_config),
    }


_worker_device = None
_worker_selected_data = None
_worker_sequence_keys = None
_worker_prepared_data_cache = None


def _initialize_gpu_worker(device, selected_data, sequence_keys):
    """Initialize one spawned worker and retain its CPU datasets between trials."""
    global _worker_device, _worker_selected_data, _worker_sequence_keys, _worker_prepared_data_cache
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    _worker_device = device
    _worker_selected_data = selected_data
    _worker_sequence_keys = sequence_keys
    _worker_prepared_data_cache = {}


def _run_gpu_trial(task):
    if _worker_device is None:
        raise RuntimeError("The GPU trial worker has not been initialized.")
    trial_index, config_values, grid_values, seed = task
    trial_config = OmegaConf.create(config_values)
    configured_device = trial_config.device
    trial_config.device = _worker_device
    record = _run_grid_trial(
        trial_index, trial_config, grid_values, _worker_selected_data,
        _worker_sequence_keys, seed, _worker_prepared_data_cache,
    )
    # Saved configurations remain reusable on a later single-GPU allocation.
    record["resolved_config"]["device"] = configured_device
    record["worker_device"] = _worker_device
    return record


def main():
    args = parse_args("local_cp")
    grid, tuning_cfg = load_grid(args.grid_config)
    base_config, base_config_path, save_dir = resolve_tuning_inputs(
        args.base_config, tuning_cfg, args.save_dir
    )
    num_sequences = resolve_num_sequences(tuning_cfg)
    delta_threshold = resolve_delta_threshold(tuning_cfg)
    num_gpus = resolve_num_gpus(tuning_cfg, getattr(args, "num_gpus", None))
    worker_devices = resolve_worker_devices(base_config, grid, num_gpus)
    base_config.tuning = dict(tuning_cfg)
    base_config.tuning.num_gpus = num_gpus

    print(f"[local_cp] base configuration: {base_config_path}", flush=True)
    print(f"[local_cp] prediction data: {base_config.data.data_path}", flush=True)
    print(f"[local_cp] saving results to: {save_dir}", flush=True)
    if worker_devices:
        print(f"[local_cp] parallel trial workers: {worker_devices}", flush=True)
    else:
        print("[local_cp] running trials sequentially", flush=True)
    data = load_data(base_config.data.data_path)
    sequence_keys = choose_sequence_keys(data, args.sequence_key, args.sequence_index, num_sequences)
    selected_data = {sequence_key: data[sequence_key] for sequence_key in sequence_keys}
    save_dir.mkdir(parents=True, exist_ok=True)

    if worker_devices:
        tasks = [
            (trial_index, plain_config(trial_config), grid_values, args.seed)
            for trial_index, (trial_config, grid_values)
            in enumerate(iter_grid_configs(base_config, grid), start=1)
        ]
        trial_records = iter_parallel_trials(
            tasks, worker_devices, _run_gpu_trial,
            initializer=_initialize_gpu_worker, initargs=(selected_data, sequence_keys),
        )
    else:
        prepared_data_cache = {}
        trial_records = (
            _run_grid_trial(
                trial_index, trial_config, grid_values, selected_data,
                sequence_keys, args.seed, prepared_data_cache,
            )
            for trial_index, (trial_config, grid_values)
            in enumerate(iter_grid_configs(base_config, grid), start=1)
        )

    trials = []
    # A failed artifact write must also stop any workers that are still running.
    with closing(trial_records):
        for record in trial_records:
            trials.append(record)
            write_trial_artifacts(
                save_dir, record["trial_index"], OmegaConf.create(record["resolved_config"]), record
            )

    # Completion order must not affect saved order or tie-breaking during ranking.
    trials.sort(key=lambda record: record["trial_index"])

    positive_trials = [trial for trial in trials if trial["result"]["positive_delta_coverage"]]
    ranked_trials = sorted(positive_trials, key=lambda item: item["result"]["selection_score"])
    top_trials = ranked_trials[: args.top_k]

    payload = {
        "method": "local_cp",
        "base_config_path": str(base_config_path),
        "grid_config_path": str(args.grid_config.resolve()),
        "sequence_keys": sequence_keys,
        "num_sequences": len(sequence_keys),
        "delta_threshold": delta_threshold,
        "num_trials": len(trials),
        "num_positive_delta_coverage_trials": len(positive_trials),
        "top_k": args.top_k,
        "calibration_split": "nominal_validation",
        "evaluation_split": "nominal_calibration",
        "final_test_evaluated": False,
        "execution": {
            "mode": "parallel_trials" if worker_devices else "serial",
            "num_gpus": num_gpus,
            "worker_devices": worker_devices,
        },
        "tuning_protocol": {
            "model_fit_dataset": "train_dataset",
            "checkpoint_selection_dataset": "model_selection_valid_dataset",
            "conformal_calibration_dataset": "calibration_dataset",
            "hyperparameter_evaluation_dataset": "tuning_evaluation_dataset",
            "model_fit_split": "early_nominal_train",
            "checkpoint_selection_split": "late_nominal_train",
            "calibration_split": "nominal_validation",
            "evaluation_split": "nominal_calibration",
            "model_selection_valid_ratio": _model_selection_valid_ratio(base_config),
            "final_test_split": "nominal_test",
            "final_test_exposed": False,
            "final_test_evaluated": False,
        },
        "top_trials": top_trials,
        "all_trials": trials,
    }
    finalize_and_save_results(save_dir, payload)


if __name__ == "__main__":
    main()
