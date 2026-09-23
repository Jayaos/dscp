import copy
from contextlib import closing
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dscp.data import ConformalPredictionData
from dscp.loss import (
    compute_iqn_interval_validation_loss,
    compute_loss_iqn_rnn,
    compute_loss_iqn_transformer,
    resolve_iqn_training_quantiles,
    resolve_iqn_validation_quantiles,
    warn_iqn_training_interval_mismatch,
)
from dscp.models.iqn import build_iqn_optimizer
from dscp.models.iqn_rnn import IQNRNN
from dscp.models.iqn_transformer import IQNTransformer
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
from utils.utils import generate_strided_feature, get_interval_quantile_indices, load_data


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


def _prediction_head_kwargs(model_config):
    """Resolve IQN prediction-head options for a tuning trial."""
    return {
        "prediction_head": model_config.get("prediction_head", "cosine_embedding"),
        "iqn_num_layers": model_config.get("iqn_num_layers", 1),
        "interval_mode": model_config.get("interval_mode", "sampling"),
        "sampling_num": model_config.get("sampling_num", 1000),
        "monotonic_num_layers": model_config.get("monotonic_num_layers", 1),
        "monotonic_hidden_dims": model_config.get("monotonic_hidden_dims"),
        "monotonic_activation": model_config.get(
            "monotonic_activation",
            "tanh",
        ),
    }


def _build_model(config, dim_feature: int, dim_x: int):
    config.model.prediction_head = str(
        config.model.get("prediction_head", "cosine_embedding")
    ).strip().lower()
    config.model.interval_mode = str(
        config.model.get("interval_mode", "sampling")
    ).strip().lower()
    config.model.sampling_num = config.model.get("sampling_num", 1000)
    config.model.iqn_num_layers = config.model.get("iqn_num_layers", 1)
    use_current_feature = bool(config.model.use_current_feature)
    current_feature_dim = dim_x if use_current_feature else 0
    shared_dim = config.model.get("shared_dim")
    if shared_dim is None:
        dim_model = int(config.model.dim_model)
    else:
        # OmegaConf resolves YAML aliases while loading, so tuning shared_dim
        # does not automatically update the three aliased fields. Resolve the
        # intended shared-width constraint explicitly for every trial.
        dim_model = int(shared_dim)

    if config.model.prediction_head == "cosine_embedding":
        hidden_value = (
            config.model.get("iqn_hidden_dim", dim_model)
            if shared_dim is None else dim_model
        )
        iqn_hidden_dim = None if hidden_value is None else int(hidden_value)
        embedding_value = (
            config.model.get("cos_emb_dim", iqn_hidden_dim)
            if shared_dim is None else dim_model
        )
        cos_emb_dim = 64 if embedding_value is None else int(embedding_value)
    elif config.model.prediction_head == "partially_monotonic":
        if config.model.get("monotonic_hidden_dims") is not None:
            iqn_hidden_dim = None
        else:
            iqn_hidden_dim = (
                config.model.get("iqn_hidden_dim", dim_model)
                if shared_dim is None else dim_model
            )
        cos_emb_dim = 64
    else:
        # Let the shared head factory report the unsupported selector.
        iqn_hidden_dim = None
        cos_emb_dim = 64

    if "rnn_type" in config.model:
        model = IQNRNN(
            rnn_type=config.model.rnn_type,
            dim_feature=dim_feature,
            dim_model=dim_model,
            num_layers=config.model.num_layers,
            current_feature_dim=current_feature_dim,
            iqn_hidden_dim=iqn_hidden_dim,
            n_cos_embedding=cos_emb_dim,
            dropout=config.model.dropout,
            **_prediction_head_kwargs(config.model),
        )
        train_loss_fn = compute_loss_iqn_rnn
        model_type = "rnn"
    else:
        model = IQNTransformer(
            dim_feature=dim_feature,
            dim_model=dim_model,
            num_head=config.model.num_heads,
            dim_ff=dim_model * 4,
            num_layers=config.model.num_layers,
            current_feature_dim=current_feature_dim,
            iqn_hidden_dim=iqn_hidden_dim,
            n_cos_embedding=cos_emb_dim,
            dropout=config.model.dropout,
            **_prediction_head_kwargs(config.model),
        )
        train_loss_fn = compute_loss_iqn_transformer
        model_type = "transformer"
    return model, train_loss_fn, model_type, use_current_feature


def _synchronize_shared_dimensions(config):
    shared_dim = config.model.get("shared_dim")
    if shared_dim is None:
        return

    shared_dim = int(shared_dim)
    config.model.dim_model = shared_dim
    config.model.iqn_hidden_dim = shared_dim
    config.model.cos_emb_dim = shared_dim


def _model_selection_valid_ratio(config) -> float:
    return float(config.tuning.get("model_selection_valid_ratio", 0.2))


def _normalization_params(config, sequence_data):
    if not bool(config.data.normalize):
        return None
    return (
        sequence_data["train_residuals_mu"],
        sequence_data["train_residuals_std"],
    )


def _run_single_trial(config, sequence_item, normalization_params):
    device = resolve_device(config.device)
    tau_mode, training_quantiles = resolve_iqn_training_quantiles(
        config.training, config.model.target_quantiles
    )
    config.training.tau_mode = tau_mode
    training_kwargs = (
        {"taus": torch.tensor(training_quantiles, dtype=torch.float32, device=device)}
        if training_quantiles is not None else {}
    )
    warn_iqn_training_interval_mismatch(tau_mode, config.model)
    validation_mode, validation_quantiles = resolve_iqn_validation_quantiles(
        config.training, config.model.target_quantiles
    )
    config.training.validation_loss = validation_mode
    validation_kwargs = (
        {"taus": torch.tensor(validation_quantiles, dtype=torch.float32, device=device)}
        if validation_quantiles is not None else {}
    )
    sorted_quantiles, pair_to_indices = get_interval_quantile_indices(config.model.target_quantiles)
    delta_threshold = float(config.tuning.get("delta_threshold", 0.0))

    train_dataset = sequence_item["train_dataset"]
    model_selection_valid_dataset = sequence_item["model_selection_valid_dataset"]
    tuning_evaluation_dataset = sequence_item["tuning_evaluation_dataset"]

    train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    model_selection_valid_dataloader = DataLoader(
        model_selection_valid_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
    )
    tuning_evaluation_dataloader = DataLoader(
        tuning_evaluation_dataset,
        batch_size=config.training.batch_size,
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
    optimizer = build_iqn_optimizer(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=float(config.training.get("weight_decay", 0.01)),
    )

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
                loss = loss_fn(
                    model, strided_feature, target_residual, config.model.num_taus,
                    target_x, **training_kwargs,
                )
            else:
                loss = loss_fn(
                    model, strided_feature, target_residual, config.model.num_taus,
                    **training_kwargs,
                )
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        epoch_train_loss = loss_sum / len(train_dataloader)
        train_loss.append(float(epoch_train_loss))

        model.eval()
        loss_sum = 0.0
        validation_weight = 0
        for batch_index, (strided_x, strided_residual, strided_y, target_x, target_residual, _, _) in enumerate(model_selection_valid_dataloader):
            strided_feature = generate_strided_feature(
                strided_x,
                strided_residual,
                strided_y,
                config.data.strided_features,
            ).to(device)
            target_residual = target_residual.to(device)
            target_x = target_x.to(device)
            with torch.no_grad():
                if validation_quantiles is not None:
                    loss = compute_iqn_interval_validation_loss(
                        model, strided_feature, target_residual,
                        validation_kwargs["taus"],
                        current_feature=target_x if use_current_feature else None,
                        sampling_seed=int(config.get("seed", 0)) + batch_index,
                    )
                elif use_current_feature:
                    loss = loss_fn(
                        model, strided_feature, target_residual, config.model.num_taus,
                        target_x, **validation_kwargs,
                    )
                else:
                    loss = loss_fn(
                        model, strided_feature, target_residual, config.model.num_taus,
                        **validation_kwargs,
                    )
            # Fixed-target loss weights each observation equally; sampled mode
            # retains the legacy mean over batches.
            weight = target_residual.shape[0] if validation_quantiles is not None else 1
            loss_sum += loss.item() * weight
            validation_weight += weight

        epoch_valid_loss = loss_sum / validation_weight
        model_selection_valid_loss.append(float(epoch_valid_loss))

        if epoch_valid_loss < best_loss:
            best_loss = epoch_valid_loss
            best_epoch = epoch_idx + 1
            best_model = copy.deepcopy(model.state_dict())

        if config.training.early_stop and (epoch_idx + 1 - best_epoch) >= config.training.early_stop:
            break

    model.load_state_dict(best_model)
    quantiles = torch.tensor(sorted_quantiles, dtype=torch.float32, device=device)

    evaluation_results = {
        tuple(confidence_pair): {
            "coverage": [],
            "interval_width": [],
            "winkler_score": [],
        }
        for confidence_pair in config.model.target_quantiles
    }

    if config.data.normalize:
        residual_mu, residual_std = normalization_params

    model.eval()
    for strided_x, strided_residual, strided_y, target_x, target_residual, target_y, target_predictions in tuning_evaluation_dataloader:
        with torch.no_grad():
            strided_feature = generate_strided_feature(
                strided_x,
                strided_residual,
                strided_y,
                config.data.strided_features,
            ).to(device)
            target_x = target_x.to(device)
            if use_current_feature:
                pred_quantile_values = model.get_predicted_quantile_values(
                    model,
                    strided_feature,
                    quantiles,
                    current_feature=target_x,
                )
            else:
                pred_quantile_values = model.get_predicted_quantile_values(model, strided_feature, quantiles)

        if device.type != "cpu":
            pred_quantile_values = pred_quantile_values.cpu().detach()

        for confidence_pair in config.model.target_quantiles:
            pair_key = tuple(confidence_pair)
            hi_idx, lo_idx = pair_to_indices[pair_key]
            hi = pred_quantile_values[:, hi_idx]
            lo = pred_quantile_values[:, lo_idx]
            evaluation_results[pair_key]["coverage"].extend(compute_coverage(hi, lo, target_residual))
            evaluation_results[pair_key]["interval_width"].extend(
                compute_interval_width(hi, lo, normalized_std=residual_std if config.data.normalize else None)
            )
            evaluation_results[pair_key]["winkler_score"].extend(
                compute_winkler_score(
                    hi,
                    lo,
                    target_y,
                    target_predictions,
                    pair_key,
                    normalized_params=(residual_mu, residual_std) if config.data.normalize else None,
                )
            )

    pair_metrics, selection_score, positive_delta_coverage = summarize_evaluation_results(
        evaluation_results,
        config.model.target_quantiles,
        delta_threshold=delta_threshold,
    )

    return {
        "model_type": model_type,
        "prediction_head": model.prediction_head,
        "iqn_num_layers": getattr(model.iqn, "iqn_num_layers", None),
        "interval_mode": model.iqn.interval_mode,
        "sampling_num": getattr(model.iqn, "sampling_num", None),
        "tau_mode": tau_mode,
        "training_quantiles": training_quantiles,
        "validation_loss": validation_mode,
        "validation_quantiles": validation_quantiles,
        "train_loss": train_loss,
        "model_fit_train_loss": train_loss,
        "model_selection_valid_loss": model_selection_valid_loss,
        # Backward-compatible aliases used by aggregate/reporting code.
        "valid_loss": model_selection_valid_loss,
        "best_model_selection_valid_loss": float(best_loss),
        "best_valid_loss": float(best_loss),
        "best_epoch": best_epoch,
        "evaluation_split": "nominal_validation",
        "final_test_evaluated": False,
        "num_train_samples": len(train_dataset),
        "num_model_selection_valid_samples": len(model_selection_valid_dataset),
        "num_tuning_evaluation_samples": len(tuning_evaluation_dataset),
        "sample_counts": {
            "train": len(train_dataset),
            "model_selection_valid": len(model_selection_valid_dataset),
            "tuning_evaluation": len(tuning_evaluation_dataset),
        },
        "pair_metrics": pair_metrics,
        "selection_score": selection_score,
        "positive_delta_coverage": positive_delta_coverage,
    }


def _prepared_data_cache_key(config):
    return (
        int(config.model.window_size),
        int(config.model.prediction_step),
        float(config.data.train_ratio),
        float(config.data.valid_ratio),
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
        model_selection_valid_ratio=_model_selection_valid_ratio(config),
    )
    return cpd


def _run_grid_trial(
    trial_index, trial_config, grid_values, selected_data, sequence_keys, seed, prepared_data_cache
):
    """Evaluate one configuration on every selected sequence, in their original order."""
    _synchronize_shared_dimensions(trial_config)
    tau_mode, training_quantiles = resolve_iqn_training_quantiles(
        trial_config.training, trial_config.model.target_quantiles
    )
    trial_config.training.tau_mode = tau_mode
    validation_mode, validation_quantiles = resolve_iqn_validation_quantiles(
        trial_config.training, trial_config.model.target_quantiles
    )
    trial_config.training.validation_loss = validation_mode
    print(
        f"[iqn_cp] starting trial {trial_index} on device={trial_config.get('device', 'default')} "
        f"with grid_values={grid_values}; tau_mode={tau_mode}, "
        f"training_quantiles={training_quantiles}; validation_loss={validation_mode}, "
        f"validation_quantiles={validation_quantiles}",
        flush=True,
    )
    set_global_seed(seed + trial_index)
    cache_key = _prepared_data_cache_key(trial_config)
    if cache_key not in prepared_data_cache:
        prepared_data_cache[cache_key] = _prepare_trial_data(selected_data, trial_config)
    cpd = prepared_data_cache[cache_key]
    sequence_results = {
        sequence_key: _run_single_trial(
            trial_config,
            cpd.dataset[sequence_key],
            _normalization_params(trial_config, cpd.data[sequence_key]),
        )
        for sequence_key in sequence_keys
    }
    result = aggregate_sequence_results(sequence_results, trial_config.model.target_quantiles)
    result["prediction_head"] = sequence_results[sequence_keys[0]]["prediction_head"]
    result["tau_mode"] = tau_mode
    result["training_quantiles"] = training_quantiles
    result["validation_loss"] = validation_mode
    result["validation_quantiles"] = validation_quantiles
    result["mean_best_model_selection_valid_loss"] = result["mean_best_valid_loss"]
    result["evaluation_split"] = "nominal_validation"
    result["final_test_evaluated"] = False
    return {
        "trial_index": trial_index,
        "sequence_keys": sequence_keys,
        "grid_values": grid_values,
        "result": result,
        "resolved_config": plain_config(trial_config),
        "evaluation_split": "nominal_validation",
        "final_test_evaluated": False,
    }


_worker_device = None
_worker_selected_data = None
_worker_sequence_keys = None
_worker_prepared_data_cache = None


def _initialize_gpu_worker(device, selected_data, sequence_keys):
    """Initialize one spawned worker, retaining CPU datasets between its trials."""
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
    args = parse_args("iqn_cp")
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

    print(f"[iqn_cp] base configuration: {base_config_path}", flush=True)
    print(f"[iqn_cp] prediction data: {base_config.data.data_path}", flush=True)
    print(f"[iqn_cp] saving results to: {save_dir}", flush=True)
    if worker_devices:
        print(f"[iqn_cp] parallel trial workers: {worker_devices}", flush=True)
    else:
        print("[iqn_cp] running trials sequentially", flush=True)
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
        "method": "iqn_cp",
        "base_config_path": str(base_config_path),
        "grid_config_path": str(args.grid_config.resolve()),
        "sequence_keys": sequence_keys,
        "num_sequences": len(sequence_keys),
        "delta_threshold": delta_threshold,
        "num_trials": len(trials),
        "num_positive_delta_coverage_trials": len(positive_trials),
        "top_k": args.top_k,
        "evaluation_split": "nominal_validation",
        "final_test_evaluated": False,
        "execution": {
            "mode": "parallel_trials" if worker_devices else "serial",
            "num_gpus": num_gpus,
            "worker_devices": worker_devices,
        },
        "tuning_protocol": {
            "model_fit_dataset": "train_dataset",
            "checkpoint_selection_dataset": "model_selection_valid_dataset",
            "hyperparameter_evaluation_dataset": "tuning_evaluation_dataset",
            "evaluation_split": "nominal_validation",
            "model_selection_valid_ratio": _model_selection_valid_ratio(base_config),
            "final_test_evaluated": False,
        },
        "top_trials": top_trials,
        "all_trials": trials,
    }
    finalize_and_save_results(save_dir, payload)


if __name__ == "__main__":
    main()
