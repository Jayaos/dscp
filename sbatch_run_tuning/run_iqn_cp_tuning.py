import copy

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dscp.data import ConformalPredictionData
from dscp.loss import compute_loss_iqn_rnn, compute_loss_iqn_transformer
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
    resolve_device,
    resolve_num_sequences,
    set_global_seed,
    summarize_evaluation_results,
    write_trial_artifacts,
)
from utils.reporting import compute_coverage, compute_interval_width, compute_winkler_score
from utils.utils import generate_strided_feature, get_interval_quantile_indices, load_data


def _build_model(config, dim_feature: int, dim_x: int):
    use_current_feature = bool(config.model.use_current_feature)
    current_feature_dim = dim_x if use_current_feature else 0

    if "rnn_type" in config.model:
        model = IQNRNN(
            rnn_type=config.model.rnn_type,
            dim_feature=dim_feature,
            dim_model=config.model.dim_model,
            num_layers=config.model.num_layers,
            current_feature_dim=current_feature_dim,
            iqn_hidden_dim=config.model.iqn_hidden_dim,
            output_dim=1,
            n_cos_embedding=config.model.cos_emb_dim,
            dropout=config.model.dropout,
        )
        train_loss_fn = compute_loss_iqn_rnn
        model_type = "rnn"
    else:
        model = IQNTransformer(
            dim_feature=dim_feature,
            dim_model=config.model.dim_model,
            num_head=config.model.num_heads,
            dim_ff=config.model.dim_model * 4,
            num_layers=config.model.num_layers,
            current_feature_dim=current_feature_dim,
            iqn_hidden_dim=config.model.iqn_hidden_dim,
            output_dim=1,
            n_cos_embedding=config.model.cos_emb_dim,
            dropout=config.model.dropout,
        )
        train_loss_fn = compute_loss_iqn_transformer
        model_type = "transformer"
    return model, train_loss_fn, model_type, use_current_feature


def _run_single_trial(config, sequence_item, sequence_data):
    device = resolve_device(config.device)
    sorted_quantiles, pair_to_indices = get_interval_quantile_indices(config.model.target_quantiles)

    train_dataset = sequence_item["train_dataset"]
    valid_dataset = sequence_item["valid_dataset"]
    test_dataset = sequence_item["test_dataset"]

    train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.training.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

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
    valid_loss = []
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
                loss = loss_fn(model, strided_feature, target_residual, config.model.num_taus, target_x)
            else:
                loss = loss_fn(model, strided_feature, target_residual, config.model.num_taus)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        epoch_train_loss = loss_sum / len(train_dataloader)
        train_loss.append(float(epoch_train_loss))

        model.eval()
        loss_sum = 0.0
        for strided_x, strided_residual, strided_y, target_x, target_residual, _, _ in valid_dataloader:
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
                    loss = loss_fn(model, strided_feature, target_residual, config.model.num_taus, target_x)
                else:
                    loss = loss_fn(model, strided_feature, target_residual, config.model.num_taus)
            loss_sum += loss.item()

        epoch_valid_loss = loss_sum / len(valid_dataloader)
        valid_loss.append(float(epoch_valid_loss))

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
        residual_mu = sequence_data["train_residuals_mu"]
        residual_std = sequence_data["train_residuals_std"]

    model.eval()
    for strided_x, strided_residual, strided_y, target_x, target_residual, target_y, target_predictions in test_dataloader:
        with torch.no_grad():
            strided_feature = generate_strided_feature(
                strided_x,
                strided_residual,
                strided_y,
                config.data.strided_features,
            ).to(device)
            target_x = target_x.to(device)
            if use_current_feature:
                pred_quantile_values = model.get_predicted_quantile_values(model, strided_feature, quantiles, target_x)
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
    )

    return {
        "model_type": model_type,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "best_valid_loss": float(best_loss),
        "best_epoch": best_epoch,
        "pair_metrics": pair_metrics,
        "selection_score": selection_score,
        "positive_delta_coverage": positive_delta_coverage,
    }


def main():
    args = parse_args("iqn_cp")
    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    base_config = OmegaConf.load(args.base_config)
    grid = load_grid(args.grid_config)

    data = load_data(base_config.data.data_path)
    cpd = ConformalPredictionData(data)
    cpd.prepare_quantile_regression_datasets(
        base_config.model.window_size,
        base_config.model.prediction_step,
        base_config.data.train_ratio,
        base_config.data.valid_ratio,
        normalize=base_config.data.normalize,
    )
    num_sequences = resolve_num_sequences(base_config)
    sequence_keys = choose_sequence_keys(cpd.dataset, args.sequence_key, args.sequence_index, num_sequences)

    trials = []
    for trial_index, (trial_config, grid_values) in enumerate(iter_grid_configs(base_config, grid), start=1):
        print(f"[iqn_cp] starting trial {trial_index} with grid_values={grid_values}", flush=True)
        set_global_seed(args.seed + trial_index)
        sequence_results = {
            sequence_key: _run_single_trial(trial_config, cpd.dataset[sequence_key], cpd.data[sequence_key])
            for sequence_key in sequence_keys
        }
        result = aggregate_sequence_results(sequence_results, trial_config.model.target_quantiles)
        record = {
            "trial_index": trial_index,
            "sequence_keys": sequence_keys,
            "grid_values": grid_values,
            "result": result,
            "resolved_config": plain_config(trial_config),
        }
        trials.append(record)
        write_trial_artifacts(save_dir, trial_index, trial_config, record)

    positive_trials = [trial for trial in trials if trial["result"]["positive_delta_coverage"]]
    ranked_trials = sorted(positive_trials, key=lambda item: item["result"]["selection_score"])
    top_trials = ranked_trials[: args.top_k]

    payload = {
        "method": "iqn_cp",
        "base_config_path": str(args.base_config.resolve()),
        "grid_config_path": str(args.grid_config.resolve()),
        "sequence_keys": sequence_keys,
        "num_sequences": len(sequence_keys),
        "num_trials": len(trials),
        "num_positive_delta_coverage_trials": len(positive_trials),
        "top_k": args.top_k,
        "top_trials": top_trials,
        "all_trials": trials,
    }
    finalize_and_save_results(save_dir, payload)


if __name__ == "__main__":
    main()
