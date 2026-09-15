from omegaconf import OmegaConf
import torch
import os
import copy
import numpy as np
from tqdm import tqdm
from dscp.models.iqn_transformer import IQNTransformer
from dscp.models.iqn_rnn import IQNRNN
from dscp.models.iqn import build_iqn_optimizer
from dscp.data import ConformalPredictionData
from dscp.loss import compute_loss_iqn_transformer, compute_loss_iqn_rnn
from utils.utils import load_data, save_data, read_setup, generate_strided_feature, get_interval_quantile_indices
from utils.reporting import compute_coverage, compute_interval_width, compute_winkler_score, construct_interval_endpoints, summarize_evaluation_results
from utils.plotting import plot_cp_prediction_intervals
from torch.utils.data import DataLoader


def _prediction_head_kwargs(model_config):
    """Resolve prediction-head options while preserving legacy configurations."""
    return {
        "prediction_head": model_config.get("prediction_head", "cosine_embedding"),
        "monotonic_num_layers": model_config.get("monotonic_num_layers", 1),
        "monotonic_hidden_dims": model_config.get("monotonic_hidden_dims"),
        "monotonic_activation": model_config.get(
            "monotonic_activation",
            "tanh",
        ),
    }


def _materialize_prediction_head_selector(model_config):
    """Make the effective legacy default explicit in saved configurations."""
    model_config.prediction_head = str(
        model_config.get("prediction_head", "cosine_embedding")
    ).strip().lower()


def _prediction_head_dimensions(model_config):
    """Resolve only dimensions used by the selected head."""
    prediction_head = model_config.get("prediction_head", "cosine_embedding")
    if prediction_head == "cosine_embedding":
        hidden_value = model_config.get("iqn_hidden_dim", model_config.dim_model)
        iqn_hidden_dim = None if hidden_value is None else int(hidden_value)
        embedding_value = model_config.get(
            "cos_emb_dim",
            iqn_hidden_dim if iqn_hidden_dim is not None else 64,
        )
        n_cos_embedding = (
            64 if embedding_value is None else int(embedding_value)
        )
        return iqn_hidden_dim, n_cos_embedding

    if prediction_head == "partially_monotonic":
        if model_config.get("monotonic_hidden_dims") is not None:
            return None, 64
        return model_config.get("iqn_hidden_dim", model_config.dim_model), 64

    # Let the head factory produce the canonical unsupported-selector error.
    return None, 64


def _save_resolved_config(config):
    """Persist the complete architecture next to state-dict checkpoints."""
    OmegaConf.save(
        config=config,
        f=os.path.join(config.saving_dir, "resolved_config.yaml"),
        resolve=True,
    )


def run_transformer_iqn_cp(config_path):

    config = OmegaConf.load(config_path)
    _materialize_prediction_head_selector(config.model)
    os.makedirs(config.saving_dir, exist_ok=True)
    _save_resolved_config(config)

    # load data
    data = load_data(config.data.data_path)  # load predictor results here
    base_predictor, data_type = read_setup(config.data.data_path)
    cpd = ConformalPredictionData(data)
    cpd.prepare_quantile_regression_datasets(config.model.window_size,
                                             config.model.prediction_step,
                                             config.data.train_ratio,
                                             config.data.valid_ratio,
                                             normalize=config.data.normalize)
    device = config.device
    sorted_quantiles, pair_to_indices = get_interval_quantile_indices(config.model.target_quantiles)
    log = dict()

    print("Experiment setup")
    print("Method: IQN - Transformer")
    print("Prediction head: {}".format(
        config.model.get("prediction_head", "cosine_embedding")
    ))
    print("Base predictor: {}".format(base_predictor))
    print("Data: {}".format(data_type))
    print("{} independent sequences".format(len(cpd.dataset)))

    for key, item in tqdm(cpd.dataset.items(), desc="repetition over independent sequences"):

        train_dataset = item["train_dataset"]
        valid_dataset = item["valid_dataset"]
        test_dataset = item["test_dataset"]

        train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.training.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

        dim_x = train_dataset.strided_x.shape[-1]
        if config.data.strided_features == "xry":
            dim_feature = dim_x + 2  # add 1 for residual and y as input
        elif config.data.strided_features == "xr":
            dim_feature = dim_x + 1  # add 1 for residual as input
        elif config.data.strided_features == "r":
            dim_feature = 1  # only residual is used for feature
        else:
            raise ValueError("wrong strided features specified")

        iqn_hidden_dim, n_cos_embedding = _prediction_head_dimensions(
            config.model
        )
        iqn_transformer = IQNTransformer(
            dim_feature=dim_feature,
            dim_model=config.model.dim_model,
            num_head=config.model.num_heads,
            dim_ff=config.model.dim_model * 4,
            num_layers=config.model.num_layers,
            current_feature_dim=dim_x if config.model.use_current_feature else 0,
            iqn_hidden_dim=iqn_hidden_dim,
            n_cos_embedding=n_cos_embedding,
            dropout=config.model.dropout,
            **_prediction_head_kwargs(config.model),
        )
        iqn_transformer.to(device)
        optimizer = build_iqn_optimizer(
            iqn_transformer,
            learning_rate=config.training.learning_rate,
            weight_decay=float(config.training.get("weight_decay", 0.01)),
        )

        train_loss = []
        valid_loss = []
        best_loss = np.inf

        for i in tqdm(range(config.training.epochs), desc="training epochs"):

            loss_sum = 0.0
            iqn_transformer.train()
            for strided_x, strided_residual, strided_y, target_x, target_residual, _, _ in tqdm(train_dataloader):

                optimizer.zero_grad()
                strided_feature = generate_strided_feature(strided_x,
                                                           strided_residual,
                                                           strided_y,
                                                           config.data.strided_features)
                strided_feature = strided_feature.to(device)
                target_residual = target_residual.to(device)
                target_x = target_x.to(device)
                if config.model.use_current_feature:
                    loss = compute_loss_iqn_transformer(
                        iqn_transformer,
                        strided_feature,
                        target_residual,
                        config.model.num_taus,
                        target_x,
                    )
                else:
                    loss = compute_loss_iqn_transformer(
                        iqn_transformer,
                        strided_feature,
                        target_residual,
                        config.model.num_taus,
                    )
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            epoch_train_loss = loss_sum / len(train_dataloader)
            train_loss.append(epoch_train_loss)
            print("training loss at epoch {}: {}".format(i + 1, epoch_train_loss))

            loss_sum = 0.0
            iqn_transformer.eval()
            for strided_x, strided_residual, strided_y, target_x, target_residual, _, _ in tqdm(valid_dataloader):

                strided_feature = generate_strided_feature(strided_x,
                                                           strided_residual,
                                                           strided_y,
                                                           config.data.strided_features)
                strided_feature = strided_feature.to(device)
                target_residual = target_residual.to(device)
                target_x = target_x.to(device)
                with torch.no_grad():
                    if config.model.use_current_feature:
                        loss = compute_loss_iqn_transformer(
                            iqn_transformer,
                            strided_feature,
                            target_residual,
                            config.model.num_taus,
                            target_x,
                        )
                    else:
                        loss = compute_loss_iqn_transformer(
                            iqn_transformer,
                            strided_feature,
                            target_residual,
                            config.model.num_taus,
                        )
                loss_sum += loss.item()

            epoch_valid_loss = loss_sum / len(valid_dataloader)
            valid_loss.append(epoch_valid_loss)
            print("validation loss at epoch {}: {}".format(i + 1, epoch_valid_loss))

            if epoch_valid_loss < best_loss:
                best_loss = epoch_valid_loss
                best_epoch = i + 1
                best_model = copy.deepcopy(iqn_transformer.state_dict())

            if config.training.early_stop:
                if (i + 1 - best_epoch) >= config.training.early_stop:
                    # if the loss did not decrease for (early_stop) epoch in a row, stop training
                    break

        iqn_transformer.load_state_dict(best_model)
        evaluation_results = {
            tuple(confidence_pair): {"coverage": [],
                                     "interval_width": [],
                                     "winkler_score": [],
                                     "upper_interval": [],
                                     "lower_interval": [],
                                     "upper_residual_quantile": [],
                                     "lower_residual_quantile": [],
                                     "target_y": [],
                                     "target_predictions": []}
            for confidence_pair in config.model.target_quantiles
        }

        if config.data.normalize:
            residuals_noramlized_mu = cpd.data[key]["train_residuals_mu"]
            residuals_noramlized_std = cpd.data[key]["train_residuals_std"]

        iqn_transformer.eval()
        quantiles = torch.tensor(sorted_quantiles, dtype=torch.float32, device=device)
        for strided_x, strided_residual, strided_y, target_x, target_residual, target_y, target_predictions in tqdm(test_dataloader):

            with torch.no_grad():
                strided_feature = generate_strided_feature(strided_x,
                                                           strided_residual,
                                                           strided_y,
                                                           config.data.strided_features)
                strided_feature = strided_feature.to(device)
                target_x = target_x.to(device)
                if config.model.use_current_feature:
                    pred_quantile_values = iqn_transformer.get_predicted_quantile_values(
                        iqn_transformer,
                        strided_feature,
                        quantiles,
                        current_feature=target_x,
                    )
                else:
                    pred_quantile_values = iqn_transformer.get_predicted_quantile_values(
                        iqn_transformer,
                        strided_feature,
                        quantiles,
                    )

            if config.device != "cpu":
                pred_quantile_values = pred_quantile_values.cpu().detach()

            for confidence_pair in config.model.target_quantiles:
                tuple_confidence_pair = tuple(confidence_pair)
                hi_idx, lo_idx = pair_to_indices[tuple_confidence_pair]
                hi = pred_quantile_values[:, hi_idx]
                lo = pred_quantile_values[:, lo_idx]
                this_coverage = compute_coverage(hi, lo, target_residual)

                if config.data.normalize:
                    this_interval_width = compute_interval_width(hi,
                                                                 lo,
                                                                 normalized_std=residuals_noramlized_std)
                    this_winkler_score = compute_winkler_score(hi,
                                                               lo,
                                                               target_y,
                                                               target_predictions,
                                                               tuple_confidence_pair,
                                                               normalized_params=(residuals_noramlized_mu,
                                                                                  residuals_noramlized_std))
                else:
                    this_interval_width = compute_interval_width(hi,
                                                                 lo,
                                                                 normalized_std=None)
                    this_winkler_score = compute_winkler_score(hi,
                                                               lo,
                                                               target_y,
                                                               target_predictions,
                                                               tuple_confidence_pair,
                                                               normalized_params=None)

                upper_interval, lower_interval = construct_interval_endpoints(
                    hi,
                    lo,
                    target_predictions,
                    normalized_params=(residuals_noramlized_mu,
                                       residuals_noramlized_std)
                    if config.data.normalize else None,
                )

                evaluation_results[tuple_confidence_pair]["upper_interval"].extend(upper_interval.tolist())
                evaluation_results[tuple_confidence_pair]["lower_interval"].extend(lower_interval.tolist())
                evaluation_results[tuple_confidence_pair]["upper_residual_quantile"].extend(hi.tolist())
                evaluation_results[tuple_confidence_pair]["lower_residual_quantile"].extend(lo.tolist())
                evaluation_results[tuple_confidence_pair]["coverage"].extend(this_coverage)
                evaluation_results[tuple_confidence_pair]["interval_width"].extend(this_interval_width)
                evaluation_results[tuple_confidence_pair]["winkler_score"].extend(this_winkler_score)
                evaluation_results[tuple_confidence_pair]["target_y"].extend(target_y.flatten().tolist())
                evaluation_results[tuple_confidence_pair]["target_predictions"].extend(target_predictions.flatten().tolist())

                if config.data.normalize:
                    evaluation_results[tuple_confidence_pair]["train_residuals_mu"] = residuals_noramlized_mu
                    evaluation_results[tuple_confidence_pair]["train_residuals_std"] = residuals_noramlized_std

        for confidence_pair in config.model.target_quantiles:
            tuple_confidence_pair = tuple(confidence_pair)
            target_alpha = max(tuple_confidence_pair) - min(tuple_confidence_pair)
            avg_coverage = np.mean(evaluation_results[tuple_confidence_pair]["coverage"])
            avg_delta_coverage = avg_coverage - target_alpha
            avg_interval_width = np.mean(evaluation_results[tuple_confidence_pair]["interval_width"])
            avg_winkler_score = np.mean(evaluation_results[tuple_confidence_pair]["winkler_score"])
            print("avg coverage: {}".format(avg_coverage))
            print("avg delta coverage: {}".format(avg_delta_coverage))
            print("avg interval width: {}".format(avg_interval_width))
            print("avg winkler score: {}".format(avg_winkler_score))
            evaluation_results[tuple_confidence_pair]["avg_coverage"] = avg_coverage
            evaluation_results[tuple_confidence_pair]["avg_delta_coverage"] = avg_delta_coverage
            evaluation_results[tuple_confidence_pair]["avg_interval_width"] = avg_interval_width
            evaluation_results[tuple_confidence_pair]["avg_winkler_score"] = avg_winkler_score

        log[key] = {"prediction_head": iqn_transformer.prediction_head,
                    "model_config": OmegaConf.to_container(config.model, resolve=True),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "evaluation_results": evaluation_results}

        torch.save(best_model, os.path.join(config.saving_dir, key + '_model.pt'))
        save_data(os.path.join(config.saving_dir, "log.pkl"), log)

    summary_results = summarize_evaluation_results(log, config.model.target_quantiles)

    for tuple_confidence_pair, summary in summary_results.items():
        print("Summary for confidence pair {}".format(tuple_confidence_pair))
        print("avg_coverage mean: {}, std: {}".format(
            summary["avg_coverage_mean"],
            summary["avg_coverage_std"])
        )
        print("avg_delta_coverage mean: {}, std: {}".format(
            summary["avg_delta_coverage_mean"],
            summary["avg_delta_coverage_std"])
        )
        print("avg_interval_width mean: {}, std: {}".format(
            summary["avg_interval_width_mean"],
            summary["avg_interval_width_std"])
        )
        print("avg_winkler_score mean: {}, std: {}".format(
            summary["avg_winkler_score_mean"],
            summary["avg_winkler_score_std"])
        )

    save_data(os.path.join(config.saving_dir, "summary_results.pkl"), summary_results)

    if config.plotting.plotting:
        plot_cp_prediction_intervals(log,
                                     config.model.target_quantiles,
                                     config.plotting.plotting_seq_len,
                                     os.path.join(config.saving_dir, "plots"))


def run_rnn_iqn_cp(config_path):

    config = OmegaConf.load(config_path)
    _materialize_prediction_head_selector(config.model)
    os.makedirs(config.saving_dir, exist_ok=True)
    _save_resolved_config(config)

    # load data
    data = load_data(config.data.data_path)  # load predictor results here
    base_predictor, data_type = read_setup(config.data.data_path)
    cpd = ConformalPredictionData(data)
    cpd.prepare_quantile_regression_datasets(config.model.window_size,
                                             config.model.prediction_step,
                                             config.data.train_ratio,
                                             config.data.valid_ratio,
                                             normalize=config.data.normalize)
    device = config.device
    sorted_quantiles, pair_to_indices = get_interval_quantile_indices(config.model.target_quantiles)
    log = dict()

    print("Experiment setup")
    print("Method: IQN - RNN")
    print("Prediction head: {}".format(
        config.model.get("prediction_head", "cosine_embedding")
    ))
    print("Base predictor: {}".format(base_predictor))
    print("Data: {}".format(data_type))
    print("{} independent sequences".format(len(cpd.dataset)))

    for key, item in tqdm(cpd.dataset.items(), desc="repetition over independent sequences"):

        train_dataset = item["train_dataset"]
        valid_dataset = item["valid_dataset"]
        test_dataset = item["test_dataset"]

        train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.training.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

        dim_x = train_dataset.strided_x.shape[-1]
        if config.data.strided_features == "xry":
            dim_feature = dim_x + 2  # add 1 for residual and y as input
        elif config.data.strided_features == "xr":
            dim_feature = dim_x + 1  # add 1 for residual as input
        elif config.data.strided_features == "r":
            dim_feature = 1  # only residual is used for feature
        else:
            raise ValueError("wrong strided features specified")

        iqn_hidden_dim, n_cos_embedding = _prediction_head_dimensions(
            config.model
        )
        iqn_rnn = IQNRNN(
            rnn_type=config.model.rnn_type,
            dim_feature=dim_feature,
            dim_model=config.model.dim_model,
            num_layers=config.model.num_layers,
            current_feature_dim=dim_x if config.model.use_current_feature else 0,
            iqn_hidden_dim=iqn_hidden_dim,
            n_cos_embedding=n_cos_embedding,
            dropout=config.model.dropout,
            **_prediction_head_kwargs(config.model),
        )
        iqn_rnn.to(device)
        optimizer = build_iqn_optimizer(
            iqn_rnn,
            learning_rate=config.training.learning_rate,
            weight_decay=float(config.training.get("weight_decay", 0.01)),
        )

        train_loss = []
        valid_loss = []
        best_loss = np.inf

        for i in tqdm(range(config.training.epochs), desc="training epochs"):

            loss_sum = 0.0
            iqn_rnn.train()
            for strided_x, strided_residual, strided_y, target_x, target_residual, _, _ in tqdm(train_dataloader):

                optimizer.zero_grad()
                strided_feature = generate_strided_feature(strided_x,
                                                           strided_residual,
                                                           strided_y,
                                                           config.data.strided_features)
                strided_feature = strided_feature.to(device)
                target_residual = target_residual.to(device)
                target_x = target_x.to(device)
                if config.model.use_current_feature:
                    loss = compute_loss_iqn_rnn(
                        iqn_rnn,
                        strided_feature,
                        target_residual,
                        config.model.num_taus,
                        target_x,
                    )
                else:
                    loss = compute_loss_iqn_rnn(
                        iqn_rnn,
                        strided_feature,
                        target_residual,
                        config.model.num_taus,
                    )
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            epoch_train_loss = loss_sum / len(train_dataloader)
            train_loss.append(epoch_train_loss)
            print("training loss at epoch {}: {}".format(i + 1, epoch_train_loss))

            loss_sum = 0.0
            iqn_rnn.eval()
            for strided_x, strided_residual, strided_y, target_x, target_residual, _, _ in tqdm(valid_dataloader):

                strided_feature = generate_strided_feature(strided_x,
                                                           strided_residual,
                                                           strided_y,
                                                           config.data.strided_features)
                strided_feature = strided_feature.to(device)
                target_residual = target_residual.to(device)
                target_x = target_x.to(device)
                with torch.no_grad():
                    if config.model.use_current_feature:
                        loss = compute_loss_iqn_rnn(
                            iqn_rnn,
                            strided_feature,
                            target_residual,
                            config.model.num_taus,
                            target_x,
                        )
                    else:
                        loss = compute_loss_iqn_rnn(
                            iqn_rnn,
                            strided_feature,
                            target_residual,
                            config.model.num_taus,
                        )
                loss_sum += loss.item()

            epoch_valid_loss = loss_sum / len(valid_dataloader)
            valid_loss.append(epoch_valid_loss)
            print("validation loss at epoch {}: {}".format(i + 1, epoch_valid_loss))

            if epoch_valid_loss < best_loss:
                best_loss = epoch_valid_loss
                best_epoch = i + 1
                best_model = copy.deepcopy(iqn_rnn.state_dict())

            if config.training.early_stop:
                if (i + 1 - best_epoch) >= config.training.early_stop:
                    # if the loss did not decrease for (early_stop) epoch in a row, stop training
                    break

        iqn_rnn.load_state_dict(best_model)
        evaluation_results = {
            tuple(confidence_pair): {"coverage": [],
                                     "interval_width": [],
                                     "winkler_score": [],
                                     "upper_interval": [],
                                     "lower_interval": [],
                                     "upper_residual_quantile": [],
                                     "lower_residual_quantile": [],
                                     "target_y": [],
                                     "target_predictions": []}
            for confidence_pair in config.model.target_quantiles
        }

        if config.data.normalize:
            residuals_noramlized_mu = cpd.data[key]["train_residuals_mu"]
            residuals_noramlized_std = cpd.data[key]["train_residuals_std"]

        iqn_rnn.eval()
        quantiles = torch.tensor(sorted_quantiles, dtype=torch.float32, device=device)
        for strided_x, strided_residual, strided_y, target_x, target_residual, target_y, target_predictions in tqdm(test_dataloader):

            with torch.no_grad():
                strided_feature = generate_strided_feature(strided_x,
                                                           strided_residual,
                                                           strided_y,
                                                           config.data.strided_features)
                strided_feature = strided_feature.to(device)
                target_x = target_x.to(device)
                if config.model.use_current_feature:
                    pred_quantile_values = iqn_rnn.get_predicted_quantile_values(
                        iqn_rnn,
                        strided_feature,
                        quantiles,
                        current_feature=target_x,
                    )
                else:
                    pred_quantile_values = iqn_rnn.get_predicted_quantile_values(
                        iqn_rnn,
                        strided_feature,
                        quantiles,
                    )

            if config.device != "cpu":
                pred_quantile_values = pred_quantile_values.cpu().detach()

            for confidence_pair in config.model.target_quantiles:
                tuple_confidence_pair = tuple(confidence_pair)
                hi_idx, lo_idx = pair_to_indices[tuple_confidence_pair]
                hi = pred_quantile_values[:, hi_idx]
                lo = pred_quantile_values[:, lo_idx]
                this_coverage = compute_coverage(hi, lo, target_residual)

                if config.data.normalize:
                    this_interval_width = compute_interval_width(hi,
                                                                 lo,
                                                                 normalized_std=residuals_noramlized_std)
                    this_winkler_score = compute_winkler_score(hi,
                                                               lo,
                                                               target_y,
                                                               target_predictions,
                                                               tuple_confidence_pair,
                                                               normalized_params=(residuals_noramlized_mu,
                                                                                  residuals_noramlized_std))
                else:
                    this_interval_width = compute_interval_width(hi,
                                                                 lo,
                                                                 normalized_std=None)
                    this_winkler_score = compute_winkler_score(hi,
                                                               lo,
                                                               target_y,
                                                               target_predictions,
                                                               tuple_confidence_pair,
                                                               normalized_params=None)

                upper_interval, lower_interval = construct_interval_endpoints(
                    hi,
                    lo,
                    target_predictions,
                    normalized_params=(residuals_noramlized_mu,
                                       residuals_noramlized_std)
                    if config.data.normalize else None,
                )

                evaluation_results[tuple_confidence_pair]["upper_interval"].extend(upper_interval.tolist())
                evaluation_results[tuple_confidence_pair]["lower_interval"].extend(lower_interval.tolist())
                evaluation_results[tuple_confidence_pair]["upper_residual_quantile"].extend(hi.tolist())
                evaluation_results[tuple_confidence_pair]["lower_residual_quantile"].extend(lo.tolist())
                evaluation_results[tuple_confidence_pair]["coverage"].extend(this_coverage)
                evaluation_results[tuple_confidence_pair]["interval_width"].extend(this_interval_width)
                evaluation_results[tuple_confidence_pair]["winkler_score"].extend(this_winkler_score)
                evaluation_results[tuple_confidence_pair]["target_y"].extend(target_y.flatten().tolist())
                evaluation_results[tuple_confidence_pair]["target_predictions"].extend(target_predictions.flatten().tolist())

                if config.data.normalize:
                    evaluation_results[tuple_confidence_pair]["train_residuals_mu"] = residuals_noramlized_mu
                    evaluation_results[tuple_confidence_pair]["train_residuals_std"] = residuals_noramlized_std

        for confidence_pair in config.model.target_quantiles:
            tuple_confidence_pair = tuple(confidence_pair)
            target_alpha = max(tuple_confidence_pair) - min(tuple_confidence_pair)
            avg_coverage = np.mean(evaluation_results[tuple_confidence_pair]["coverage"])
            avg_delta_coverage = avg_coverage - target_alpha
            avg_interval_width = np.mean(evaluation_results[tuple_confidence_pair]["interval_width"])
            avg_winkler_score = np.mean(evaluation_results[tuple_confidence_pair]["winkler_score"])
            print("avg coverage: {}".format(avg_coverage))
            print("avg delta coverage: {}".format(avg_delta_coverage))
            print("avg interval width: {}".format(avg_interval_width))
            print("avg winkler score: {}".format(avg_winkler_score))
            evaluation_results[tuple_confidence_pair]["avg_coverage"] = avg_coverage
            evaluation_results[tuple_confidence_pair]["avg_delta_coverage"] = avg_delta_coverage
            evaluation_results[tuple_confidence_pair]["avg_interval_width"] = avg_interval_width
            evaluation_results[tuple_confidence_pair]["avg_winkler_score"] = avg_winkler_score

        log[key] = {"prediction_head": iqn_rnn.prediction_head,
                    "model_config": OmegaConf.to_container(config.model, resolve=True),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "evaluation_results": evaluation_results}

        torch.save(best_model, os.path.join(config.saving_dir, key + '_model.pt'))
        save_data(os.path.join(config.saving_dir, "log.pkl"), log)

    summary_results = summarize_evaluation_results(log, config.model.target_quantiles)

    for tuple_confidence_pair, summary in summary_results.items():
        print("Summary for confidence pair {}".format(tuple_confidence_pair))
        print("avg_coverage mean: {}, std: {}".format(
            summary["avg_coverage_mean"],
            summary["avg_coverage_std"])
        )
        print("avg_delta_coverage mean: {}, std: {}".format(
            summary["avg_delta_coverage_mean"],
            summary["avg_delta_coverage_std"])
        )
        print("avg_interval_width mean: {}, std: {}".format(
            summary["avg_interval_width_mean"],
            summary["avg_interval_width_std"])
        )
        print("avg_winkler_score mean: {}, std: {}".format(
            summary["avg_winkler_score_mean"],
            summary["avg_winkler_score_std"])
        )

    save_data(os.path.join(config.saving_dir, "summary_results.pkl"), summary_results)

    if config.plotting.plotting:
        plot_cp_prediction_intervals(log,
                                     config.model.target_quantiles,
                                     config.plotting.plotting_seq_len,
                                     os.path.join(config.saving_dir, "plots"))
