from omegaconf import OmegaConf
import torch
import os
import copy
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from baselines.hopcpt.model import HopfieldNet
from baselines.hopcpt.loss import compute_hopfield_net_loss
from dscp.data import ConformalPredictionData, initialize_valid_dataloader, initialize_test_dataloader
from utils.utils import load_data, save_data, read_setup
from utils.utils import generate_feature_hopcpt_training, generate_feature_hopcpt_test, estimate_hopcpt_residual_interval
from utils.reporting import compute_coverage, compute_interval_width, compute_winkler_score, summarize_evaluation_results
from utils.plotting import plot_cp_prediction_intervals


def _device_from_config(device):
    if device == "cpu":
        return "cpu"
    if isinstance(device, str):
        if device.startswith("cuda") or device == "cpu":
            return device
        if device.isdigit():
            return "cuda:{}".format(device)
    return "cuda:{}".format(int(device))


def _parallel_devices(config):
    enabled = OmegaConf.select(config, "parallel.enabled", default=False)
    if not enabled:
        return None

    devices = OmegaConf.select(config, "parallel.devices", default=None)
    if devices is not None:
        return [_device_from_config(device) for device in devices]

    num_gpus = OmegaConf.select(config, "parallel.num_gpus", default=None)
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    num_gpus = int(num_gpus)
    if num_gpus <= 0:
        raise ValueError("parallel.enabled=True requires at least one CUDA GPU.")
    return ["cuda:{}".format(i) for i in range(num_gpus)]


def _split_keys_by_device(keys, devices):
    chunks = {device: [] for device in devices}
    for i, key in enumerate(keys):
        chunks[devices[i % len(devices)]].append(key)
    return {device: chunk for device, chunk in chunks.items() if chunk}


def _hopcpt_residual_normalization_params(data, normalize):
    if not normalize:
        return None

    std = np.asarray(data["heldout_train_y_std"], dtype=float) + 1e-8
    if std.size == 1:
        return 0.0, float(std.reshape(-1)[0])

    return np.zeros_like(std), std


def _run_hopcpt_sequence(key, data, config, device):
    target_quantiles = config.model.target_quantiles
    selection_confidence_pair = tuple(target_quantiles[0]) # this is used for validation
    selection_target_coverage = max(selection_confidence_pair) - min(selection_confidence_pair)
    predict_absolute_residual = OmegaConf.select(
        config, "model.predict_absolute_residual",
        default=OmegaConf.select(config, "model.use_absolute_residual", default=True))
    conformal_absolute_residual = OmegaConf.select(
        config, "model.conformal_absolute_residual",
        default=OmegaConf.select(config, "model.use_absolute_residual", default=False))

    train_size = data["train_size"]
    valid_size = data["valid_size"]
    test_size = data["test_size"]
    valid_dataloader_size = data["valid_dataloader_size"]
    test_dataloader_size = data["test_dataloader_size"]
    residual_normalization_params = _hopcpt_residual_normalization_params(data,
                                                                          config.data.normalize)
    residual_normalized_std = (residual_normalization_params[1]
                               if residual_normalization_params is not None
                               else None)

    dim_feature = data["heldout_context"].shape[-1]
    dim_context_encoding = config.model.dim_context_encoding
    if dim_context_encoding is None or dim_context_encoding == "auto":
        dim_context_encoding = dim_feature
    dim_hopfield_hidden = config.model.dim_hopfield_hidden
    if dim_hopfield_hidden is None or dim_hopfield_hidden == "auto":
        dim_hopfield_hidden = dim_context_encoding

    # model initialization
    hopfield_net = HopfieldNet(dim_feature,
                               dim_context_encoding,
                               dim_hopfield_hidden,
                               config.model.beta,
                               config.model.use_temporal_encoding)
    hopfield_net.to(device)
    optimizer = torch.optim.AdamW(hopfield_net.parameters(),
                                  lr=config.training.learning_rate) # TODO: params for adamW?

    train_loss = []
    valid_coverages = []
    valid_delta_coverages = []
    valid_interval_widths = []
    best_interval_width = np.inf
    best_delta_coverage = -np.inf
    best_epoch = 0
    best_model = copy.deepcopy(hopfield_net.state_dict())

    epoch_iter = tqdm(range(config.training.epochs),
                      desc="{} training epochs on {}".format(key, device),
                      leave=False)
    for i in epoch_iter:

        hopfield_net.train()
        optimizer.zero_grad()
        memory_feature = generate_feature_hopcpt_training(data["heldout_train_context"])
        memory_feature = memory_feature.to(device) # (1, memory_length, feature_dim)
        memory_residual = torch.from_numpy(data["heldout_train_residual"]).to(torch.float32).to(device)
        loss = compute_hopfield_net_loss(hopfield_net,
                                         memory_feature,
                                         memory_residual,
                                         predict_absolute_residual)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        epoch_iter.set_postfix(loss=loss.item())

        if (i+1) % config.training.validation_epochs == 0:

            # evaluate on the validation set
            hopfield_net.eval()
            this_coverages = []
            this_interval_widths = []

            valid_context_dataloader, valid_target_y_dataloader, \
                valid_residual_dataloader, valid_prediction_dataloader = initialize_valid_dataloader(data,
                                                                                                    train_size,
                                                                                                    valid_size,
                                                                                                    config.model.prediction_step,
                                                                                                    config.model.memory_size,
                                                                                                    config.data.normalize)

            for j in range(valid_dataloader_size):

                strided_context, target_context = next(valid_context_dataloader)
                _, target_y = next(valid_target_y_dataloader)
                strided_residual, target_residual = next(valid_residual_dataloader)
                _, target_predictions = next(valid_prediction_dataloader)

                memory_feature, query_feature = generate_feature_hopcpt_test(strided_context,
                                                                             target_context)
                memory_feature = memory_feature.to(device) # (batch_size, memeory_length, feature_dim)
                query_feature = query_feature.to(device) # (batch_size, 1, feature_dim)
                with torch.no_grad():
                    # (batch_size, 1, 1, memory_length)
                    association_matrix = hopfield_net.obtain_association_matrix(memory_feature, query_feature)
                lo, hi = estimate_hopcpt_residual_interval(association_matrix,
                                                     strided_residual,
                                                     selection_confidence_pair,
                                                     config.model.sampling_num,
                                                     conformal_absolute_residual)
                this_coverage = compute_coverage(hi, lo, target_residual)
                this_interval_width = compute_interval_width(hi,
                                                             lo,
                                                             normalized_std=residual_normalized_std)
                this_coverages.extend(this_coverage)
                this_interval_widths.extend(this_interval_width)

            this_delta_coverage = np.mean(this_coverages) - selection_target_coverage
            valid_coverages.append(np.mean(this_coverages))
            valid_delta_coverages.append(this_delta_coverage)
            this_avg_interval_width = np.mean(this_interval_widths)
            valid_interval_widths.append(this_avg_interval_width)
            epoch_iter.set_postfix(loss=loss.item(),
                                   valid_coverage=valid_coverages[-1],
                                   valid_delta=this_delta_coverage,
                                   valid_width=this_avg_interval_width)

            valid_coverage = this_delta_coverage >= 0
            best_has_valid_coverage = best_delta_coverage >= 0
            if (valid_coverage and
                    (not best_has_valid_coverage or this_avg_interval_width < best_interval_width)) or \
               (not valid_coverage and
                    not best_has_valid_coverage and this_delta_coverage > best_delta_coverage):
                best_delta_coverage = this_delta_coverage
                best_interval_width = this_avg_interval_width
                best_epoch = i+1
                best_model = copy.deepcopy(hopfield_net.state_dict())

    # load best model
    print("{} best model: epoch {}".format(key, best_epoch))
    hopfield_net.load_state_dict(best_model)

    test_context_dataloader, test_target_y_dataloader, \
        test_residual_dataloader, test_prediction_dataloader = initialize_test_dataloader(data,
                                                                                         train_size,
                                                                                         valid_size,
                                                                                         config.model.prediction_step,
                                                                                         config.model.memory_size,
                                                                                         config.data.normalize)

    evaluation_results = {
        tuple(confidence_pair): {"coverage": [],
                                 "interval_width": [],
                                 "winkler_score" : [],
                                 "upper_interval" : [],
                                 "lower_interval" : [],
                                 "target_y" : [],
                                 "target_predictions" : []}
        for confidence_pair in target_quantiles
    }

    hopfield_net.eval()
    for j in range(test_dataloader_size):

        strided_context, target_context = next(test_context_dataloader)
        _, target_y = next(test_target_y_dataloader)
        strided_residual, target_residual = next(test_residual_dataloader)
        _, target_predictions = next(test_prediction_dataloader)

        with torch.no_grad():
            memory_feature, query_feature = generate_feature_hopcpt_test(strided_context,
                                                                         target_context)
            memory_feature = memory_feature.to(device) # (batch_size, memeory_length, feature_dim)
            query_feature = query_feature.to(device) # (batch_size, 1, feature_dim)
            # (batch_size, 1, 1, memory_length)
            association_matrix = hopfield_net.obtain_association_matrix(memory_feature, query_feature)

        for confidence_pair in target_quantiles:
            tuple_confidence_pair = tuple(confidence_pair)
            lo, hi = estimate_hopcpt_residual_interval(association_matrix,
                                                       strided_residual,
                                                       tuple_confidence_pair,
                                                       config.model.sampling_num,
                                                       conformal_absolute_residual)
            this_coverage = compute_coverage(hi, lo, target_residual)
            this_interval_width = compute_interval_width(hi,
                                                         lo,
                                                         normalized_std=residual_normalized_std)
            this_winkler_score = compute_winkler_score(hi, lo,
                                                       target_y,
                                                       target_predictions,
                                                       tuple_confidence_pair,
                                                       normalized_params=residual_normalization_params)

            evaluation_results[tuple_confidence_pair]["upper_interval"].extend(hi.cpu().detach().tolist())
            evaluation_results[tuple_confidence_pair]["lower_interval"].extend(lo.cpu().detach().tolist())
            evaluation_results[tuple_confidence_pair]["coverage"].extend(this_coverage)
            evaluation_results[tuple_confidence_pair]["interval_width"].extend(this_interval_width)
            evaluation_results[tuple_confidence_pair]["winkler_score"].extend(this_winkler_score)
            evaluation_results[tuple_confidence_pair]["target_y"].extend(target_y.flatten().tolist())
            evaluation_results[tuple_confidence_pair]["target_predictions"].extend(target_predictions.flatten().tolist())
            if residual_normalization_params is not None:
                evaluation_results[tuple_confidence_pair]["train_residuals_mu"] = residual_normalization_params[0]
                evaluation_results[tuple_confidence_pair]["train_residuals_std"] = residual_normalization_params[1]

    for confidence_pair in target_quantiles:
        tuple_confidence_pair = tuple(confidence_pair)
        target_alpha = max(tuple_confidence_pair) - min(tuple_confidence_pair)
        avg_coverage = np.mean(evaluation_results[tuple_confidence_pair]["coverage"])
        avg_delta_coverage = avg_coverage - target_alpha
        avg_interval_width = np.mean(evaluation_results[tuple_confidence_pair]["interval_width"])
        avg_winkler_score = np.mean(evaluation_results[tuple_confidence_pair]["winkler_score"])
        print("{} avg coverage: {}".format(key, avg_coverage))
        print("{} avg delta coverage: {}".format(key, avg_delta_coverage))
        print("{} avg interval width: {}".format(key, avg_interval_width))
        print("{} avg winkler score: {}".format(key, avg_winkler_score))
        evaluation_results[tuple_confidence_pair]["avg_coverage"] = avg_coverage
        evaluation_results[tuple_confidence_pair]["avg_delta_coverage"] = avg_delta_coverage
        evaluation_results[tuple_confidence_pair]["avg_interval_width"] = avg_interval_width
        evaluation_results[tuple_confidence_pair]["avg_winkler_score"] = avg_winkler_score

    sequence_log = {"train_loss" : train_loss,
                    "valid_coverages" : valid_coverages,
                    "valid_delta_coverages" : valid_delta_coverages,
                    "valid_interval_widths" : valid_interval_widths,
                    "best_epoch" : best_epoch,
                    "best_delta_coverage" : best_delta_coverage,
                    "best_interval_width" : best_interval_width,
                    "evaluation_results" : evaluation_results}

    torch.save(best_model, os.path.join(config.saving_dir, key + '_model.pt'))
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return key, sequence_log


def _run_hopcpt_sequence_chunk(items, config, device):
    config = OmegaConf.create(config)
    if device != "cpu":
        torch.cuda.set_device(torch.device(device))
    chunk_log = {}
    for key, data in tqdm(items, desc="sequences on {}".format(device)):
        key, sequence_log = _run_hopcpt_sequence(key, data, config, device)
        chunk_log[key] = sequence_log
    return chunk_log


def _initialize_hopcpt_model(data, config, device):
    dim_feature = data["heldout_context"].shape[-1]
    dim_context_encoding = config.model.dim_context_encoding
    if dim_context_encoding is None or dim_context_encoding == "auto":
        dim_context_encoding = dim_feature
    dim_hopfield_hidden = config.model.dim_hopfield_hidden
    if dim_hopfield_hidden is None or dim_hopfield_hidden == "auto":
        dim_hopfield_hidden = dim_context_encoding

    hopfield_net = HopfieldNet(dim_feature,
                               dim_context_encoding,
                               dim_hopfield_hidden,
                               config.model.beta,
                               config.model.use_temporal_encoding)
    hopfield_net.to(device)
    return hopfield_net


def _build_sequence_training_batch(cpd_data, keys, device):
    memory_features = []
    memory_residuals = []
    memory_lengths = set()
    residual_lengths = set()
    feature_dims = set()

    for key in keys:
        data = cpd_data[key]
        memory_feature = generate_feature_hopcpt_training(data["heldout_train_context"])
        memory_residual = torch.from_numpy(data["heldout_train_residual"]).to(torch.float32)

        memory_lengths.add(memory_feature.shape[1])
        residual_lengths.add(memory_residual.shape[0])
        feature_dims.add(memory_feature.shape[-1])
        memory_features.append(memory_feature)
        memory_residuals.append(memory_residual)

    if len(memory_lengths) != 1 or len(residual_lengths) != 1:
        raise ValueError(
            "Sequence-batch HopCPT requires equal training memory lengths across "
            "batched sequences. Got context lengths {} and residual lengths {}."
            .format(sorted(memory_lengths), sorted(residual_lengths))
        )
    if len(feature_dims) != 1:
        raise ValueError(
            "Sequence-batch HopCPT requires equal feature dimensions across "
            "batched sequences. Got feature dimensions {}.".format(sorted(feature_dims))
        )

    memory_feature = torch.cat(memory_features, dim=0).to(device)
    memory_residual = torch.stack(memory_residuals, dim=0).to(device)
    return memory_feature, memory_residual


def _sequence_batch_size(config, num_sequences):
    batch_size = OmegaConf.select(config, "training.sequence_batch_size", default=None)
    if batch_size is None or batch_size == "all":
        return num_sequences
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("training.sequence_batch_size must be a positive integer or 'all'.")
    return min(batch_size, num_sequences)


def _collect_hopcpt_split_metrics(hopfield_net,
                                  data,
                                  config,
                                  device,
                                  split,
                                  target_quantiles):
    train_size = data["train_size"]
    valid_size = data["valid_size"]
    if split == "valid":
        dataloader_size = data["valid_dataloader_size"]
        context_dataloader, target_y_dataloader, \
            residual_dataloader, prediction_dataloader = initialize_valid_dataloader(data,
                                                                                    train_size,
                                                                                    valid_size,
                                                                                    config.model.prediction_step,
                                                                                    config.model.memory_size,
                                                                                    config.data.normalize)
    elif split == "test":
        dataloader_size = data["test_dataloader_size"]
        context_dataloader, target_y_dataloader, \
            residual_dataloader, prediction_dataloader = initialize_test_dataloader(data,
                                                                                   train_size,
                                                                                   valid_size,
                                                                                   config.model.prediction_step,
                                                                                   config.model.memory_size,
                                                                                   config.data.normalize)
    else:
        raise ValueError("split must be 'valid' or 'test'.")

    conformal_absolute_residual = OmegaConf.select(
        config, "model.conformal_absolute_residual",
        default=OmegaConf.select(config, "model.use_absolute_residual", default=False))
    residual_normalization_params = _hopcpt_residual_normalization_params(data,
                                                                          config.data.normalize)
    residual_normalized_std = (residual_normalization_params[1]
                               if residual_normalization_params is not None
                               else None)

    split_results = {
        tuple(confidence_pair): {"coverage": [],
                                 "interval_width": [],
                                 "winkler_score": [],
                                 "upper_interval": [],
                                 "lower_interval": [],
                                 "target_y": [],
                                 "target_predictions": []}
        for confidence_pair in target_quantiles
    }

    hopfield_net.eval()
    for _ in range(dataloader_size):
        strided_context, target_context = next(context_dataloader)
        _, target_y = next(target_y_dataloader)
        strided_residual, target_residual = next(residual_dataloader)
        _, target_predictions = next(prediction_dataloader)

        with torch.no_grad():
            memory_feature, query_feature = generate_feature_hopcpt_test(strided_context,
                                                                         target_context)
            memory_feature = memory_feature.to(device)
            query_feature = query_feature.to(device)
            association_matrix = hopfield_net.obtain_association_matrix(memory_feature,
                                                                        query_feature)

        for confidence_pair in target_quantiles:
            tuple_confidence_pair = tuple(confidence_pair)
            lo, hi = estimate_hopcpt_residual_interval(association_matrix,
                                                       strided_residual,
                                                       tuple_confidence_pair,
                                                       config.model.sampling_num,
                                                       conformal_absolute_residual)
            this_coverage = compute_coverage(hi, lo, target_residual)
            this_interval_width = compute_interval_width(hi,
                                                         lo,
                                                         normalized_std=residual_normalized_std)

            split_results[tuple_confidence_pair]["coverage"].extend(this_coverage)
            split_results[tuple_confidence_pair]["interval_width"].extend(this_interval_width)

            if split == "test":
                this_winkler_score = compute_winkler_score(hi, lo,
                                                           target_y,
                                                           target_predictions,
                                                           tuple_confidence_pair,
                                                           normalized_params=residual_normalization_params)
                split_results[tuple_confidence_pair]["upper_interval"].extend(hi.cpu().detach().tolist())
                split_results[tuple_confidence_pair]["lower_interval"].extend(lo.cpu().detach().tolist())
                split_results[tuple_confidence_pair]["winkler_score"].extend(this_winkler_score)
                split_results[tuple_confidence_pair]["target_y"].extend(target_y.flatten().tolist())
                split_results[tuple_confidence_pair]["target_predictions"].extend(target_predictions.flatten().tolist())
                if residual_normalization_params is not None:
                    split_results[tuple_confidence_pair]["train_residuals_mu"] = residual_normalization_params[0]
                    split_results[tuple_confidence_pair]["train_residuals_std"] = residual_normalization_params[1]

    return split_results


def _evaluate_hopcpt_sequence_with_model(key, data, config, device, hopfield_net):
    target_quantiles = config.model.target_quantiles
    evaluation_results = _collect_hopcpt_split_metrics(hopfield_net,
                                                       data,
                                                       config,
                                                       device,
                                                       "test",
                                                       target_quantiles)

    for confidence_pair in target_quantiles:
        tuple_confidence_pair = tuple(confidence_pair)
        target_alpha = max(tuple_confidence_pair) - min(tuple_confidence_pair)
        avg_coverage = np.mean(evaluation_results[tuple_confidence_pair]["coverage"])
        avg_delta_coverage = avg_coverage - target_alpha
        avg_interval_width = np.mean(evaluation_results[tuple_confidence_pair]["interval_width"])
        avg_winkler_score = np.mean(evaluation_results[tuple_confidence_pair]["winkler_score"])
        print("{} avg coverage: {}".format(key, avg_coverage))
        print("{} avg delta coverage: {}".format(key, avg_delta_coverage))
        print("{} avg interval width: {}".format(key, avg_interval_width))
        print("{} avg winkler score: {}".format(key, avg_winkler_score))
        evaluation_results[tuple_confidence_pair]["avg_coverage"] = avg_coverage
        evaluation_results[tuple_confidence_pair]["avg_delta_coverage"] = avg_delta_coverage
        evaluation_results[tuple_confidence_pair]["avg_interval_width"] = avg_interval_width
        evaluation_results[tuple_confidence_pair]["avg_winkler_score"] = avg_winkler_score

    return evaluation_results


def run_hopcpt_sequence_batch(config_path):
    """
    Train one HopCPT model with independent sequences stacked as the batch axis.

    The original run_hopcpt() trains one HopCPT model per sequence. This runner
    follows the paper-style setup where each independent sequence contributes one
    batch row: memory_feature has shape (num_sequences, memory_length, dim).
    """

    config = OmegaConf.load(config_path)
    os.makedirs(config.saving_dir, exist_ok=True)

    data = load_data(config.data.data_path)
    base_predictor, data_type = read_setup(config.data.data_path)
    cpd = ConformalPredictionData(data)
    predict_absolute_residual = OmegaConf.select(
        config, "model.predict_absolute_residual",
        default=OmegaConf.select(config, "model.use_absolute_residual", default=True))
    conformal_absolute_residual = OmegaConf.select(
        config, "model.conformal_absolute_residual",
        default=OmegaConf.select(config, "model.use_absolute_residual", default=False))

    cpd.prepare_hopcpt_datasets(config.model.prediction_step,
                                config.model.y_lags,
                                config.data.train_ratio,
                                config.data.valid_ratio,
                                config.data.normalize,
                                predict_absolute_residual,
                                conformal_absolute_residual)

    keys = list(cpd.data.keys())
    if not keys:
        raise ValueError("No sequences found in HopCPT data.")

    device = _device_from_config(config.device)
    target_quantiles = config.model.target_quantiles
    selection_confidence_pair = tuple(target_quantiles[0])
    selection_target_coverage = max(selection_confidence_pair) - min(selection_confidence_pair)

    print("Experiment setup")
    print("Method: HopCPT sequence batch")
    print("Base predictor: {}".format(base_predictor))
    print("Data: {}".format(data_type))
    print("{} independent sequences as one batch".format(len(keys)))

    hopfield_net = _initialize_hopcpt_model(cpd.data[keys[0]], config, device)
    optimizer = torch.optim.AdamW(hopfield_net.parameters(),
                                  lr=config.training.learning_rate)
    memory_feature, memory_residual = _build_sequence_training_batch(cpd.data,
                                                                     keys,
                                                                     device)
    sequence_batch_size = _sequence_batch_size(config, len(keys))
    average_batch_loss = OmegaConf.select(config,
                                          "training.average_sequence_batch_loss",
                                          default=True)
    shuffle_sequence_batch = OmegaConf.select(config,
                                             "training.shuffle_sequence_batch",
                                             default=True)
    print("Sequence training batch size: {}".format(sequence_batch_size))

    train_loss = []
    valid_coverages = []
    valid_delta_coverages = []
    valid_interval_widths = []
    best_interval_width = np.inf
    best_delta_coverage = -np.inf
    best_epoch = 0
    best_model = copy.deepcopy(hopfield_net.state_dict())

    epoch_iter = tqdm(range(config.training.epochs),
                      desc="sequence-batch HopCPT training on {}".format(device),
                      leave=False)
    for i in epoch_iter:
        hopfield_net.train()
        if shuffle_sequence_batch:
            sequence_order = torch.randperm(len(keys), device=device)
        else:
            sequence_order = torch.arange(len(keys), device=device)

        epoch_loss = 0.0
        num_batches = 0
        for start_idx in range(0, len(keys), sequence_batch_size):
            batch_idx = sequence_order[start_idx:start_idx + sequence_batch_size]
            optimizer.zero_grad()
            loss = compute_hopfield_net_loss(hopfield_net,
                                             memory_feature[batch_idx],
                                             memory_residual[batch_idx],
                                             predict_absolute_residual)
            if average_batch_loss:
                loss = loss / batch_idx.numel()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss = epoch_loss / num_batches
        train_loss.append(epoch_loss)
        epoch_iter.set_postfix(loss=epoch_loss)

        if (i + 1) % config.training.validation_epochs == 0:
            this_coverages = []
            this_interval_widths = []
            for key in keys:
                valid_results = _collect_hopcpt_split_metrics(hopfield_net,
                                                              cpd.data[key],
                                                              config,
                                                              device,
                                                              "valid",
                                                              [selection_confidence_pair])
                result = valid_results[selection_confidence_pair]
                this_coverages.extend(result["coverage"])
                this_interval_widths.extend(result["interval_width"])

            this_delta_coverage = np.mean(this_coverages) - selection_target_coverage
            valid_coverages.append(np.mean(this_coverages))
            valid_delta_coverages.append(this_delta_coverage)
            this_avg_interval_width = np.mean(this_interval_widths)
            valid_interval_widths.append(this_avg_interval_width)
            epoch_iter.set_postfix(loss=epoch_loss,
                                   valid_coverage=valid_coverages[-1],
                                   valid_delta=this_delta_coverage,
                                   valid_width=this_avg_interval_width)

            valid_coverage = this_delta_coverage >= 0
            best_has_valid_coverage = best_delta_coverage >= 0
            if (valid_coverage and
                    (not best_has_valid_coverage or this_avg_interval_width < best_interval_width)) or \
               (not valid_coverage and
                    not best_has_valid_coverage and this_delta_coverage > best_delta_coverage):
                best_delta_coverage = this_delta_coverage
                best_interval_width = this_avg_interval_width
                best_epoch = i + 1
                best_model = copy.deepcopy(hopfield_net.state_dict())

    print("sequence-batch best model: epoch {}".format(best_epoch))
    hopfield_net.load_state_dict(best_model)
    torch.save(best_model, os.path.join(config.saving_dir, "sequence_batch_model.pt"))

    log = {}
    for key in tqdm(keys, desc="sequence-batch evaluation over independent sequences"):
        evaluation_results = _evaluate_hopcpt_sequence_with_model(key,
                                                                  cpd.data[key],
                                                                  config,
                                                                  device,
                                                                  hopfield_net)
        log[key] = {"train_loss": train_loss,
                    "valid_coverages": valid_coverages,
                    "valid_delta_coverages": valid_delta_coverages,
                    "valid_interval_widths": valid_interval_widths,
                    "best_epoch": best_epoch,
                    "best_delta_coverage": best_delta_coverage,
                    "best_interval_width": best_interval_width,
                    "evaluation_results": evaluation_results}
        save_data(os.path.join(config.saving_dir, "log.pkl"), log)

    summary_results = summarize_evaluation_results(log, target_quantiles)

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

    if OmegaConf.select(config, "plotting.plotting", default=False):
        plot_cp_prediction_intervals(log,
                                     target_quantiles,
                                     config.plotting.plotting_seq_len,
                                     os.path.join(config.saving_dir, "plots"))

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_hopcpt(config_path):

    config = OmegaConf.load(config_path)
    os.makedirs(config.saving_dir, exist_ok=True)

    # load data
    data = load_data(config.data.data_path) # load predictor results here
    base_predictor, data_type = read_setup(config.data.data_path)
    cpd = ConformalPredictionData(data)
    predict_absolute_residual = OmegaConf.select(
        config, "model.predict_absolute_residual",
        default=OmegaConf.select(config, "model.use_absolute_residual", default=True))
    conformal_absolute_residual = OmegaConf.select(
        config, "model.conformal_absolute_residual",
        default=OmegaConf.select(config, "model.use_absolute_residual", default=False))

    cpd.prepare_hopcpt_datasets(config.model.prediction_step,
                                config.model.y_lags,
                                config.data.train_ratio,
                                config.data.valid_ratio,
                                config.data.normalize,
                                predict_absolute_residual,
                                conformal_absolute_residual)

    device = config.device
    target_quantiles = config.model.target_quantiles
    log = dict()

    print("Experiment setup")
    print("Method: HopCPT")
    print("Base predictor: {}".format(base_predictor))
    print("Data: {}".format(data_type))
    print("{} independent sequences".format(len(cpd.data)))

    parallel_devices = _parallel_devices(config)
    if parallel_devices is None:
        device = _device_from_config(device)
        for key, data in tqdm(cpd.data.items(), desc="repetition over independent sequences"):
            key, sequence_log = _run_hopcpt_sequence(key, data, config, device)
            log[key] = sequence_log
            save_data(os.path.join(config.saving_dir, "log.pkl"), log)
    else:
        print("Parallel HopCPT devices: {}".format(parallel_devices))
        key_chunks = _split_keys_by_device(list(cpd.data.keys()), parallel_devices)
        config_payload = OmegaConf.to_container(config, resolve=True)
        with ProcessPoolExecutor(max_workers=len(key_chunks)) as executor:
            futures = []
            for chunk_device, chunk_keys in key_chunks.items():
                items = [(key, cpd.data[key]) for key in chunk_keys]
                futures.append(executor.submit(_run_hopcpt_sequence_chunk,
                                               items,
                                               config_payload,
                                               chunk_device))
            for future in as_completed(futures):
                log.update(future.result())
                save_data(os.path.join(config.saving_dir, "log.pkl"), log)

    summary_results = summarize_evaluation_results(log, target_quantiles)

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

    if OmegaConf.select(config, "plotting.plotting", default=False):
        plot_cp_prediction_intervals(log,
                                     target_quantiles,
                                     config.plotting.plotting_seq_len,
                                     os.path.join(config.saving_dir, "plots"))
                                       
