import math
from numbers import Integral, Real

import torch
from utils.utils import get_sorted_unique_quantiles

from .models.iqn import ImplicitQuantileNetwork


def compute_loss_quantile_regression_transformer(model, x, target, target_quantiles, current_feature=None):
    """
    compute quantile loss for quantile regression Transformer
    :param x: input, (batch_size, window_size, feature_dim)
    :param target: target_residual, (batch_size, 1)
    :param target_quantiles: list of interval quantile pairs, [(0.975, 0.025), ...]
    :param current_feature: Optional, (batch_size, 1, current_feature_dim)
    """
    device = x.device

    target = target.unsqueeze(-1) # (batch_size, 1, 1)
    target_quantiles = torch.tensor(
        get_sorted_unique_quantiles(target_quantiles),
        dtype=torch.float32,
        device=device,
    )
    causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(device)

    if current_feature != None:
        # (batch_size, window_size, len(target_quantiles))
        output = model(x, src_mask=causal_mask, src_key_padding_mask=None, current_feature=current_feature)
    else:
        # (batch_size, window_size, len(target_quantiles))
        output = model(x, src_mask=causal_mask, src_key_padding_mask=None) 
        
    errors = target - output[:, -1, :].unsqueeze(1) # (batch_size, 1, num_quantiles)
    loss_tensor = torch.max((target_quantiles - 1) * errors, target_quantiles * errors) # (batch_size, 1, num_quantiles)

    return loss_tensor.mean()


def compute_loss_quantile_regression_rnn(model, x, target, target_quantiles, current_feature=None):
    """
    compute quantile loss for quantile regression RNN
    :param x: input, (batch_size, window_size, feature_dim)
    :param target: target_residual, (batch_size, 1)
    :param target_quantiles: list of interval quantile pairs, [(0.975, 0.025), ...]
    :param current_feature: Optional, (batch_size, 1, current_feature_dim)
    """
    device = x.device

    target = target.unsqueeze(-1) # (batch_size, 1, 1)
    target_quantiles = torch.tensor(
        get_sorted_unique_quantiles(target_quantiles),
        dtype=torch.float32,
        device=device,
    )
    
    if current_feature != None:
        # (batch_size, window_size, len(target_quantiles))
        output = model(x, current_feature=current_feature)
    else:
        # (batch_size, window_size, len(target_quantiles))
        output = model(x)
        
    errors = target - output[:, -1, :].unsqueeze(1) # (batch_size, 1, num_quantiles)
    loss_tensor = torch.max((target_quantiles - 1) * errors, target_quantiles * errors) # (batch_size, 1, num_quantiles)

    return loss_tensor.mean()


def compute_loss_transformer_predictor(transformer_predictor, x, target, current_feature=None):
    """
    Train the Local-CP Transformer encoder with mean fixed-quantile pinball loss.
    :param x: input, (batch_size, window_size, feature_dim)
    :param target: target_residual, (batch_size, 1)
    Quantile levels come from the model's training_quantiles buffer.
    """
    device = x.device
    causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(device)

    output = transformer_predictor(x,
                                   src_mask=causal_mask,
                                   src_key_padding_mask=None,
                                   current_feature=current_feature) # (batch_size, window_size, num_quantiles)

    return _local_cp_quantile_loss(transformer_predictor, output[:, -1, :], target)


def compute_loss_rnn_predictor(rnn_predictor, x, target, current_feature=None):
    """
    Train the Local-CP RNN encoder with mean fixed-quantile pinball loss.
    :param x: input, (batch_size, window_size, feature_dim)
    :param target: target_residual, (batch_size, 1)
    Quantile levels come from the model's training_quantiles buffer.
    """
    output = rnn_predictor(x, current_feature=current_feature)  # (batch_size, window_size, num_quantiles)

    return _local_cp_quantile_loss(rnn_predictor, output[:, -1, :], target)


def _local_cp_quantile_loss(model, quantile_predictions, target):
    if target.ndim != 2 or target.shape != (quantile_predictions.shape[0], 1):
        raise ValueError("Local-CP quantile training requires target shape (batch_size, 1).")
    errors = target - quantile_predictions
    taus = model.training_quantiles
    return torch.maximum((taus - 1.0) * errors, taus * errors).mean()


def resolve_iqn_validation_quantiles(training_config, target_quantiles):
    """Resolve checkpoint loss without changing the sampled IQN training objective.

    Target mode uses each distinct interval endpoint once. Sampled mode preserves
    the legacy validation behavior and does not supply explicit quantile levels.
    """
    mode = str(training_config.get("validation_loss", "sampled_quantiles")).strip().lower()
    if mode not in ("sampled_quantiles", "target_quantiles"):
        raise ValueError(
            "training.validation_loss must be 'sampled_quantiles' or 'target_quantiles'."
        )
    if mode == "sampled_quantiles":
        return mode, None

    error = (
        "model.target_quantiles must contain nonempty interval pairs of distinct, "
        "finite quantile levels strictly between 0 and 1 for target_quantiles validation."
    )
    try:
        pairs = list(target_quantiles)
    except TypeError as exc:
        raise ValueError(error) from exc
    if not pairs:
        raise ValueError(error)
    levels = set()
    for pair in pairs:
        try:
            endpoints = list(pair)
        except TypeError as exc:
            raise ValueError(error) from exc
        if len(endpoints) != 2 or any(
            isinstance(level, bool)
            or not isinstance(level, Real)
            or not math.isfinite(level)
            or not 0.0 < level < 1.0
            for level in endpoints
        ):
            raise ValueError(error)
        if endpoints[0] == endpoints[1]:
            raise ValueError(error)
        levels.update(float(level) for level in endpoints)
    return mode, sorted(levels)


@torch.no_grad()
def compute_iqn_interval_validation_loss(
    model,
    x,
    target,
    quantiles,
    current_feature=None,
    *,
    sampling_seed=None,
):
    """Score requested endpoints using the model's deployed interval rule.

    The caller must put the model in evaluation mode. Cosine heads use their
    configured direct or sampling interval mode, including ``sampling_num``;
    partially monotonic heads always evaluate the requested levels directly.
    This validation-only helper does not change the sampled training objective.

    A supplied seed gives reproducible Monte Carlo draws without advancing the
    caller's CPU or selected CUDA-device RNG state. Runners reuse a seed for
    each validation batch across epochs so checkpoint comparisons use common
    random numbers.
    """
    if target.ndim != 2 or target.shape != (x.shape[0], 1):
        raise ValueError("IQN interval validation requires target shape (batch_size, 1).")
    taus = ImplicitQuantileNetwork._prepare_taus(
        taus=quantiles,
        batch_size=x.shape[0],
        device=x.device,
        dtype=x.dtype,
    )
    if taus.shape[1] == 0:
        raise ValueError("IQN interval validation requires at least one quantile level.")

    def predict():
        return model.predict_quantiles(
            src=x,
            quantiles=taus,
            current_feature=current_feature,
        )

    if sampling_seed is None:
        quantile_values = predict()
    else:
        if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, Integral):
            raise TypeError("sampling_seed must be an integer or None.")
        cuda_devices = [x.device.index] if x.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            # Seed only the generators whose states are saved by fork_rng.
            # torch.manual_seed would also modify unrelated CUDA-device states.
            cpu_generator = torch.Generator(device="cpu").manual_seed(int(sampling_seed))
            torch.set_rng_state(cpu_generator.get_state())
            if x.device.type == "cuda":
                device_generator = torch.Generator(device=x.device).manual_seed(
                    int(sampling_seed)
                )
                torch.cuda.set_rng_state(device_generator.get_state(), device=x.device)
            quantile_values = predict()

    if quantile_values.shape != taus.shape:
        raise ValueError("IQN interval predictions must have shape (batch_size, num_quantiles).")
    errors = target.to(device=quantile_values.device) - quantile_values
    taus = taus.to(dtype=quantile_values.dtype)
    return torch.maximum((taus - 1.0) * errors, taus * errors).mean()


def compute_loss_iqn_transformer(model, x, target, num_taus, current_feature=None, *, taus=None):
    """
    Compute IQN pinball loss at sampled or explicitly requested quantile levels.

    :param x: input, (batch_size, window_size, feature_dim)
    :param target: target_residual, (batch_size, 1)
    :param num_taus: number of quantile fractions to sample per instance
    :param taus: optional fixed levels; None preserves full-range sampled training
    """
    device = x.device
    target = target.to(device)
    causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(device)
    quantile_values, taus = model(
        x,
        current_feature=current_feature,
        num_taus=num_taus,
        taus=taus,
        src_mask=causal_mask,
        src_key_padding_mask=None,
    )

    errors = target - quantile_values
    loss_tensor = torch.max((taus - 1.0) * errors, taus * errors)
    return loss_tensor.mean()


def compute_loss_iqn_rnn(model, x, target, num_taus, current_feature=None, *, taus=None):
    """
    Compute IQN pinball loss at sampled or explicitly requested quantile levels.

    :param x: input, (batch_size, window_size, feature_dim)
    :param target: target_residual, (batch_size, 1)
    :param num_taus: number of quantile fractions to sample per instance
    :param taus: optional fixed levels; None preserves full-range sampled training
    """
    device = x.device
    target = target.to(device)
    quantile_values, taus = model(
        x,
        current_feature=current_feature,
        num_taus=num_taus,
        taus=taus,
    )

    errors = target - quantile_values
    loss_tensor = torch.max((taus - 1.0) * errors, taus * errors)
    return loss_tensor.mean()
