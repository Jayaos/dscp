import torch
from utils.utils import get_sorted_unique_quantiles


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


def compute_loss_iqn_transformer(model, x, target, num_taus, current_feature=None):
    """
    Compute sampled quantile loss for an IQN Transformer.

    :param x: input, (batch_size, window_size, feature_dim)
    :param target: target_residual, (batch_size, 1)
    :param num_taus: number of quantile fractions to sample per instance
    """
    device = x.device
    target = target.to(device)
    causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(device)
    quantile_values, taus = model(
        x,
        current_feature=current_feature,
        num_taus=num_taus,
        src_mask=causal_mask,
        src_key_padding_mask=None,
    )

    errors = target - quantile_values
    loss_tensor = torch.max((taus - 1.0) * errors, taus * errors)
    return loss_tensor.mean()


def compute_loss_iqn_rnn(model, x, target, num_taus, current_feature=None):
    """
    Compute sampled quantile loss for an IQN RNN.

    :param x: input, (batch_size, window_size, feature_dim)
    :param target: target_residual, (batch_size, 1)
    :param num_taus: number of quantile fractions to sample per instance
    """
    device = x.device
    target = target.to(device)
    quantile_values, taus = model(
        x,
        current_feature=current_feature,
        num_taus=num_taus,
    )

    errors = target - quantile_values
    loss_tensor = torch.max((taus - 1.0) * errors, taus * errors)
    return loss_tensor.mean()
