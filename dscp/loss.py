import torch


def compute_loss_quantile_regression_transformer(model, x, target, target_quantiles, current_feature=None):
    """
    compute quantile loss for quantile regression Transformer
    :param x: input, (batch_size, window_size, feature_dim)
    :param target: target_residual, (batch_size, 1)
    :param target_quantiles: list of target quantiles, [(0.975, 0.025), ...]
    :param current_feature: Optional, (batch_size, 1, current_feature_dim)
    """
    device = x.device

    target = target.unsqueeze(-1) # (batch_size, 1, 1)
    target_quantiles = torch.tensor(target_quantiles, dtype=torch.float32, device=device).flatten() # (2*len(target_quantiles))
    causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(device)

    if current_feature != None:
        # (batch_size, window_size, len(target_quantiles))
        output = model(x, src_mask=causal_mask, src_key_padding_mask=None, current_feature=current_feature)
    else:
        # (batch_size, window_size, len(target_quantiles))
        output = model(x, src_mask=causal_mask, src_key_padding_mask=None) 
        
    errors = target - output[:, -1, :].unsqueeze(1) # (batch_size, 1, len(target_quantiles))
    loss_tensor = torch.max((target_quantiles - 1) * errors, target_quantiles * errors) # (batch_size, 1, len(target_quantiles))

    return loss_tensor.mean()


def compute_loss_quantile_regression_rnn(model, x, target, target_quantiles, current_feature=None):
    """
    compute quantile loss for quantile regression RNN
    :param x: input, (batch_size, window_size, feature_dim)
    :param target: target_residual, (batch_size, 1)
    :param target_quantiles: list of target quantiles, [(0.975, 0.025), ...]
    :param current_feature: Optional, (batch_size, 1, current_feature_dim)
    """
    device = x.device

    target = target.unsqueeze(-1) # (batch_size, 1, 1)
    target_quantiles = torch.tensor(target_quantiles, dtype=torch.float32, device=device).flatten() # (2*len(target_quantiles))
    
    if current_feature != None:
        # (batch_size, window_size, len(target_quantiles))
        output = model(x, current_feature=current_feature)
    else:
        # (batch_size, window_size, len(target_quantiles))
        output = model(x)
        
    errors = target - output[:, -1, :].unsqueeze(1) # (batch_size, 1, len(target_quantiles))
    loss_tensor = torch.max((target_quantiles - 1) * errors, target_quantiles * errors) # (batch_size, 1, len(target_quantiles))

    return loss_tensor.mean()


def compute_loss_transformer_predictor(transformer_predictor, x, target):
    """
    compute prediction loss for quantile transformer
    :param x: input, (batch_size, window_size, feature_dim)
    :param target: target_residual, (batch_size, prediction_step)
    :param target_quantiles: list of target quantiles, [(0.975, 0.025), ...]
    """
    device = x.device
    causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(device)

    output = transformer_predictor(x, src_mask=causal_mask, src_key_padding_mask=None) # (batch_size, window_size, prediction_step)
    errors = target - output[:, -1, :] # (batch_size, prediction_step)

    return (errors**2).mean()


def compute_loss_rnn_predictor(rnn_predictor, x, target):
    """
    compute prediction loss for RNN predictor
    :param x: input, (batch_size, window_size, feature_dim)
    :param target: target_residual, (batch_size, prediction_step)
    """
    output = rnn_predictor(x)  # (batch_size, window_size, prediction_step)
    errors = target - output[:, -1, :]  # (batch_size, prediction_step)

    return (errors**2).mean()
