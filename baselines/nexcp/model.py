import numpy as np


def get_nexcp_weights(length, mode, rho):
    if length <= 0:
        return np.empty(0, dtype=float)

    if mode in ("nexcp-ls", "nexcp-wls", "ls", "wls"):
        weights = np.power(float(rho), np.arange(length - 1, 0, -1, dtype=float))
        weights = np.r_[weights, 1.0]
        return weights / np.sum(weights)

    if mode in ("cp-ls", "cp"):
        return np.ones(length, dtype=float) / length

    raise ValueError(
        "model.method must be one of 'cp-ls', 'nexcp-ls', or 'nexcp-wls'."
    )


def weighted_abs_residual_quantile(past_residuals, alpha, mode, rho):
    past_eps = np.abs(np.asarray(past_residuals, dtype=float).reshape(-1))
    if len(past_eps) == 0:
        raise ValueError("NexCP requires at least one past residual.")

    weights = get_nexcp_weights(len(past_eps), mode, rho)
    sort_idx = np.argsort(past_eps)
    try:
        quantile_idx = np.min(np.where(np.cumsum(weights[sort_idx]) >= 1.0 - alpha))
        return np.sort(past_eps)[quantile_idx]
    except ValueError:
        return np.sort(past_eps)[-1]


def estimate_residual_interval(past_residuals, alpha, mode, rho):
    quantile_value = weighted_abs_residual_quantile(past_residuals,
                                                   alpha,
                                                   mode,
                                                   rho)
    return -quantile_value, quantile_value
