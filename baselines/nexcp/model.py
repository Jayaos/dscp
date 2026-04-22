import numpy as np


def context_slice(target_idx, max_past):

    if max_past is None:
        return slice(0, target_idx)

    start_idx = max(0, target_idx - int(max_past))
    return slice(start_idx, target_idx)


def get_nexcp_weights(length, rho):

    weights = np.power(float(rho), np.arange(length - 1, 0, -1, dtype=float))
    weights = np.r_[weights, 1.0]

    return weights / np.sum(weights)


def weighted_abs_residual_quantile(past_residuals, alpha, rho):
    past_eps = np.abs(np.asarray(past_residuals, dtype=float).reshape(-1))
    if len(past_eps) == 0:
        raise ValueError("NexCP requires at least one past residual.")

    weights = get_nexcp_weights(len(past_eps), rho)
    sort_idx = np.argsort(past_eps)
    try:
        quantile_idx = np.min(np.where(np.cumsum(weights[sort_idx]) >= 1.0 - alpha))
        return np.sort(past_eps)[quantile_idx]
    except ValueError:
        return np.sort(past_eps)[-1]


def estimate_residual_interval(past_residuals, alpha, rho):
    quantile_value = weighted_abs_residual_quantile(past_residuals,
                                                    alpha,
                                                    rho)
    return -quantile_value, quantile_value
