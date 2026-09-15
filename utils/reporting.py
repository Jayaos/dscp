import torch
import numpy as np


def compute_coverage(upper_interval, lower_interval, target):

    if type(upper_interval) == np.ndarray:
        upper_interval = torch.from_numpy(upper_interval)

    if type(lower_interval) == np.ndarray:
        lower_interval = torch.from_numpy(lower_interval)

    if target.ndim == 2:
        target = target.flatten()

    upper_tf = (target <= upper_interval)
    lower_tf = (target >= lower_interval)
    coverage_tf = upper_tf * lower_tf

    return coverage_tf.detach().cpu().tolist()


def compute_interval_width(upper_interval, lower_interval, normalized_std=None):
    """
    if normalized_params are given, compute interval width 
    """

    if type(upper_interval) == np.ndarray:
        upper_interval = torch.from_numpy(upper_interval)

    if type(lower_interval) == np.ndarray:
        lower_interval = torch.from_numpy(lower_interval)

    if normalized_std:
        interval_width = (upper_interval - lower_interval) * normalized_std
    else:
        interval_width = upper_interval - lower_interval

    return interval_width.detach().cpu().tolist()


def construct_interval_endpoints(
    upper_residual_quantile,
    lower_residual_quantile,
    preds,
    normalized_params=None,
):
    """Convert residual-quantile outputs to prediction-interval endpoints.

    ``preds`` must be on the original response scale. Residual quantiles are
    assumed to be on the model scale; when ``normalized_params`` is provided,
    they are converted back to the original residual scale before being added
    to the base prediction.

    Returns the upper endpoint followed by the lower endpoint.
    """
    if type(upper_residual_quantile) == np.ndarray:
        upper_residual_quantile = torch.from_numpy(upper_residual_quantile)

    if type(lower_residual_quantile) == np.ndarray:
        lower_residual_quantile = torch.from_numpy(lower_residual_quantile)

    if type(preds) == np.ndarray:
        preds = torch.from_numpy(preds)

    if preds.ndim == 2:
        preds = preds.flatten()

    if normalized_params is not None:
        device = upper_residual_quantile.device
        mean = torch.tensor(normalized_params[0], dtype=torch.float32, device=device)
        std = torch.tensor(normalized_params[1], dtype=torch.float32, device=device)
        upper_residual_value = upper_residual_quantile * std + mean
        lower_residual_value = lower_residual_quantile * std + mean
    else:
        upper_residual_value = upper_residual_quantile
        lower_residual_value = lower_residual_quantile

    upper_interval = preds + upper_residual_value
    lower_interval = preds + lower_residual_value
    return upper_interval, lower_interval


def compute_winkler_score(upper_interval, lower_interval, y, preds, confidence_pair, normalized_params=None):
    """Compute the (possibly asymmetric) Winkler interval score.

    When ``confidence_pair`` contains the two quantile levels defining the
    interval, the lower- and upper-tail penalties are weighted separately. For
    lower and upper quantile levels ``tau_l`` and ``tau_u``, respectively, the
    score is

        (upper - lower)
        + 1(y < lower) * (lower - y) / tau_l
        + 1(y > upper) * (y - upper) / (1 - tau_u).

    This reduces to the usual Winkler score with penalty ``2 / alpha`` on both
    sides for an equal-tailed interval. A scalar ``confidence_pair`` is treated
    as ``alpha`` and retains that equal-tailed behavior.

    The supplied quantile levels must describe the actual lower and upper
    interval endpoints. Their order in ``confidence_pair`` does not matter.

    :param upper_interval: upper residual endpoint, shape (size)
    :param lower_interval: lower residual endpoint, shape (size)
    :param y: observed response, shape (size)
    :param preds: base prediction, shape (size)
    :param confidence_pair: two endpoint quantile levels or scalar alpha
    :param normalized_params: optional residual-normalization (mean, std)
    """
    is_scalar_alpha = np.isscalar(confidence_pair) or (
        hasattr(confidence_pair, "ndim") and confidence_pair.ndim == 0
    )
    if is_scalar_alpha:
        alpha = float(confidence_pair)
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be finite and strictly between 0 and 1.")
        lower_penalty_weight = 2.0 / alpha
        upper_penalty_weight = 2.0 / alpha
    else:
        try:
            quantile_levels = [float(level) for level in confidence_pair]
        except TypeError as exc:
            raise TypeError(
                "confidence_pair must be a scalar alpha or two quantile levels."
            ) from exc

        if len(quantile_levels) != 2:
            raise ValueError("confidence_pair must contain exactly two quantile levels.")

        lower_quantile, upper_quantile = sorted(quantile_levels)
        if (
            not np.isfinite(lower_quantile)
            or not np.isfinite(upper_quantile)
            or not 0.0 < lower_quantile < upper_quantile < 1.0
        ):
            raise ValueError(
                "quantile levels must be finite and satisfy "
                "0 < lower_quantile < upper_quantile < 1."
            )

        lower_penalty_weight = 1.0 / lower_quantile
        upper_penalty_weight = 1.0 / (1.0 - upper_quantile)

    if y.ndim == 2:
        y = y.flatten()

    upper_interval_value, lower_interval_value = construct_interval_endpoints(
        upper_interval,
        lower_interval,
        preds,
        normalized_params=normalized_params,
    )

    interval_width = upper_interval_value - lower_interval_value

    upper_fail = y > upper_interval_value
    lower_fail = y < lower_interval_value
    upper_fail_penalty = (
        upper_fail * (y - upper_interval_value) * upper_penalty_weight
    )
    lower_fail_penalty = (
        lower_fail * (lower_interval_value - y) * lower_penalty_weight
    )
    penalty = upper_fail_penalty + lower_fail_penalty
    
    return (interval_width + penalty).detach().cpu().tolist()


def summarize_evaluation_results(log, target_quantiles):

    summary_results = dict()

    for confidence_pair in target_quantiles:
        tuple_confidence_pair = tuple(confidence_pair)

        avg_coverages = []
        avg_delta_coverages = []
        avg_interval_widths = []
        avg_winkler_scores = []

        for key, item in log.items():
            if key == "summary_results":
                continue

            result = item["evaluation_results"][tuple_confidence_pair]
            avg_coverages.append(result["avg_coverage"])
            avg_delta_coverages.append(result["avg_delta_coverage"])
            avg_interval_widths.append(result["avg_interval_width"])
            avg_winkler_scores.append(result["avg_winkler_score"])

        summary_results[tuple_confidence_pair] = {
            "avg_coverage_mean": float(np.mean(avg_coverages)),
            "avg_coverage_std": float(np.std(avg_coverages)),
            "avg_delta_coverage_mean": float(np.mean(avg_delta_coverages)),
            "avg_delta_coverage_std": float(np.std(avg_delta_coverages)),
            "avg_interval_width_mean": float(np.mean(avg_interval_widths)),
            "avg_interval_width_std": float(np.std(avg_interval_widths)),
            "avg_winkler_score_mean": float(np.mean(avg_winkler_scores)),
            "avg_winkler_score_std": float(np.std(avg_winkler_scores)),
        }

    return summary_results
