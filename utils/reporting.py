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


def compute_winkler_score(upper_interval, lower_interval, y, preds, confidence_pair, normalized_params=None):
    """
    :param upper_interval: (size)
    :param lower_interval: (size)
    :param y: (size)
    :param preds: (size)
    :param confidence_pair: (upper_confidence, lower_confidence) or alpha
    :param normalized_params: (mu, std)
    """
    try:
        device = upper_interval.device
    except:
        pass
    
    if isinstance(confidence_pair, tuple):
        alpha = 1 - (max(confidence_pair) - min(confidence_pair))
    else:
        alpha = confidence_pair

    if type(upper_interval) == np.ndarray:
        upper_interval = torch.from_numpy(upper_interval)

    if type(lower_interval) == np.ndarray:
        lower_interval = torch.from_numpy(lower_interval)

    if y.ndim == 2:
        y = y.flatten()

    if preds.ndim == 2:
        preds = preds.flatten()

    if normalized_params:
        mean = torch.tensor(normalized_params[0], dtype=torch.float32, device=device)
        std = torch.tensor(normalized_params[1], dtype=torch.float32, device=device)
        upper_interval_value =  upper_interval * std + mean + preds
        lower_interval_value =  lower_interval * std + mean + preds
        interval_width = upper_interval_value - lower_interval_value

        # binary, 1 if y is larger than upper_interval_value and 0 if y is smaller than upper_interval_value
        upper_fail = (y > upper_interval_value) 
        # binary, 1 if y is smaller than lower_interval_value and 0 if y is larger than lower_interval_value
        lower_fail = (y < lower_interval_value)

        upper_fail_penalty = upper_fail * (y - upper_interval_value) * (2/alpha)
        lower_fail_penalty = lower_fail * (lower_interval_value - y) * (2/alpha)
        penalty = upper_fail_penalty + lower_fail_penalty

    else:
        upper_interval_value =  upper_interval + preds
        lower_interval_value =  lower_interval + preds
        interval_width = upper_interval_value - lower_interval_value

        # binary, 1 if y is larger than upper_interval_value and 0 if y is smaller than upper_interval_value
        upper_fail = (y > upper_interval_value) 
        # binary, 1 if y is smaller than lower_interval_value and 0 if y is larger than lower_interval_value
        lower_fail = (y < lower_interval_value)

        upper_fail_penalty = upper_fail * (y - upper_interval_value) * (2/alpha)
        lower_fail_penalty = lower_fail * (lower_interval_value - y) * (2/alpha)
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
