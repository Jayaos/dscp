from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from pathlib import Path
import math


def load_data(data_dir):
    file = open(data_dir,'rb')
    
    return pickle.load(file)


def save_data(save_dir, data_dict):
    with open(save_dir, 'wb') as f:
        pickle.dump(data_dict, f)


def read_setup(data_path):
    """
    read setup from data path
    """
    path = Path(data_path)
    base_predictor, data_type, _ = path.stem.split("_")

    return base_predictor, data_type


def get_sorted_unique_quantiles(target_quantiles):
    """
    Return the sorted unique quantile levels referenced by interval pairs.

    Args:
        target_quantiles (list[list[float]]): e.g. [[0.95, 0.05], [0.9, 0.1]]

    Returns:
        list[float]: e.g. [0.05, 0.1, 0.9, 0.95]
    """
    return sorted({float(x) for pair in target_quantiles for x in pair})


def get_interval_quantile_indices(target_quantiles):
    """
    Map each configured interval pair to indices in the sorted unique quantile list.

    Args:
        target_quantiles (list[list[float]]): e.g. [[0.95, 0.05], [0.9, 0.1]]

    Returns:
        tuple[list[float], dict[tuple[float, float], tuple[int, int]]]:
            sorted_quantiles and a map from interval pair to (upper_idx, lower_idx)
    """
    sorted_quantiles = get_sorted_unique_quantiles(target_quantiles)
    index_map = {q: idx for idx, q in enumerate(sorted_quantiles)}
    pair_to_indices = {
        tuple(pair): (index_map[max(pair)], index_map[min(pair)])
        for pair in target_quantiles
    }
    return sorted_quantiles, pair_to_indices


def to_strided_residual(residual_sequence, window_len, pred_horizon=1):
    """
    residual_sequence: 1D array of length T

    Args:
        pred_horizon : number of future residual steps you want to predict after the input window, int > 0.

    Returns:
        inputs  : (N, window_len)
        targets : (N, pred_horizon)
    where N = T - window_len - pred_horizon + 1
    """

    if isinstance(residual_sequence, list):
        residual_sequence = np.array(residual_sequence)

    if residual_sequence.ndim == 2:
        residual_sequence = residual_sequence.flatten()

    seq = np.asarray(residual_sequence)
    T = len(seq)
    N = T - window_len - pred_horizon + 1
    if N <= 0:
        raise ValueError("Sequence too short for given window_len and pred_horizon.")

    inputs = sliding_window_view(seq, window_len)[:N] # (N, window_len)
    targets = np.stack([seq[i+window_len : i+window_len+pred_horizon] for i in range(N)], axis=0)  # (N, H)
    return inputs, targets


def to_strided_feature(feature_sequence, window_len, pred_horizon=1, return_target=False):
    """
    feature_sequence: array of shape (T, d)
    Returns:
      X: (N, window_len, d)
    aligned with N from residuals: N = T - window_len - pred_horizon + 1
    """
    X = np.asarray(feature_sequence)
    if X.ndim == 1:
        # reshape (T,) to (T,1)
        X = X.reshape(-1,1)
    T, d = X.shape
    N = T - window_len - pred_horizon + 1
    
    if N <= 0:
        raise ValueError("Sequence too short for given window_len and pred_horizon.")

    # windows: (T - window_len + 1, window_len, d)
    sw = sliding_window_view(X, window_len, axis=0)
    sw = sw[:N]                          # keep only the first N to align with targets
    # transpose to (N, d, window_len)
    sw = np.transpose(sw, (0, 2, 1)) # (N, window_len, d)
    targets = np.stack([X[i+window_len : i+window_len+pred_horizon] for i in range(N)], axis=0)  # (N, 1, d)

    if return_target:
        return sw, targets
    else:
        return sw


def build_hopcpt_context_features(x, y, predictions, y_lags):
    """
    Build HopCPT context rows Z_t = [Y_{t-k}, ..., Y_{t-1}, X_t, yhat_t].

    Returns rows for t = y_lags, ..., T-1, so arrays that should align with
    these contexts must be shifted by y_lags before striding.
    """
    if y_lags <= 0:
        raise ValueError("y_lags must be a positive integer.")

    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    predictions = np.asarray(predictions)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    if not (x.shape[0] == y.shape[0] == predictions.shape[0]):
        raise ValueError("x, y, and predictions must have the same time length.")
    if x.shape[0] <= y_lags:
        raise ValueError("Sequence too short for the requested y_lags.")

    y_lag_features = [y[y_lags - lag: -lag] for lag in range(y_lags, 0, -1)]
    return np.concatenate([*y_lag_features, x[y_lags:], predictions[y_lags:]], axis=-1)


def chronological_split(arrays, train_ratio=0.7, valid_ratio=0.15):
    """
    Chronological split along `axis` 0: train | valid | test (leftover).

    Parameters
    ----------
    arrays : list or tuple of array-like
        All inputs must have the same length along `axis`.
    train_ratio : float
    valid_ratio : float
    axis : int
        Axis along which to split (time dimension).

    Returns
    -------
    train, valid, test : tuples
        Each is a tuple of arrays, in the same order as input.
    """
    if not isinstance(arrays, (list, tuple)) or len(arrays) == 0:
        raise ValueError("`arrays` must be a non-empty list or tuple.")

    # Check lengths
    arrays = [np.asarray(a) for a in arrays]
    N = arrays[0].shape[0]
    for i, a in enumerate(arrays[1:], start=1):
        if a.shape[0] != N:
            raise ValueError(
                f"All arrays must have the same length along axis={0}. "
                f"arrays[0]={N}, arrays[{i}]={a.shape[0]}"
            )

    # Compute split sizes
    n_tr = int(np.floor(N * train_ratio))
    n_va = int(np.floor(N * valid_ratio))
    n_te = N - n_tr - n_va
    if n_tr <= 0 or n_va <= 0 or n_te <= 0:
        raise ValueError(f"Empty split: N={N}, train={n_tr}, valid={n_va}, test={n_te}")

    def take(a, sl):
        idx = [slice(None)] * a.ndim
        idx[0] = sl
        return a[tuple(idx)]

    sl_tr = slice(0, n_tr)
    sl_va = slice(n_tr, n_tr + n_va)
    sl_te = slice(n_tr + n_va, None)

    train = tuple(take(a, sl_tr) for a in arrays)
    valid = tuple(take(a, sl_va) for a in arrays)
    test  = tuple(take(a, sl_te) for a in arrays)

    return train, valid, test


def chronological_split_fixed_test(arrays, valid_size, test_size):

    # Check lengths
    arrays = [np.asarray(a) for a in arrays]
    N = arrays[0].shape[0]
    for i, a in enumerate(arrays[1:], start=1):
        if a.shape[0] != N:
            raise ValueError(
                f"All arrays must have the same length along axis={0}. "
                f"arrays[0]={N}, arrays[{i}]={a.shape[0]}"
            )

    # Compute split sizes
    n_va = valid_size
    n_te = test_size
    n_tr = N - valid_size - test_size # valid and test size not affected by window size
    if n_tr <= 0 or n_va <= 0 or n_te <= 0:
        raise ValueError(f"Empty split: N={N}, train={n_tr}, valid={n_va}, test={n_te}")

    def take(a, sl):
        idx = [slice(None)] * a.ndim
        idx[0] = sl
        return a[tuple(idx)]

    sl_tr = slice(0, n_tr)
    sl_va = slice(n_tr, n_tr + n_va)
    sl_te = slice(n_tr + n_va, None)

    train = tuple(take(a, sl_tr) for a in arrays)
    valid = tuple(take(a, sl_va) for a in arrays)
    test  = tuple(take(a, sl_te) for a in arrays)

    return train, valid, test


def chronological_split_fixed_calibration_test(arrays, valid_size, calibration_size, test_size):
    """Split aligned arrays as train | validation | calibration | test.

    Validation, calibration, and test have fixed target counts so that a
    history window only reduces the number of training examples.
    """
    if not isinstance(arrays, (list, tuple)) or len(arrays) == 0:
        raise ValueError("`arrays` must be a non-empty list or tuple.")

    arrays = [np.asarray(a) for a in arrays]
    N = arrays[0].shape[0]
    for i, a in enumerate(arrays[1:], start=1):
        if a.shape[0] != N:
            raise ValueError(
                "All arrays must have the same length along axis=0. "
                f"arrays[0]={N}, arrays[{i}]={a.shape[0]}"
            )

    n_va = int(valid_size)
    n_ca = int(calibration_size)
    n_te = int(test_size)
    n_tr = N - n_va - n_ca - n_te
    if n_tr <= 0 or n_va <= 0 or n_ca <= 0 or n_te <= 0:
        raise ValueError(
            "Empty split: "
            f"N={N}, train={n_tr}, valid={n_va}, "
            f"calibration={n_ca}, test={n_te}"
        )

    def take(a, sl):
        idx = [slice(None)] * a.ndim
        idx[0] = sl
        return a[tuple(idx)]

    valid_start = n_tr
    calibration_start = valid_start + n_va
    test_start = calibration_start + n_ca
    split_slices = (
        slice(0, valid_start),
        slice(valid_start, calibration_start),
        slice(calibration_start, test_start),
        slice(test_start, None),
    )
    return tuple(tuple(take(a, sl) for a in arrays) for sl in split_slices)


def normalize_array(X):
    """
    standardize array
    """
    mu = X.mean(axis=0)      
    std = X.std(axis=0)      
    
    X_normalized = (X - mu) / (std + 1e-8)

    return X_normalized, (mu, std)


def normalize_array_with_params(X, mu, std):

    X_normalized = (X - mu) / (std + 1e-8)

    return X_normalized


def compute_mean_std(X):
    """
    compute mean and std of the given array
    """

    return X.mean(axis=0), X.std(axis=0)


def denormalize_array(X, mu, std):
    """
    standardize array
    """
    X_denormalized = X * (std + 1e-8) + mu
    
    return X_denormalized


def dot_product(q, k):
    """
    :param q: query vector, (batch_size, dim_model)
    :param k: key vector, (calibration_size, dim_model)
    """

    if q.shape[-1] != k.shape[-1]:
        raise ValueError("query and key dimension must match")

    qk = q @ k.T

    return qk  # (batch_size, calibration_size)


def cos_similarity(q, k):
    """
    :param q: query vector, (batch_size, dim_model)
    :param k: key vector, (calibration_size, dim_model)
    """

    if q.shape[-1] != k.shape[-1]:
        raise ValueError("query and key dimension must match")
    
    qn = F.normalize(q, p=2, dim=-1, eps=1e-8)   # (batch_size, dim_model)
    kn = F.normalize(k, p=2, dim=-1, eps=1e-8)   # (calibration_size, dim_model)
    cos = qn @ kn.T

    return cos # (batch_size, calibration_size)


def generate_strided_feature(strided_x, strided_residual, strided_y, features_used):
    """
    strided_x : (batch_size, window, dim)
    strided_residual : (batch_size, window) or (batch_size, window, 1)
    strided_y : (batch_size, window) or (batch_size, window, 1)
    """
    if strided_residual.ndim == 2:
        strided_residual.unsqueeze_(-1)
    
    if strided_y.ndim == 2:
        strided_y.unsqueeze_(-1)

    if features_used == "xry":
        strided_feature = torch.cat([strided_x, strided_residual, strided_y], dim=-1)

    elif features_used == "xr":
        strided_feature = torch.cat([strided_x, strided_residual], dim=-1)

    elif features_used == "r":
        strided_feature = strided_residual

    return strided_feature


def generate_feature_hopcpt_training(train_context):
    """
    :param train_context: HopCPT context rows, (memory_len, dim)
    """
    if isinstance(train_context, np.ndarray):
        train_context = torch.from_numpy(train_context).to(torch.float32)

    if train_context.ndim == 2:
        train_context = train_context.unsqueeze(0)

    return train_context # (1, memory_len, dim)


def generate_feature_hopcpt_test(strided_context, target_context):
    """
    :param strided_context: memory contexts, (batch_size, memory_len, dim) or (memory_len, dim)
    :param target_context: query context, (batch_size, 1, dim) or (1, dim)
    """

    if strided_context.ndim == 2:
        strided_context = strided_context.unsqueeze(0)

    if target_context.ndim == 2:
        target_context = target_context.unsqueeze(0)

    return strided_context, target_context


def estimate_quantile_values(weights: torch.Tensor, 
                             strided_residual: torch.Tensor, 
                             target_quantiles: list, 
                             sampling_num: int):
    """
    estimate quantile values based on the weighted cdf approximation by MC sampling

    :param weights: (batch_size, memory_length)
    :param strided_residual: (batch_size, memory_length) or (memory_length,)
    :param target_quantiles: list of array
    :param sampling_num: int
    """
    weights = weights.detach().cpu()

    if not isinstance(target_quantiles, torch.Tensor):
        target_quantiles = torch.tensor(target_quantiles).to(torch.float32)

    if weights.dim() == 4:
        weights.squeeze_(1).squeeze_(1) # (batch_size, 1, 1, memory_length) to (batch_size, memory_length)

    if weights.dim() == 3:
        weights.squeeze_(-1) # (batch_size, memory_length, 1) to (batch_size, memory_length)

    if strided_residual.dim() == 1:
        strided_residual.unsqueeze_(0) # (memory_length,) to (batch_size, memory_length)
    
    if strided_residual.dim() == 3:
        strided_residual.squeeze_(-1) # (batch_size, memory_length, 1) to (batch_size, memory_length)
    
    sampled_idx = torch.multinomial(weights, num_samples=sampling_num, replacement=True) # (batch_size, sampling_num)
    empirical = torch.gather(strided_residual, dim=1, index=sampled_idx) 

    return torch.quantile(empirical, target_quantiles, dim=1) # (len(target_quantiles), query_size)


def estimate_hopcpt_residual_interval(association_matrix,
                                      strided_residual,
                                      confidence_pair,
                                      sampling_num,
                                      absolute_residual=False):
    if absolute_residual:
        target_coverage = max(confidence_pair) - min(confidence_pair)
        estimated_width = estimate_quantile_values(association_matrix,
                                                   strided_residual,
                                                   target_coverage,
                                                   sampling_num)
        return -estimated_width, estimated_width

    estimated_quantile_values = estimate_quantile_values(
        association_matrix,
        strided_residual,
        [min(confidence_pair), max(confidence_pair)],
        sampling_num)
    return estimated_quantile_values[0], estimated_quantile_values[1]
