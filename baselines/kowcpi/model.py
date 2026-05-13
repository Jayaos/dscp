"""
Kernel-based optimally weighted conformal residual intervals.

This module ports the interval-estimation part of KOWCPI_Codes/ into the
baseline layout used by this repository.  The runner supplies point forecasts
from the saved base-predictor output, while this module estimates residual
intervals with the KOWCPI weighted Nadaraya-Watson quantile step.
"""

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted


def _epanechnikov_from_distances(distances, bandwidth):
    z = distances / bandwidth
    kernel = 0.75 * (1.0 - z * z)
    kernel[z > 1.0] = 0.0
    kernel[kernel < 0.0] = 0.0
    return kernel


def _gaussian_from_squared_distances(squared_distances, bandwidth):
    return np.exp(-squared_distances / (2.0 * bandwidth * bandwidth))


class WeightedNadarayaWatson(BaseEstimator, RegressorMixin):
    """
    Weighted Nadaraya-Watson estimator used by KOWCPI.

    The bandwidth selection and empirical-likelihood style weight correction
    follow KOWCPI_Codes/weighted_nw.py, with small guardrails for short
    calibration windows.
    """

    def __init__(self, bandwidth=1.0, kernel="epanechnikov", bandwidth_range=None):
        self.bandwidth = bandwidth
        self.bandwidth_range = bandwidth_range
        self.kernel = kernel
        self.X_ = None
        self.y_ = None
        self.sample_weight_ = None
        self._bw_selected = False

    def _prepare_pairwise(self, x):
        self._distances = cdist(x, x, metric="euclidean")
        self._squared_distances = self._distances * self._distances

    def _aic_for_bandwidths_fast(self, y, bandwidths):
        n = len(y)
        y = y.reshape(-1, 1)
        aics = []

        for bandwidth in bandwidths:
            if self.kernel == "epanechnikov":
                kernel = _epanechnikov_from_distances(self._distances, bandwidth)
            elif self.kernel == "gaussian":
                kernel = _gaussian_from_squared_distances(self._squared_distances, bandwidth)
            else:
                raise ValueError("Unsupported kernel: {}".format(self.kernel))

            row_sum = kernel.sum(axis=1, keepdims=True)
            zero_mask = row_sum == 0.0
            row_sum[zero_mask] = 1.0
            weights = kernel / row_sum

            yhat = weights @ y
            resid = y - yhat
            rss = float((resid * resid).sum())
            trace_ss = float((weights * weights).sum())
            denominator = max(n - (trace_ss + 2.0), 1e-8)
            aics.append(np.log(max(rss, 1e-12)) + (n + trace_ss) / denominator)

        return np.asarray(aics)

    def kernel_function(self, x1, x2):
        bandwidth = 1.0 if self.bandwidth is None else float(self.bandwidth)
        norm = np.linalg.norm((x1 - x2) / bandwidth)
        if self.kernel == "gaussian":
            return np.exp(-(norm ** 2) / 2.0)
        if self.kernel == "epanechnikov":
            return 0.75 * (1.0 - norm ** 2) if abs(norm) <= 1.0 else 0.0
        raise ValueError("Unsupported kernel: {}".format(self.kernel))

    def L(self, lmbda, x_train, x_query):
        lmbda = float(np.asarray(lmbda).reshape(-1)[0])
        value = 0.0
        for row in x_train:
            kernel_value = self.kernel_function(x_query, row)
            barrier = 1.0 - lmbda * (row[0] - x_query[0]) * kernel_value
            if barrier <= 0.0:
                return np.inf
            value -= np.log(barrier)
        return value

    def fit(self, x, y, sample_weight=None):
        self.X_, self.y_ = check_X_y(x, y)
        if sample_weight is None:
            self.sample_weight_ = np.ones(len(self.y_), dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
            if len(sample_weight) != len(self.y_):
                raise ValueError("sample_weight must have the same length as y.")
            self.sample_weight_ = sample_weight

        if self.bandwidth is None and self.bandwidth_range is None:
            self.bandwidth = 1.0

        if (self.bandwidth is not None and self.bandwidth_range is None) or self._bw_selected:
            return self

        self._prepare_pairwise(self.X_)
        bandwidth_grid = np.asarray(self.bandwidth_range, dtype=float)
        bandwidth_grid = bandwidth_grid[bandwidth_grid > 0.0]
        if bandwidth_grid.size == 0:
            self.bandwidth = 1.0
            self.bandwidth_range = None
            self._bw_selected = True
            return self

        if bandwidth_grid.size > 12:
            idx = np.round(np.linspace(0, bandwidth_grid.size - 1, 8)).astype(int)
            coarse_grid = bandwidth_grid[idx]
            coarse_aic = self._aic_for_bandwidths_fast(self.y_, coarse_grid)
            best_bandwidth = float(coarse_grid[np.argmin(coarse_aic)])

            left = max(float(bandwidth_grid.min()), best_bandwidth * 0.7)
            right = min(float(bandwidth_grid.max()), best_bandwidth * 1.3)
            fine_grid = np.linspace(left, right, 5)
            fine_aic = self._aic_for_bandwidths_fast(self.y_, fine_grid)
            best_bandwidth = float(fine_grid[np.argmin(fine_aic)])
        else:
            aic = self._aic_for_bandwidths_fast(self.y_, bandwidth_grid)
            best_bandwidth = float(bandwidth_grid[np.argmin(aic)])

        self.bandwidth = best_bandwidth
        self.bandwidth_range = None
        self._bw_selected = True
        self._distances = None
        self._squared_distances = None
        return self

    def get_p_t_values(self, x_query):
        check_is_fitted(self, ["X_", "y_"])
        result = minimize(self.L, [0.1], args=(self.X_, x_query), method="L-BFGS-B")
        lambda_value = 0.0 if not result.success else float(result.x[0])

        p_values = np.zeros(len(self.X_), dtype=float)
        for idx, row in enumerate(self.X_):
            kernel_value = self.kernel_function(x_query, row)
            denominator = 1.0 - lambda_value * (row[0] - x_query[0]) * kernel_value
            if denominator <= 1e-12:
                denominator = 1e-12
            p_values[idx] = 1.0 / len(self.X_) / denominator
        return p_values

    def get_weights(self, x_query):
        p_values = self.get_p_t_values(x_query)
        kernel_values = np.asarray(
            [self.kernel_function(x_query, row) for row in self.X_],
            dtype=float,
        )
        weights = p_values * kernel_values * self.sample_weight_
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            return np.ones(len(weights), dtype=float) / len(weights)
        return weights / weight_sum


class WeightedNadarayaWatsonQuantile(WeightedNadarayaWatson):
    """Weighted Nadaraya-Watson quantile regressor."""

    def __init__(self, *args, quantiles=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantiles = np.asarray([0.5] if quantiles is None else quantiles, dtype=float)

    def predict(self, x_query):
        check_is_fitted(self, ["X_", "y_"])
        x_query = np.asarray(x_query, dtype=float).reshape(-1)
        weights = self.get_weights(x_query)

        sorted_idx = np.argsort(self.y_)
        sorted_y = self.y_[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cumulative = np.cumsum(sorted_weights)

        quantile_idx = np.searchsorted(cumulative, self.quantiles, side="left")
        quantile_idx = np.minimum(quantile_idx, len(sorted_y) - 1)
        return sorted_y[quantile_idx]


def glr_select_block_w(
    past_resid,
    w_min=15,
    w_max=None,
    m0=10,
    c=2.5,
    fallback=None,
):
    """GLR-based recent block-length selector from KOWCPI_Codes/adaptive_window.py."""
    x = np.asarray(past_resid, dtype=float).reshape(-1)
    n = x.size
    if w_max is None:
        w_max = max(2 * m0 + 1, n - 1)
    length = min(w_max, n - 1)
    if length < 2 * m0:
        return int(fallback if fallback is not None else max(w_min, m0))

    z = x[-length:]
    cumulative = np.cumsum(z)
    cumulative_sq = np.cumsum(z * z)

    split = np.arange(m0, length - m0 + 1)
    n1 = split
    n2 = length - split

    sum1 = cumulative[split - 1]
    sum2 = cumulative[-1] - sum1
    ss1 = cumulative_sq[split - 1]
    ss2 = cumulative_sq[-1] - ss1

    eps = 1e-12
    sse1 = np.maximum(ss1 - (sum1 ** 2) / n1, eps)
    sse2 = np.maximum(ss2 - (sum2 ** 2) / n2, eps)
    variance1 = sse1 / n1
    variance2 = sse2 / n2
    variance_all = (sse1 + sse2) / length

    statistic = length * np.log(variance_all) - (
        n1 * np.log(variance1) + n2 * np.log(variance2)
    )
    best_idx = int(np.argmax(statistic))
    threshold = c * np.log(length)

    if statistic[best_idx] > threshold:
        selected = int(n2[best_idx])
    else:
        selected = int(fallback if fallback is not None else max(w_min, m0))

    return int(np.clip(selected, max(w_min, m0), length - m0))


def empirical_beta_interval(past_resid, alpha, bins=5, fixed_beta=None):
    """
    Empirical fallback for the KOWCPI beta search.

    If fixed_beta is provided, only the corresponding
    [fixed_beta, 1 - alpha + fixed_beta] interval is evaluated.
    Otherwise, it searches beta in [0, alpha] and returns the shortest
    residual interval.
    """
    past_resid = np.asarray(past_resid, dtype=float).reshape(-1)
    if past_resid.size == 0:
        raise ValueError("At least one residual is required.")

    if fixed_beta is None:
        beta_grid = np.linspace(0.0, float(alpha), int(bins))
    else:
        fixed_beta = float(fixed_beta)
        if fixed_beta < 0.0 or fixed_beta > float(alpha):
            raise ValueError("fixed_beta must be in [0, alpha].")
        beta_grid = np.asarray([fixed_beta], dtype=float)
    widths = np.zeros(len(beta_grid), dtype=float)
    lows = np.zeros(len(beta_grid), dtype=float)
    highs = np.zeros(len(beta_grid), dtype=float)

    for idx, beta in enumerate(beta_grid):
        low_q = np.clip(beta, 0.0, 1.0)
        high_q = np.clip(1.0 - alpha + beta, 0.0, 1.0)
        lows[idx] = np.quantile(past_resid, low_q)
        highs[idx] = np.quantile(past_resid, high_q)
        widths[idx] = highs[idx] - lows[idx]

    best_idx = int(np.argmin(widths))
    return float(lows[best_idx]), float(highs[best_idx])


class KOWCPIResidualIntervalEstimator:
    """
    Estimate online residual intervals with KOWCPI's RNW quantile step.

    Parameters mirror the original KOWCPI implementation:
    - `block_size` in `predict_residual_intervals` corresponds to the original
      `past_window` used to build residual blocks.
    - `history_window` corresponds to the number of recent residuals used for
      each online update.  If omitted, it defaults to the calibration length.
    """

    def __init__(
        self,
        bandwidth=None,
        kernel="epanechnikov",
        bandwidth_range=None,
        bins=5,
        weigh_residuals=False,
        residual_weight_decay=0.995,
        max_training_blocks=None,
        min_training_blocks=2,
        fixed_beta=None,
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.bandwidth_range = None if bandwidth is not None else bandwidth_range
        self.bins = int(bins)
        self.weigh_residuals = bool(weigh_residuals)
        self.residual_weight_decay = float(residual_weight_decay)
        self.max_training_blocks = max_training_blocks
        self.min_training_blocks = int(min_training_blocks)
        self.fixed_beta = None if fixed_beta is None else float(fixed_beta)

    def _make_residual_design(self, past_resid, block_size):
        past_resid = np.asarray(past_resid, dtype=float).reshape(-1)
        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if past_resid.size <= block_size:
            return None, None, None

        residual_windows = sliding_window_view(past_resid, block_size)[:, ::-1].copy()
        train_x = residual_windows[:-1]
        train_y = past_resid[block_size:]
        last_x = residual_windows[-1]
        return train_x, train_y, last_x

    def _sample_weight(self, n):
        if not self.weigh_residuals:
            return None
        return self.residual_weight_decay ** np.arange(n, 0, -1, dtype=float)

    def _beta_grid(self, alpha):
        if self.fixed_beta is None:
            return np.linspace(0.0, float(alpha), self.bins)
        if self.fixed_beta < 0.0 or self.fixed_beta > float(alpha):
            raise ValueError("fixed_beta must be in [0, alpha].")
        return np.asarray([self.fixed_beta], dtype=float)

    def _fit_widths(self, past_resid, alpha, block_size):
        train_x, train_y, last_x = self._make_residual_design(past_resid, block_size)
        if train_x is None or len(train_y) < self.min_training_blocks:
            return empirical_beta_interval(past_resid, alpha, self.bins, self.fixed_beta)

        if self.max_training_blocks is not None:
            max_blocks = int(self.max_training_blocks)
            if max_blocks <= 0:
                raise ValueError("max_training_blocks must be positive when set.")
            train_x = train_x[-max_blocks:]
            train_y = train_y[-max_blocks:]
            if len(train_y) < self.min_training_blocks:
                return empirical_beta_interval(past_resid, alpha, self.bins, self.fixed_beta)

        beta_grid = self._beta_grid(alpha)
        quantiles = np.append(beta_grid, 1.0 - float(alpha) + beta_grid)

        quantile_model = WeightedNadarayaWatsonQuantile(
            bandwidth=self.bandwidth,
            kernel=self.kernel,
            bandwidth_range=self.bandwidth_range,
            quantiles=quantiles,
        )
        prediction = quantile_model.fit(
            train_x,
            train_y,
            sample_weight=self._sample_weight(len(train_y)),
        ).predict(last_x)

        num_low = len(prediction) // 2
        low_pred = prediction[:num_low]
        high_pred = prediction[num_low:]
        best_idx = int(np.argmin(high_pred - low_pred))

        if self.bandwidth is None:
            self.bandwidth = float(quantile_model.bandwidth)
            self.bandwidth_range = None

        return float(low_pred[best_idx]), float(high_pred[best_idx])

    def _select_block_size(self, past_resid, block_size, use_adaptive_window, glr_params):
        if not use_adaptive_window:
            selected = int(block_size)
        else:
            params = {} if glr_params is None else dict(glr_params)
            selected = glr_select_block_w(
                past_resid,
                w_min=int(params.get("w_min", 5)),
                w_max=params.get("w_max", None),
                m0=int(params.get("m0", max(5, int(block_size) + 1))),
                c=float(params.get("c", 2.5)),
                fallback=int(params.get("fallback", block_size)),
            )

        return max(1, min(int(selected), len(past_resid) - 1))

    def predict_residual_intervals(
        self,
        residuals,
        calibration_size,
        test_size,
        alpha,
        block_size,
        history_window=None,
        use_adaptive_window=False,
        glr_params=None,
        update_with_test=True,
    ):
        """
        Return lower and upper residual intervals for each test point.

        `residuals` may include the full calibration+test residual sequence.
        When `update_with_test=True`, interval i uses calibration residuals and
        the first i realized test residuals, matching the online construction in
        the original KOWCPI code.
        """
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        calibration_size = int(calibration_size)
        test_size = int(test_size)
        if calibration_size <= 0 or test_size <= 0:
            raise ValueError("calibration_size and test_size must be positive.")
        if residuals.size < calibration_size + test_size:
            raise ValueError("residuals must contain calibration and test residuals.")

        if history_window is None:
            history_window = calibration_size
        history_window = int(history_window)
        if history_window <= 0:
            raise ValueError("history_window must be positive.")

        lower = np.zeros(test_size, dtype=float)
        upper = np.zeros(test_size, dtype=float)
        progress_step = max(1, math.ceil(test_size / 20))

        for test_idx in range(test_size):
            history_end = calibration_size + test_idx if update_with_test else calibration_size
            history_start = max(0, history_end - history_window)
            past_resid = residuals[history_start:history_end]
            if past_resid.size == 0:
                raise ValueError("No residual history available for KOWCPI.")

            if past_resid.size <= 1:
                low, high = empirical_beta_interval(past_resid, alpha, self.bins, self.fixed_beta)
            else:
                selected_block = self._select_block_size(
                    past_resid,
                    block_size,
                    use_adaptive_window,
                    glr_params,
                )
                low, high = self._fit_widths(past_resid, alpha, selected_block)

            lower[test_idx] = low
            upper[test_idx] = high

            if test_idx == 0 or (test_idx + 1) % progress_step == 0:
                width = high - low
                print("KOWCPI residual width at test {} is {}".format(test_idx, width))

        return lower, upper
