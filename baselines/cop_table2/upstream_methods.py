"""COP paper Table 2 methods, extracted from the authors' experiment code.

Source: https://github.com/creator-xi/Conformal-Optimistic-Prediction
File: core/methods.py
Commit: a41c0dcadc10979d8521ab9b5b8f3d460807d883
Copyright (c) 2025 Junxi Wu; MIT license in UPSTREAM_LICENSE.

These are the versions used by COP's comparison, including its LQT heuristic
initialization from the first T_burnin scores and ACI warmup/infinite quantiles.
They are separate from this repository's standalone baseline implementations.

Only integration changes: remove tqdm progress wrappers, use np.inf in place of
the removed NumPy alias np.infty, and remove COP's unused np.random.seed(42).
No random samples are drawn by the active source code. Algorithm arithmetic,
indexing, arguments, and returned arrays are preserved.
"""

import numpy as np


def aci(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in range(T_test):
        t_pred = t - ahead + 1
        if t_pred > T_burnin:
            # Setup: current gradient
            if alphat <= 1/(t_pred+1):
                qs[t] = np.inf
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - lr*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = np.inf
    results = { "method": "ACI", "q" : qs, "alpha" : alphas}

    return results


def LQT(
    scores,
    alpha,
    lr,
    ahead,
    proportional_lr=True,
    p_order=5,
    bias=1.0,
    *args,
    **kwargs
):
    T_burnin = kwargs.get('T_burnin', 0)
    T_test = scores.shape[0]
    
    # Initialize quantiles and parameters
    qs = np.zeros(T_test)
    theta = np.zeros((T_test, p_order + 1))
    phi = np.zeros((T_test, p_order + 1))
    
    # Initialize theta with heuristic values
    if p_order > 0:
        initial_quantile = np.quantile(scores[:min(T_burnin, len(scores))], 1-alpha) if T_burnin > 0 else np.median(scores[:min(10, len(scores))])
        theta[p_order] = np.ones(p_order + 1) * (initial_quantile / p_order)
    else:
        theta[0] = np.ones(p_order + 1) * np.quantile(scores[:min(T_burnin, len(scores))], 1-alpha) if T_burnin > 0 else np.median(scores[:min(10, len(scores))])
    
    # Main loop
    for t in range(p_order, T_test):
        t_pred = t - ahead + 1
        
        if t_pred < p_order:
            qs[t] = 0
            
        # Compute learning rate (proportional if specified)
        # t_lr = t_pred
        # t_lr_min = max(t_lr - T_burnin, 0) if T_burnin > 0 else 0
        # if proportional_lr and t_lr > p_order:
        #     score_range = scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()
        #     lr_t = lr * score_range if score_range > 0 else lr
        # else:
        #     lr_t = lr
        lr_t = lr

        # Construct feature vector phi
        if t_pred >= p_order:
            phi[t_pred] = np.concatenate((scores[t_pred-p_order:t_pred], [bias]))
            
            # Compute quantile prediction
            qs[t] = theta[t_pred].T @ phi[t_pred]
            
            # Compute coverage error
            err_t = (scores[t] > qs[t]).astype(int)
            
            # Update parameters for next time step
            if t < T_test - 1 and t_pred + 1 < T_test:
                gradient = (err_t - alpha) * phi[t_pred]
                theta[t_pred + 1] = theta[t_pred] + lr_t * gradient
            
            # Propagate theta for consistency
            if t_pred + 1 < T_test:
                for future_t in range(t_pred + 1, min(T_test, t_pred + ahead + 1)):
                    if future_t < T_test:
                        theta[future_t] = theta[t_pred + 1].copy()
    
    results = {"method": "LQT", "q": qs}
    return results


def COP(
    scores,
    alpha,
    lr,
    T_burnin,
    ahead,
    proportional_lr=True,
    *args,
    **kwargs
):
    # Initialization
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    scale = 0.5

    # Run the main loop
    for t in range(T_test):
        t_lr = t
        t_lr_min = max(t_lr - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        #lr_t = lr # for extreme drift
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue # We can't make any predictions yet if our prediction time has not yet arrived
        # First, observe y_t and calculate coverage
        covereds[t] = qs[t] >= scores[t]
        # Next, calculate the quantile update and saturation function
        grad = alpha if covereds[t_pred] else -(1-alpha)

        # caluate the distribution-informed gradient
        if t_pred < T_burnin:
            grad_i = 0
        else:
            window_s = scores[t_pred-T_burnin:t_pred]
            grad_i = np.mean(window_s <= qts[t] - lr_t * grad) - (1-alpha)  # ECDF
            # grad_i = 0 * grad_i + 1 * np.random.uniform(0, 1) - (1-alpha)  # inaccuarte ECDF, add noise
            # grad_i = sliding_window_kde(qts[t] - lr_t * grad, window_s, bandwidth_method='silverman') - (1-alpha)  # Gaussian kernel
            # grad_i = sliding_window_kde(qts[t] - lr_t * grad, window_s, bandwidth_method='scott') - (1-alpha)  # Gaussian kernel
        integrator = -scale * grad_i

        # Update the next quantile
        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad
            integrators[t+1] = lr_t * integrator
            qs[t+1] = qts[t+1] + integrators[t+1]
    results = {"method": "conditional_quantile_length", "q" : qs}

    return results
