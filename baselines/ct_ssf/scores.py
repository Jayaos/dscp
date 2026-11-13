"""Frozen-head latent inversion, following upstream FeatRegressorNc.inv_g."""

import numpy as np
import torch


def latent_scores(head, features, output_indices, targets, location, scale,
                  steps=200, learning_rate=0.01, optimizer="adam", batch_size=256):
    """Return approximate L-inf surrogate scores and final model-scale errors.

    Every row is optimized independently. Summing row losses makes SGD invariant
    to the execution batch size. The pretrained head is never optimized. This
    recovers the executable latent-inversion design in upstream commit 85843cbc;
    it does NOT implement the paper's unspecified adaptive gradient weights.
    """
    if steps < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Inversion steps, batch size, and learning rate must be positive.")
    if optimizer not in {"adam", "sgd"}:
        raise ValueError("inverse_optimizer must be 'adam' or 'sgd'.")
    head.eval().requires_grad_(False)
    device = next(head.parameters()).device
    scores, errors = [], []
    for start in range(0, len(targets), batch_size):
        stop = start + batch_size
        z0 = torch.as_tensor(features[start:stop], dtype=torch.float32, device=device)
        idx = torch.as_tensor(output_indices[start:stop], dtype=torch.long, device=device)
        y = torch.as_tensor(targets[start:stop], dtype=torch.float32, device=device)
        loc = torch.as_tensor(location[start:stop], dtype=torch.float32, device=device)
        scl = torch.as_tensor(scale[start:stop], dtype=torch.float32, device=device)
        y = head.to_model_scale(y, loc, scl)
        z = z0.detach().clone().requires_grad_(True)
        opt = (torch.optim.Adam if optimizer == "adam" else torch.optim.SGD)([z], lr=learning_rate)
        with torch.enable_grad():
            for _ in range(steps):
                loss = (head.select(z, idx) - y).square().sum()
                if not torch.isfinite(loss):
                    raise ValueError("Latent inversion diverged; reduce inverse_lr or inspect the data.")
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            scores.append((z - z0).abs().amax(1).cpu().numpy())
            errors.append((head.select(z, idx) - y).abs().cpu().numpy())
    result, residual = np.concatenate(scores), np.concatenate(errors)
    if not np.isfinite(result).all() or not np.isfinite(residual).all():
        raise ValueError("Latent inversion produced nonfinite scores/errors.")
    return result, residual


def weighted_quantile(scores, weights, alpha, test_weight=1.0):
    """Equation (8), including the test point's mass at +infinity.

    Raw calibration and test weights share a common scale. In particular,
    calibration weights are NOT separately normalized to sum to one.
    """
    scores, weights = np.asarray(scores, dtype=float), np.asarray(weights, dtype=float)
    if scores.ndim != 1 or weights.shape != scores.shape or len(scores) == 0:
        raise ValueError("Scores and weights must be nonempty, matching vectors.")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Calibration scores must be finite and nonnegative.")
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isfinite(test_weight) or test_weight <= 0:
        raise ValueError("Weights must be finite and nonnegative; test weight must be positive.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1).")
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[order])
    threshold = (1 - alpha) * (weights.sum() + test_weight)
    rank = np.searchsorted(cumulative, threshold, side="left")
    return float(scores[order[rank]]) if rank < len(scores) else float("inf")
