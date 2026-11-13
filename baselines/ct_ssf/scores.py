"""Frozen-head latent inversion, following upstream FeatRegressorNc.inv_g."""

import numpy as np
import torch


def latent_scores(head, features, output_indices, targets, location, scale,
                  steps=200, learning_rate=0.01, optimizer="adam", batch_size=256,
                  return_details=False, semantic_end=None):
    """Return approximate L-inf surrogate scores and final model-scale errors.

    Every row is optimized independently. Summing row losses makes SGD invariant
    to the execution batch size. The pretrained head is never optimized. This
    recovers the executable latent-inversion design in upstream commit 85843cbc;
    it does NOT implement the paper's Eq. (7) adaptive gradient weights.

    With return_details, also return displacement vectors u-z, model-scale
    targets, and reconstructed responses. Retain vectors only up to semantic_end
    (default: all rows); attention needs only the train/validation prefix.
    """
    if steps < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Inversion steps, batch size, and learning rate must be positive.")
    if optimizer not in {"adam", "sgd"}:
        raise ValueError("inverse_optimizer must be 'adam' or 'sgd'.")
    semantic_end = len(targets) if semantic_end is None else semantic_end
    if not 0 <= semantic_end <= len(targets):
        raise ValueError("semantic_end must lie within the observation sequence.")
    head.eval().requires_grad_(False)
    device = next(head.parameters()).device
    scores, errors = [], []
    displacements, model_targets, reconstructed = [], [], []
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
                z.grad, = torch.autograd.grad(loss, z)
                opt.step()
        with torch.no_grad():
            delta, fitted = z - z0, head.select(z, idx)
            scores.append(delta.abs().amax(1).cpu().numpy())
            errors.append((fitted - y).abs().cpu().numpy())
            if return_details:
                if start < semantic_end:
                    displacements.append(delta[:semantic_end - start].cpu().numpy())
                model_targets.append(y.cpu().numpy())
                reconstructed.append(head.to_response(fitted.double(), loc.double(), scl.double()).cpu().numpy())
    result, residual = np.concatenate(scores), np.concatenate(errors)
    if not np.isfinite(result).all() or not np.isfinite(residual).all():
        raise ValueError("Latent inversion produced nonfinite scores/errors.")
    if not return_details:
        return result, residual
    return result, residual, {
        "semantic_errors": np.concatenate(displacements) if displacements else np.empty((0, features.shape[1]), dtype=np.float32),
        "model_targets": np.concatenate(model_targets),
        "reconstructed_targets": np.concatenate(reconstructed),
    }


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
    # Common rescaling preserves the distribution and avoids mass overflow.
    scale = max(float(weights.max()), float(test_weight))
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[order] / scale)
    threshold = (1 - alpha) * (cumulative[-1] + test_weight / scale)
    rank = np.searchsorted(cumulative, threshold, side="left")
    return float(scores[order[rank]]) if rank < len(scores) else float("inf")
