"""Chronos IBP/CROWN bounds and exact bounds for the linear LSTM output head.

The perturbation is an L-infinity ball around the *head input*, not the raw
context. Calculations use float64 and a small outward numerical margin. These
are analytic relaxations, not an interval-arithmetic floating-point proof.
"""

import copy

import torch
from torch.nn import functional as F


class HeadBounder:
    def __init__(self, head, method="crown"):
        linear = getattr(head, "head_type", None) == "linear"
        if method not in ({"ibp", "crown", "linear_exact"} if linear else {"ibp", "crown"}):
            raise ValueError("bound_method must be 'ibp' or 'crown', or 'linear_exact' for an LSTM head.")
        self.head = copy.deepcopy(head).double().eval().requires_grad_(False)
        self.method = "linear_exact" if linear else method

    @torch.no_grad()
    def __call__(self, features, indices, radius, location, scale):
        head = self.head
        z = features.to(device=next(head.parameters()).device, dtype=torch.float64)
        indices = torch.as_tensor(indices, device=z.device, dtype=torch.long).reshape(-1)
        eps = torch.as_tensor(radius, device=z.device, dtype=z.dtype).expand(len(z))
        loc = torch.as_tensor(location, device=z.device, dtype=z.dtype).expand(len(z))
        scl = torch.as_tensor(scale, device=z.device, dtype=z.dtype).expand(len(z))
        if torch.isnan(eps).any() or (eps < 0).any():
            raise ValueError("Latent radius must be nonnegative (positive infinity is allowed).")
        if not torch.isfinite(z).all() or not torch.isfinite(loc).all() or not torch.isfinite(scl).all() or (scl <= 0).any():
            raise ValueError("Features and normalization must be finite; scale must be positive.")
        lower = torch.full_like(eps, -torch.inf)
        upper = torch.full_like(eps, torch.inf)
        finite = torch.isfinite(eps)
        if not finite.any():
            return lower, upper
        x, e, idx = z[finite], eps[finite, None], indices[finite]
        if self.method == "linear_exact":
            weights = head.linear.weight[idx]
            center = (weights * x).sum(1) + head.linear.bias[idx]
            spread = e.squeeze(1) * weights.abs().sum(1)
            lb, ub = center - spread, center + spread
            margin = 1e-9 * (1 + torch.maximum(lb.abs(), ub.abs()))
            lower[finite] = head.to_response(lb - margin, loc[finite], scl[finite])
            upper[finite] = head.to_response(ub + margin, loc[finite], scl[finite])
            overflow = ~torch.isfinite(lower) | ~torch.isfinite(upper)
            lower[overflow], upper[overflow] = -torch.inf, torch.inf
            return lower, upper
        w, b = head.hidden_layer.weight, head.hidden_layer.bias
        a = head.output_layer.weight[idx]
        r = head.residual_layer.weight[idx]
        c = head.output_layer.bias[idx] + head.residual_layer.bias[idx]
        center = F.linear(x, w, b)
        spread = e * w.abs().sum(1)
        lo, hi = center - spread, center + spread
        pos, neg = a.clamp(min=0), a.clamp(max=0)
        skip = (r * x).sum(1) + c
        skip_spread = e.squeeze(1) * r.abs().sum(1)
        ibp_lo = (pos * lo.relu() + neg * hi.relu()).sum(1) + skip - skip_spread
        ibp_hi = (pos * hi.relu() + neg * lo.relu()).sum(1) + skip + skip_spread
        lb, ub = ibp_lo, ibp_hi
        if self.method == "crown":
            # For a crossing ReLU: secant upper bound and either x or 0 below.
            crossing = (lo < 0) & (hi > 0)
            upper_slope = torch.where(crossing, hi / (hi - lo).clamp_min(1e-300), (lo >= 0).double())
            upper_bias = torch.where(crossing, -lo * upper_slope, torch.zeros_like(lo))
            lower_slope = torch.where(crossing, (hi >= -lo).double(), (lo >= 0).double())
            for is_lower in (True, False):
                use_lower = (a >= 0) if is_lower else (a < 0)
                slope = torch.where(use_lower, lower_slope, upper_slope)
                intercept = torch.where(use_lower, torch.zeros_like(lo), upper_bias)
                affine = (a * slope) @ w + r
                constant = (a * (slope * b + intercept)).sum(1) + c
                value = (affine * x).sum(1) + constant
                variation = e.squeeze(1) * affine.abs().sum(1)
                if is_lower:
                    lb = torch.maximum(ibp_lo, value - variation)
                else:
                    ub = torch.minimum(ibp_hi, value + variation)
        # Intersecting two valid relaxations remains a valid bound.
        margin = 1e-9 * (1 + torch.maximum(lb.abs(), ub.abs()))
        lb, ub = lb - margin, ub + margin
        lower[finite] = head.to_response(lb, loc[finite], scl[finite])
        upper[finite] = head.to_response(ub, loc[finite], scl[finite])
        # Extremely large finite sinh inputs may overflow to one-sided infinity;
        # return a conservative full interval instead of producing NaN widths.
        overflow = ~torch.isfinite(lower) | ~torch.isfinite(upper)
        lower[overflow], upper[overflow] = -torch.inf, torch.inf
        return lower, upper
