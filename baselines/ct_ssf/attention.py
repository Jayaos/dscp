"""Semantic-error attention aligned with the repaired local CT-SSF clone.

The shared embedding learns KL similarity of inversion displacement vectors.
This is a reconstruction, not recovered author code. Chronos retains causal
memory and validation early stopping, both before the calibration fold.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


OBJECTIVE = "semantic_error_similarity_kl"
ATTENTION_FORMAT_VERSION = 2


class SemanticErrorAttention(nn.Module):
    def __init__(self, input_dim, embedding_dim=32, temperature=0.5):
        super().__init__()
        if input_dim < 1 or embedding_dim < 1 or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Attention embedding dimension and temperature must be positive.")
        self.projection = nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("std", torch.ones(input_dim))
        self.temperature = float(temperature)

    def forward(self, x):
        return F.normalize(self.projection((x - self.mean) / self.std), dim=-1, eps=1e-8)

    def weights(self, query, memory):
        """Masses for one encoded query and its encoded calibration memory.

        Normalize calibration and test self-logits together in float64, as in
        CT-SSF/conformal/attention.py. No label or score enters this calculation.
        """
        query, memory = query.double(), memory.double()
        logits = (memory @ query) / self.temperature
        self_logit = query.square().sum().reshape(1) / self.temperature
        masses = torch.softmax(torch.cat((logits, self_logit)), dim=0)
        return masses[:-1], masses[-1]


def fit_attention(inputs, semantic_errors, train_end, valid_end, options, device="cpu"):
    """Fit error-similarity KL on causal windows; select epochs on validation.

    Errors are vectors u-z, not scalar L-inf scores or output residuals.
    Inputs may also include the output-position encoding, so their dimension
    need not equal the error dimension. Both normalizers use training only.
    """
    allowed = {"embedding_dim", "temperature", "memory_size", "epochs", "patience", "batch_size", "learning_rate"}
    if set(options) - allowed:
        raise ValueError(f"Unknown attention settings: {sorted(set(options) - allowed)}")
    memory_size = int(options.get("memory_size", 128))
    epochs, patience = int(options.get("epochs", 30)), int(options.get("patience", 5))
    batch_size = int(options.get("batch_size", 64))
    lr = float(options.get("learning_rate", 0.001))
    if min(memory_size, epochs, patience, batch_size) < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Attention training settings must be positive.")
    if not 2 <= train_end < valid_end <= min(len(inputs), len(semantic_errors)):
        raise ValueError("Attention requires nonempty training and validation prefixes.")
    x = torch.as_tensor(inputs[:valid_end], dtype=torch.float32, device=device).detach()
    errors = torch.as_tensor(semantic_errors[:valid_end], dtype=torch.float32, device=device).detach()
    if x.ndim != 2 or errors.ndim != 2 or min(x.shape[1], errors.shape[1]) < 1:
        raise ValueError("Attention requires aligned feature and semantic-error matrices.")
    if not torch.isfinite(x).all() or not torch.isfinite(errors).all():
        raise ValueError("Nonfinite attention features or semantic errors.")
    error_mean = errors[:train_end].mean(0)
    error_std = errors[:train_end].std(0, unbiased=False).clamp_min(1e-6)
    errors = (errors - error_mean) / error_std
    model = SemanticErrorAttention(x.shape[1], options.get("embedding_dim", 32), options.get("temperature", 0.5)).to(device)
    with torch.no_grad():
        model.mean.copy_(x[:train_end].mean(0))
        model.std.copy_(x[:train_end].std(0, unbiased=False).clamp_min(1e-6))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    offsets = torch.arange(1, memory_size + 1, device=device)

    def objective(rows):
        memory = rows[:, None] - offsets
        mask = memory >= 0
        memory = memory.clamp_min(0)
        query, keys = model(x[rows]), model(x[memory])
        logits = (query[:, None, :] * keys).sum(-1) / model.temperature
        with torch.no_grad():
            distances = (errors[rows, None, :] - errors[memory]).square().mean(-1)
            teacher = torch.softmax((-distances).masked_fill(~mask, -torch.inf), dim=1)
        log_weights = F.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
        # Teacher mass is exactly zero in padding; remove -inf there before KL
        # to avoid 0 * -inf, without admitting padded keys for large distances.
        log_weights = log_weights.masked_fill(~mask, 0)
        return F.kl_div(log_weights, teacher, reduction="batchmean")

    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(epochs):
        model.train()
        # Shuffling optimization rows does not change the strictly past-only
        # memory of each row, and never includes calibration/test observations.
        rows = torch.randperm(train_end - 1, device=device) + 1
        for batch in rows.split(batch_size):
            loss = objective(batch)
            if not torch.isfinite(loss):
                raise ValueError("Attention training diverged.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            total = 0.0
            for batch in torch.arange(train_end, valid_end, device=device).split(batch_size):
                total += float(objective(batch)) * len(batch)
            validation = total / (valid_end - train_end)
        history.append(validation)
        if validation < best_loss:
            best_loss, best_state, best_epoch, stale = validation, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise ValueError("Attention validation did not produce a finite loss.")
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    return model, {
        "format_version": ATTENTION_FORMAT_VERSION, "objective": OBJECTIVE,
        "reconstruction": True, "memory_policy": "strictly_past",
        "training_samples": train_end, "validation_samples": valid_end - train_end,
        "semantic_error_dim": errors.shape[1],
        "error_mean": error_mean.cpu().tolist(), "error_std": error_std.cpu().tolist(),
        "best_epoch": best_epoch, "validation_loss": best_loss, "validation_history": history,
    }
