"""Causal relevance learning used to repair the missing upstream attention path.

This is an explicit DSCP adaptation, not recovered author code. A learned
query/key embedding predicts the next log latent score from earlier scores.
Training and early stopping finish before the calibration fold starts.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ScoreAttention(nn.Module):
    def __init__(self, input_dim, embedding_dim=32, temperature=0.5):
        super().__init__()
        if embedding_dim < 1 or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Attention embedding dimension and temperature must be positive.")
        self.projection = nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("std", torch.ones(input_dim))
        self.temperature = float(temperature)

    def forward(self, x):
        return F.normalize(self.projection((x - self.mean) / self.std), dim=-1, eps=1e-8)

    def weights(self, query, memory):
        # Shift by the query self-logit; test weight is exactly one. The same
        # shift is applied to all weights, so it cancels in the quantile.
        self_logit = query.square().sum(-1, keepdim=True)
        return torch.exp(((query * memory).sum(-1) - self_logit) / self.temperature)


def fit_attention(inputs, scores, train_end, valid_end, options, device="cpu"):
    """Fit using causal windows; select epochs on the following validation fold."""
    allowed = {"embedding_dim", "temperature", "memory_size", "epochs", "patience", "batch_size", "learning_rate"}
    if set(options) - allowed:
        raise ValueError(f"Unknown attention settings: {sorted(set(options) - allowed)}")
    memory_size = int(options.get("memory_size", 128))
    epochs, patience = int(options.get("epochs", 30)), int(options.get("patience", 5))
    batch_size = int(options.get("batch_size", 64))
    lr = float(options.get("learning_rate", 0.001))
    if min(memory_size, epochs, patience, batch_size) < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Attention training settings must be positive.")
    x = torch.as_tensor(inputs[:valid_end], dtype=torch.float32, device=device)
    values = torch.log1p(torch.as_tensor(scores[:valid_end], dtype=torch.float32, device=device))
    # Scale the objective using training data only, so tiny latent errors do not
    # produce vanishing learning signals. This does not modify conformal scores.
    values = (values - values[:train_end].mean()) / values[:train_end].std(unbiased=False).clamp_min(1e-6)
    model = ScoreAttention(x.shape[1], options.get("embedding_dim", 32), options.get("temperature", 0.5)).to(device)
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
        weights = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
        predicted = (weights * values[memory]).sum(1)
        return (predicted - values[rows]).square().mean()

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
    return model, {"best_epoch": best_epoch, "validation_loss": best_loss, "validation_history": history}
