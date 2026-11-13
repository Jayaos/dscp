"""Recover rolling LSTM features from the existing --save-model artifacts.

The trained predictor is frozen. Each z_t is the final hidden state of the
window ending at t-1, and the perturbed head is the scalar output_fc layer.
No extra base-predictor export or training is needed.
"""

import copy
import hashlib

import numpy as np
import torch
from torch import nn

from base_predictor.chronos_features import FEATURE_FORMAT_VERSION, head_fingerprint
from base_predictor.lstm_predictor import load_lstm_checkpoint


class LSTMLinearHead(nn.Module):
    head_type = "linear"
    output_patch_size = 1

    def __init__(self, output_fc):
        super().__init__()
        self.linear = copy.deepcopy(output_fc)

    def forward(self, features):
        return self.linear(features)

    def select(self, features, output_indices):
        return self(features).gather(1, output_indices.long().reshape(-1, 1)).squeeze(1)

    def to_response(self, values, location, scale):
        return values * scale + location

    def to_model_scale(self, values, location, scale):
        return (values - location) / scale


def load_lstm_backbone(path, device="cpu"):
    model, checkpoint = load_lstm_checkpoint(path, device=device)
    metadata = checkpoint["metadata"]
    required = {
        "evaluation_mode": "rolling_one_step", "prediction_step": 1,
        "uses_lagged_targets": True, "input_order": ["covariates", "lagged_target"],
        "normalization_scope": "pooled_inner_training_prefixes",
    }
    for name, expected in required.items():
        if metadata.get(name) != expected:
            raise ValueError(f"Unsupported LSTM checkpoint {name}: expected {expected!r}.")
    window = metadata.get("window_length")
    if not isinstance(window, int) or window < 1:
        raise ValueError("LSTM checkpoint must contain a positive window_length.")
    if model.output_fc.out_features != 1:
        raise ValueError("CT-SSF LSTM supports scalar rolling one-step predictions only.")
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all()
           for value in checkpoint["model_state_dict"].values()):
        raise ValueError("CT-SSF LSTM requires finite float32 predictor weights.")
    stats = checkpoint["normalization"]
    epsilon = stats.get("epsilon")
    if not isinstance(epsilon, (int, float)) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Invalid LSTM normalization epsilon.")
    for name in ("x_mean", "x_std", "y_mean", "y_std"):
        value = stats[name]
        expected_size = model.input_embedding.in_features - 1 if name.startswith("x_") else 1
        if value.numel() != expected_size or not torch.isfinite(value).all():
            raise ValueError(f"Invalid LSTM normalization {name}.")
        if name.endswith("std") and (value < 0).any():
            raise ValueError(f"LSTM normalization {name} must be nonnegative.")
    model.eval().requires_grad_(False)
    head = LSTMLinearHead(model.output_fc).eval().requires_grad_(False)
    # The full checkpoint fingerprint also binds encoder, normalization, and
    # forecast metadata, which the head-only fingerprint cannot distinguish.
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    artifact = {
        "format_version": FEATURE_FORMAT_VERSION, "base_predictor": "lstm",
        "architecture": {"input_dim": head.linear.in_features, "output_patch_size": 1, "head_type": "linear"},
        "state_dict": {name: value.detach().cpu().clone() for name, value in head.state_dict().items()},
        "checkpoint_sha256": digest.hexdigest(),
        "provenance": {
            "window_length": window, "prediction_length": 1, "uses_future_covariates": False,
            "feature_location": "encode(window)[:, -1, :]", "checkpoint_sha256": digest.hexdigest(),
            "base_training": dict(metadata),
        },
    }
    artifact["head_sha256"] = head_fingerprint(artifact)
    return model, head, checkpoint, artifact


def _scalar_sequence(values, name):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] != 1 or not np.isfinite(values).all():
        raise ValueError(f"LSTM {name} must be a finite scalar sequence.")
    return values


def attach_lstm_features(item, model, checkpoint, artifact, batch_size=256):
    """Return a new record with features; keep saved predictions unchanged.

    A strided view and execution batches avoid materializing all T*window*D
    inputs. The final window (which would predict beyond heldout_y) is excluded.
    prepare_sequence subsequently verifies head(features) against every saved
    forecast and rejects a mismatched model/data pair.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("LSTM feature batch_size must be a positive integer.")
    window = checkpoint["metadata"]["window_length"]
    stats = checkpoint["normalization"]
    epsilon = stats["epsilon"]
    sequences = {}
    for split in ("train", "heldout"):
        x = np.asarray(item[f"{split}_x"])
        y = _scalar_sequence(item[f"{split}_y"], f"{split}_y")
        if x.shape != (len(y), model.input_embedding.in_features - 1) or not np.isfinite(x).all():
            raise ValueError(f"Invalid LSTM {split} covariates or target alignment.")
        sequences[f"{split}_x"], sequences[f"{split}_y"] = x, y
    train_length, n = len(sequences["train_y"]), len(sequences["heldout_y"])
    if train_length < window or n == 0:
        raise ValueError("LSTM requires window_length training history and nonempty heldout observations.")
    normalized = []
    for kind in ("x", "y"):
        context = np.concatenate([sequences[f"train_{kind}"][-window:], sequences[f"heldout_{kind}"]])
        mean, std = (stats[f"{kind}_{name}"].numpy().reshape(-1) for name in ("mean", "std"))
        # Match the base predictor: normalize in the source array's precision,
        # then cast the completed windows to float32 for the network.
        normalized.append((context - mean) / (std + epsilon))
    inputs = np.concatenate(normalized, axis=1)
    if not np.isfinite(inputs).all():
        raise ValueError("LSTM normalized inputs must be finite.")
    windows = np.lib.stride_tricks.sliding_window_view(inputs, window, axis=0)
    windows = np.moveaxis(windows, -1, 1)[:n]
    device = next(model.parameters()).device
    features = []
    model.eval().requires_grad_(False)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = torch.from_numpy(np.array(windows[start:start + batch_size], dtype=np.float32, copy=True)).to(device)
            features.append(model.encode(batch)[:, -1, :].cpu().numpy())
    # Use the same dtype arithmetic for std+epsilon as the predictor.
    location = float(stats["y_mean"].reshape(-1)[0])
    scale = float((stats["y_std"] + epsilon).reshape(-1)[0])
    raw_indices = np.arange(train_length, train_length + n, dtype=np.int64)
    cache = {
        "format_version": FEATURE_FORMAT_VERSION, "head_sha256": artifact["head_sha256"],
        "checkpoint_sha256": artifact["checkpoint_sha256"],
        "features": np.concatenate(features), "feature_indices": np.arange(n, dtype=np.int64),
        "output_indices": np.zeros(n, dtype=np.int64), "forecast_origin": raw_indices.copy(),
        "lead": np.ones(n, dtype=np.int64), "raw_target_indices": raw_indices,
        "location": np.full(n, location), "scale": np.full(n, scale),
    }
    return {**item, "ctssf": cache}
