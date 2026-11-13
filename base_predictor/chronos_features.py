"""Portable Chronos-2 median head and capture of its actual forecast features.

The exported head uses the ResidualBlock architecture in chronos-forecasting
v2.2.2 (Apache-2.0): hidden Linear/ReLU, output Linear, and a Linear skip.
No Chronos/Transformers installation is needed to consume this artifact.
"""

import hashlib
import json

import numpy as np
import torch
from torch import nn


FEATURE_FORMAT_VERSION = 1


class ChronosMedianHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_patch_size, use_arcsinh):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_patch_size)
        self.residual_layer = nn.Linear(input_dim, output_patch_size)
        self.output_patch_size = output_patch_size
        self.use_arcsinh = use_arcsinh

    def forward(self, features):
        return self.output_layer(torch.relu(self.hidden_layer(features))) + self.residual_layer(features)

    def select(self, features, output_indices):
        return self(features).gather(1, output_indices.long().reshape(-1, 1)).squeeze(1)

    def to_response(self, values, location, scale):
        if self.use_arcsinh:
            values = torch.sinh(values)
        return values * scale + location

    def to_model_scale(self, values, location, scale):
        values = (values - location) / scale
        return torch.asinh(values) if self.use_arcsinh else values


def head_fingerprint(artifact):
    digest = hashlib.sha256(json.dumps(artifact["architecture"], sort_keys=True).encode())
    for name, tensor in sorted(artifact["state_dict"].items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def export_chronos_head(model, provenance=None):
    source = model.output_patch_embedding
    if model.config.dense_act_fn != "relu" or source.use_layer_norm:
        raise ValueError("CT-SSF export supports the Chronos-2 ReLU head without output layer normalization.")
    if any(p.dtype != torch.float32 for p in source.parameters()):
        raise ValueError("CT-SSF export requires float32 Chronos weights.")
    patch_size = int(model.chronos_config.output_patch_size)
    quantiles = list(model.chronos_config.quantiles)
    median_index = quantiles.index(0.5)
    median_rows = slice(median_index * patch_size, (median_index + 1) * patch_size)
    architecture = {
        "input_dim": source.hidden_layer.in_features,
        "hidden_dim": source.hidden_layer.out_features,
        "output_patch_size": patch_size,
        "use_arcsinh": bool(model.instance_norm.use_arcsinh),
    }
    state = {}
    for name, tensor in source.state_dict().items():
        if name.startswith(("output_layer.", "residual_layer.")):
            tensor = tensor[median_rows]
        state[name] = tensor.detach().cpu().clone()
    artifact = {
        "format_version": FEATURE_FORMAT_VERSION,
        "architecture": architecture,
        "state_dict": state,
        "provenance": dict(provenance or {}),
        "feature_location": "input_to_output_patch_embedding",
        "quantile": 0.5,
    }
    artifact["head_sha256"] = head_fingerprint(artifact)
    return artifact


def load_chronos_head(artifact_or_path, device="cpu"):
    artifact = artifact_or_path if isinstance(artifact_or_path, dict) else torch.load(
        artifact_or_path, map_location="cpu", weights_only=True,
    )
    if artifact.get("format_version") != FEATURE_FORMAT_VERSION:
        raise ValueError("Unsupported Chronos CT-SSF head format.")
    if artifact.get("head_sha256") != head_fingerprint(artifact):
        raise ValueError("Chronos head fingerprint does not match its weights/configuration.")
    model = ChronosMedianHead(**artifact["architecture"])
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, artifact


class ChronosFeatureCapture:
    """Capture one predict_df call for one scalar target plus past-only covariates.

    Hooks observe the actual prediction pass; they do not rerun embed() with a
    different covariate setup. Target row zero follows Chronos-2's task layout.
    Autoregressive unrolling/multiple model passes are rejected, never guessed.
    """

    def __init__(self, pipeline, head_artifact):
        self.model = pipeline.model
        self.head, _ = load_chronos_head(head_artifact, device=next(self.model.parameters()).device)
        self.head_sha256 = head_artifact["head_sha256"]
        self.handles = []

    def __enter__(self):
        self.features = []
        self.loc_scales = []
        self.handles = [
            self.model.output_patch_embedding.register_forward_pre_hook(self._features_hook),
            self.model.instance_norm.register_forward_hook(self._normalization_hook),
        ]
        return self

    def _features_hook(self, module, args):
        self.features.append(args[0].detach().clone())

    def _normalization_hook(self, module, args, output):
        self.loc_scales.append(tuple(t.detach().clone() for t in output[1]))

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def export_block(self, predictions, origin, covariate_dim):
        if len(self.features) != 1 or not self.loc_scales:
            raise ValueError("CT-SSF export requires one direct Chronos model pass per forecast block.")
        all_features = self.features[0]
        if all_features.ndim != 3 or all_features.shape[0] != covariate_dim + 1:
            raise ValueError("Unexpected Chronos task layout; expected one target followed by its covariates.")
        features = all_features[0]
        loc, scale = (t[0].reshape(()) for t in self.loc_scales[0])
        if not torch.isfinite(loc) or not torch.isfinite(scale) or scale <= 0:
            raise ValueError("Chronos exported invalid normalization values.")
        for other_loc, other_scale in self.loc_scales[1:]:
            if not torch.equal(other_loc[0].reshape(()), loc) or not torch.equal(other_scale[0].reshape(()), scale):
                raise ValueError("Normalization changed within a forecast block.")
        count = len(predictions)
        if count > len(features) * self.head.output_patch_size:
            raise ValueError("Forecast horizon exceeds captured forecast patches.")
        with torch.no_grad():
            reconstructed = self.head.to_response(self.head(features).flatten()[:count], loc, scale)
        expected = torch.as_tensor(predictions, device=reconstructed.device, dtype=reconstructed.dtype)
        if not torch.isfinite(features).all() or not torch.allclose(reconstructed, expected, rtol=1e-4, atol=1e-5):
            error = (reconstructed - expected).abs().max().item()
            raise ValueError(f"CT-SSF features do not reconstruct the saved forecast (max error={error}).")
        return {
            "features": features.cpu().numpy(),
            "feature_indices": np.arange(count, dtype=np.int64) // self.head.output_patch_size,
            "output_indices": np.arange(count, dtype=np.int64) % self.head.output_patch_size,
            "location": np.full(count, loc.item(), dtype=np.float32),
            "scale": np.full(count, scale.item(), dtype=np.float32),
            "forecast_origin": np.full(count, origin, dtype=np.int64),
            "lead": np.arange(1, count + 1, dtype=np.int64),
            "raw_target_indices": np.arange(origin, origin + count, dtype=np.int64),
        }


def combine_feature_blocks(blocks, head_sha256):
    offset = 0
    adjusted = []
    for block in blocks:
        block = dict(block)
        block["feature_indices"] = block["feature_indices"] + offset
        offset += len(block["features"])
        adjusted.append(block)
    if not adjusted:
        raise ValueError("No forecast blocks available for CT-SSF export.")
    return {
        "format_version": FEATURE_FORMAT_VERSION,
        "head_sha256": head_sha256,
        **{name: np.concatenate([block[name] for block in adjusted]) for name in adjusted[0]},
    }
