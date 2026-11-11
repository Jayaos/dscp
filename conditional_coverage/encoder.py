"""Independent median forecasting encoder for shared evaluation groups.

For target index ``j``, the encoder sees feature rows ``[j - W, j)``.
Its one prediction head is trained against the observed response at ``j``;
base forecasts and residuals are never used by this module.
"""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dscp.models.rnn_predictor import RNNPredictor


MODEL_DEFAULTS = {
    "rnn_type": "lstm",
    "dim_model": 64,
    "num_layers": 1,
    "dropout": 0.0,
    "window_size": 100,
}
TRAINING_DEFAULTS = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 50,
    "early_stop": 5,
}


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return int(value)


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive.")
    return float(value)


def resolve_model_config(config=None):
    """Return validated, explicit encoder settings suitable for serialization."""
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("model must be a mapping.")
    config = dict(config or {})
    unknown = set(config) - MODEL_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown evaluation encoder settings: {sorted(unknown)}.")
    settings = {**MODEL_DEFAULTS, **config}
    if not isinstance(settings["rnn_type"], str):
        raise ValueError("model.rnn_type must be lstm, gru, or rnn.")
    settings["rnn_type"] = settings["rnn_type"].lower()
    if settings["rnn_type"] not in {"lstm", "gru", "rnn"}:
        raise ValueError("model.rnn_type must be lstm, gru, or rnn.")
    for name in ("dim_model", "num_layers", "window_size"):
        settings[name] = _positive_integer(settings[name], f"model.{name}")
    settings["dropout"] = _number(settings["dropout"], "model.dropout")
    if not 0 <= settings["dropout"] < 1:
        raise ValueError("model.dropout must be in [0, 1).")
    return settings


def resolve_training_config(config=None):
    """Return validated, explicit median training settings."""
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("training must be a mapping.")
    config = dict(config or {})
    unknown = set(config) - TRAINING_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown evaluation encoder training settings: {sorted(unknown)}.")
    settings = {**TRAINING_DEFAULTS, **config}
    for name in ("batch_size", "epochs", "early_stop"):
        settings[name] = _positive_integer(settings[name], f"training.{name}")
    settings["learning_rate"] = _number(
        settings["learning_rate"], "training.learning_rate", positive=True
    )
    return settings


def resolve_device(device="cpu"):
    """Validate a CPU or CUDA device before constructing training state."""
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:N'.") from exc
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("The evaluation encoder supports CPU and CUDA devices.")
    if resolved.type == "cpu" and resolved.index is not None:
        raise ValueError("Use 'cpu' without a device index.")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but it is not available.")
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is not available.")
        resolved = torch.device("cuda", index)
    return resolved


def _indices(values, length, window, label):
    indices = np.asarray(values)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
        raise ValueError(f"{label} must be a nonempty one-dimensional integer array.")
    if np.any(indices < window) or np.any(indices >= length):
        raise ValueError(f"{label} must have full preceding context and lie within the sequence.")
    indices = indices.astype(np.int64, copy=False)
    if np.any(np.diff(indices) <= 0):
        raise ValueError(f"{label} must be sorted and contain no duplicates.")
    return indices


class _ContextDataset(Dataset):
    """Materialize only one causal context per item, rather than every window."""

    def __init__(self, features, target_indices, window_size, targets=None):
        raw_features = np.asarray(features)
        if raw_features.ndim != 2 or raw_features.shape[1] == 0 or raw_features.dtype.kind not in "iuf":
            raise ValueError("features must be a real numeric array of shape (T, D), D > 0.")
        self.features = np.asarray(raw_features, dtype=np.float32)
        self.window_size = _positive_integer(window_size, "window_size")
        self.target_indices = _indices(
            target_indices, len(self.features), self.window_size, "target_indices"
        )
        # Inspect exactly the union of required contexts. In particular, the
        # outcome at the last requested target is not read for representations.
        left = int(self.target_indices[0]) - self.window_size
        right = int(self.target_indices[0])
        for index in self.target_indices[1:]:
            start, end = int(index) - self.window_size, int(index)
            if start > right:
                self._validate_context(left, right)
                left = start
            right = end
        self._validate_context(left, right)
        self.targets = None
        if targets is not None:
            raw_targets = np.asarray(targets)
            if raw_targets.ndim != 1 or len(raw_targets) != len(self.features) or raw_targets.dtype.kind not in "iuf":
                raise ValueError("y_normalized must be a real numeric vector aligned with features.")
            self.targets = np.asarray(raw_targets, dtype=np.float32)
            if not np.isfinite(self.targets[self.target_indices]).all():
                raise ValueError("Encoder fitting/validation targets must be finite.")

    def _validate_context(self, left, right):
        if not np.isfinite(self.features[left:right]).all():
            raise ValueError("Encoder contexts must contain finite observed values.")

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, position):
        index = int(self.target_indices[position])
        context = torch.from_numpy(self.features[index - self.window_size:index])
        if self.targets is None:
            return context
        return context, torch.tensor(self.targets[index], dtype=torch.float32)


def _epoch_loss(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for context, target in loader:
            context, target = context.to(device), target.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(context)[:, -1, 0]
            # rho_0.5(y - q) = |y - q| / 2, on the observed-response target.
            loss = 0.5 * torch.abs(target - prediction).mean()
            if not torch.isfinite(loss):
                raise ValueError("The evaluation encoder produced a nonfinite median loss.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += float(loss.detach().cpu()) * len(target)
            count += len(target)
    return total / count


def train_encoder(prepared, model_config=None, training_config=None, seed=2026, device="cpu"):
    """Fit fresh per-sequence weights and return the frozen best checkpoint.

    ``prepared`` supplies normalized observed ``features`` and ``y_normalized``
    plus chronological ``fit_indices``, ``validation_indices`` and the full
    eligible ``calibration_target_indices``. Reserved test targets never enter
    fitting or checkpoint selection. Random number state is restored on exit.
    """
    model_settings = resolve_model_config(model_config)
    training_settings = resolve_training_config(training_config)
    device = resolve_device(device)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be an integer in [0, 2**63).")
    seed = int(seed)
    window = model_settings["window_size"]
    features = prepared["features"]
    fit = _ContextDataset(features, prepared["fit_indices"], window, prepared["y_normalized"])
    valid = _ContextDataset(features, prepared["validation_indices"], window, prepared["y_normalized"])
    calibration_indices = _indices(
        prepared["calibration_target_indices"], len(fit.features), window, "calibration_target_indices"
    )
    if fit.target_indices[-1] >= valid.target_indices[0]:
        raise ValueError("Checkpoint validation must follow fitting chronologically.")
    if not np.isin(np.concatenate((fit.target_indices, valid.target_indices)), calibration_indices).all():
        raise ValueError("Encoder fitting and validation must stay within calibration.")
    if "test_indices" in prepared and np.intersect1d(
        calibration_indices, np.asarray(prepared["test_indices"])
    ).size:
        raise ValueError("Calibration and test targets must be disjoint.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fit_loader = DataLoader(fit, batch_size=training_settings["batch_size"], shuffle=True,
                            generator=generator, num_workers=0)
    valid_loader = DataLoader(valid, batch_size=training_settings["batch_size"], shuffle=False,
                              generator=generator, num_workers=0)
    devices = [device.index] if device.type == "cuda" else []
    history = {"fit_loss": [], "validation_loss": []}
    best_loss, best_epoch, best_state, stale_epochs = float("inf"), 0, None, 0
    with torch.random.fork_rng(devices=devices), torch.backends.cudnn.flags(benchmark=False, deterministic=True):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        model = RNNPredictor(
            rnn_type=model_settings["rnn_type"], dim_feature=fit.features.shape[1],
            dim_model=model_settings["dim_model"], num_layer=model_settings["num_layers"],
            prediction_step=1, dropout=model_settings["dropout"], training_quantiles=[0.5],
        ).to(device)
        model.window_size = window
        optimizer = torch.optim.Adam(model.parameters(), lr=training_settings["learning_rate"])
        for epoch in range(1, training_settings["epochs"] + 1):
            fit_loss = _epoch_loss(model, fit_loader, device, optimizer)
            valid_loss = _epoch_loss(model, valid_loader, device)
            history["fit_loss"].append(fit_loss)
            history["validation_loss"].append(valid_loss)
            if valid_loss < best_loss:
                best_loss, best_epoch, stale_epochs = valid_loss, epoch, 0
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            else:
                stale_epochs += 1
            if stale_epochs >= training_settings["early_stop"]:
                break
        model.load_state_dict(best_state)
    model.eval()
    model.requires_grad_(False)
    metadata = {
        **history,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_trained": len(history["fit_loss"]),
        "fit_count": len(fit),
        "validation_count": len(valid),
        "target": "observed_response",
        "quantile": 0.5,
        "loss": "median_pinball",
        "device": str(device),
        "model_config": model_settings,
        "training_config": training_settings,
    }
    return model, metadata


def extract_embeddings(model, prepared, target_indices, batch_size=256, device="cpu", normalization_epsilon=1e-12):
    """Return unit L2 final hidden states using only past observed contexts."""
    batch_size = _positive_integer(batch_size, "embedding batch_size")
    epsilon = _number(normalization_epsilon, "normalization_epsilon", positive=True)
    device = resolve_device(device)
    window = getattr(model, "window_size", prepared.get("window_size"))
    dataset = _ContextDataset(prepared["features"], target_indices, window)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        generator=torch.Generator(device="cpu").manual_seed(0))
    model.to(device)
    model.eval()
    representations = []
    with torch.no_grad():
        for context in loader:
            hidden = model.encode(model, context.to(device))[:, -1, :]
            norms = torch.linalg.vector_norm(hidden, ord=2, dim=1, keepdim=True)
            if not torch.isfinite(hidden).all() or not torch.isfinite(norms).all():
                raise ValueError("Encoder embeddings must be finite.")
            if torch.any(norms <= epsilon):
                raise ValueError("Encoder produced a zero or near-zero embedding; L2 normalization is undefined.")
            representations.append((hidden / norms).cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(representations, axis=0)
