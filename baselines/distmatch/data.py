"""Causal residual preparation on the saved base-predictor held-out suffix."""

import numpy as np

from baselines.distmatch.config import normalize_residual_enabled, split_boundaries


def _sequence_array(values, name):
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty scalar sequence with shape [T] or [T, 1].")
    return array


def scalar_sequence(values, name="residuals"):
    try:
        array = np.asarray(_sequence_array(values, name), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric scalar sequence.") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _normalization(item, data_config, y, evaluation_start):
    """Use upstream's target statistics without reading evaluation targets.

    Include saved ``train_y`` when available. Artifacts without that history
    (such as Chronos forecasts) use only the observed held-out prefix.
    torch.std in the upstream loader uses the sample standard deviation.
    """
    enabled = normalize_residual_enabled(data_config)
    info = {"enabled": enabled, "target_mean": 0.0, "target_std": 1.0}
    if not enabled:
        return info
    history = y[:evaluation_start]
    source = "heldout_y[:evaluation_start]"
    if "train_y" in item:
        train_y = scalar_sequence(item["train_y"], "train_y (target normalization)")
        history = np.concatenate((train_y, history))
        source = "train_y + heldout_y[:evaluation_start]"
    if len(history) < 2:
        raise ValueError("Upstream target normalization requires at least two outcomes.")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(history.mean())
        std = float(history.std(ddof=1))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("Target normalization statistics must be finite.")
    info.update({
        "target_mean": mean,
        "target_std": std if std > 0 else 1.0,
        "ddof": 1,
        "fit_size": int(len(history)),
        "heldout_fit_end": int(evaluation_start),
        "source": source,
    })
    return info


def prepare_sequence(item, config, split="test"):
    """Slice the permitted region before validating or transforming its values.

    In validation mode the reserved test suffix is used only for its length.
    Target normalization includes saved ``train_y`` when present; otherwise it
    uses only the held-out prefix preceding the active evaluation split.
    No point forecaster is fitted and no covariates are consumed.
    """
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")
    if not isinstance(item, dict):
        raise ValueError("Each forecast series must be a mapping.")
    missing = {"heldout_y", "heldout_predictions"} - set(item)
    if missing:
        raise ValueError(f"Forecast series is missing fields: {sorted(missing)}")
    y_array = _sequence_array(item["heldout_y"], "heldout_y")
    prediction_array = _sequence_array(item["heldout_predictions"], "heldout_predictions")
    if len(y_array) != len(prediction_array):
        raise ValueError("heldout_y and heldout_predictions must have matching lengths.")
    data_config = config["data"]
    boundaries = split_boundaries(len(y_array), *(
        data_config[name] for name in ("train_ratio", "valid_ratio", "test_ratio")
    ))
    train_end, valid_end = boundaries["train_end"], boundaries["validation_end"]
    if split == "validation":
        if boundaries["validation_size"] == 0:
            raise ValueError("Validation evaluation requires a positive data.valid_ratio.")
        start, end = train_end, valid_end
    else:
        start, end = valid_end, len(y_array)
    y = scalar_sequence(y_array[:end], "heldout_y (available prefix)")
    predictions = scalar_sequence(prediction_array[:end], "heldout_predictions (available prefix)")
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = y - predictions
    if not np.isfinite(residuals).all():
        raise ValueError("Signed residuals must remain finite after subtraction.")
    normalization = _normalization(item, data_config, y, start)
    # (y - mean) / std - (prediction - mean) / std = residual / std.
    # Scale both the windows and regression targets, without residual centering.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residuals = residuals / normalization["target_std"]
    if not np.isfinite(residuals).all():
        raise ValueError("Signed residuals must remain finite after target normalization.")
    return {
        "train_residuals": residuals[:train_end].copy(),
        "warmup_residuals": residuals[train_end:start].copy(),
        "residuals": residuals[start:end].copy(),
        "y": y[start:end].copy(),
        "predictions": predictions[start:end].copy(),
        "target_indices": np.arange(start, end, dtype=np.int64),
        "boundaries": boundaries,
        "normalization": normalization,
    }
