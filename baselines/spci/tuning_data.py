"""Prepare SPCI tuning examples entirely within the nominal training prefix."""

import copy
from numbers import Integral

import numpy as np

from dscp.data import ConformalPredictionData


def _fraction(value, name):
    try:
        fraction = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1.") from exc
    if isinstance(value, bool) or not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1.")
    return fraction


def prepare_spci_tuning_data(data, config):
    """Return fit/validation datasets without exposing outer validation or test.

    The later ``tuning.model_selection_valid_ratio`` fraction of the nominal
    training region evaluates candidate forests. Earlier observations fit the
    forest and determine normalization. The nested split helper names this
    evaluation subset ``model_selection_valid_dataset``; SPCI has no separate
    epoch or checkpoint selection step.
    """
    tuning = config.get("tuning", {}) or {}
    ratio = _fraction(
        tuning.get("model_selection_valid_ratio", 0.2),
        "tuning.model_selection_valid_ratio",
    )
    window = config.model.window_size
    if isinstance(window, bool) or not isinstance(window, Integral) or window <= 0:
        raise ValueError("model.window_size must be a positive integer.")
    window = int(window)
    horizon = config.model.prediction_step
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon != 1:
        raise ValueError("SPCI tuning supports model.prediction_step=1 only.")

    train_ratio = _fraction(config.data.train_ratio, "data.train_ratio")
    valid_ratio = _fraction(config.data.valid_ratio, "data.valid_ratio")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("data.train_ratio + data.valid_ratio must be less than 1.")
    if not data:
        raise ValueError("SPCI tuning requires at least one sequence.")

    for key, sequence in data.items():
        size = len(sequence["heldout_y"])
        if len(sequence["heldout_x"]) != size or len(sequence["heldout_predictions"]) != size:
            raise ValueError(f"Held-out arrays must have matching lengths for {key!r}.")
        nominal_train_size = int(np.floor(size * train_ratio))
        fit_size = int(np.floor(np.nextafter(nominal_train_size * (1 - ratio), np.inf)))
        inner_valid_size = nominal_train_size - fit_size
        outer_valid_size = int(np.ceil(size * valid_ratio))
        test_size = size - nominal_train_size - outer_valid_size
        if min(fit_size - window, inner_valid_size, outer_valid_size, test_size) <= 0:
            raise ValueError(
                f"Insufficient data for SPCI tuning on {key!r}: "
                f"length={size}, fit={fit_size}, window_size={window}, "
                f"inner_validation={inner_valid_size}, "
                f"outer_validation={outer_valid_size}, test={test_size}."
            )

    prepared = ConformalPredictionData(copy.deepcopy(data))
    prepared.prepare_quantile_regression_datasets(
        window,
        1,
        train_ratio,
        valid_ratio,
        normalize=config.data.normalize,
        model_selection_valid_ratio=ratio,
    )
    for key, datasets in prepared.dataset.items():
        prepared.dataset[key] = {
            "train_dataset": datasets["train_dataset"],
            "model_selection_valid_dataset": datasets["model_selection_valid_dataset"],
        }
    return prepared
