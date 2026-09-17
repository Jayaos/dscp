"""Prepare SPCI tuning examples entirely within the training prefix."""

from baselines.spci.data import _fraction, prepare_spci_data


def prepare_spci_tuning_data(data, config):
    """Fit on the training prefix's head and evaluate its held-out tail."""
    tuning = config.get("tuning", {}) or {}
    ratio = _fraction(
        tuning.get("model_selection_valid_ratio", 0.2),
        "tuning.model_selection_valid_ratio",
    )
    return prepare_spci_data(data, config, model_selection_valid_ratio=ratio)
