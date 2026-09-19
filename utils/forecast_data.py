"""Validate scalar forecast artifacts before residual arithmetic."""

from collections.abc import Mapping

import numpy as np


def _scalar_column(values, field, sequence):
    label = f"Forecast sequence {sequence!r} field {field!r}"
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric sequence with shape (T,) or (T, 1).") from exc
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2 or array.shape[1] != 1:
        raise ValueError(f"{label} must have shape (T,) or (T, 1); got {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError(f"{label} must be nonempty.")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain real numeric values; got dtype {array.dtype}.")
    return array


def canonicalize_forecast_data(data):
    """Return scalar targets and predictions as matching ``(T, 1)`` arrays.

    Accept vectors, single-column arrays, and lists without changing values,
    dtypes, or time order. Reject other shapes instead of flattening multiple
    targets or forecast horizons into the time axis. Validate lengths before
    any subtraction can broadcast a vector against a column into ``(T, T)``.

    Copy the dictionaries so preparation can add or replace fields without
    modifying the caller's artifact. Array storage is shared where possible;
    covariates, optional forecaster history, and other metadata are unchanged.
    An empty mapping is allowed for callers that fill prepared datasets later.

    This handles structure only. It does not fit normalization statistics or
    inspect finite values in reserved evaluation targets.
    """
    if not isinstance(data, Mapping):
        raise ValueError("Forecast data must be a mapping of sequence names to records.")
    prepared = {}
    for sequence, item in data.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"Forecast sequence {sequence!r} must be a mapping.")
        missing = {"heldout_y", "heldout_predictions"} - item.keys()
        if missing:
            raise ValueError(f"Forecast sequence {sequence!r} is missing fields: {sorted(missing)}.")
        targets = _scalar_column(item["heldout_y"], "heldout_y", sequence)
        predictions = _scalar_column(item["heldout_predictions"], "heldout_predictions", sequence)
        if len(targets) != len(predictions):
            raise ValueError(
                f"Forecast sequence {sequence!r}: heldout_y and heldout_predictions "
                f"must have matching lengths; got {len(targets)} and {len(predictions)}."
            )
        prepared[sequence] = {
            **item,
            "heldout_y": targets,
            "heldout_predictions": predictions,
        }
    return prepared
