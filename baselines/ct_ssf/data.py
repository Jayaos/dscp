"""Validate paired feature/head artifacts and preserve forecast chronology."""

import math

import numpy as np
import torch

from base_predictor.chronos_features import FEATURE_FORMAT_VERSION


def _check_reconstruction(reconstructed, model_outputs, predictions, location, scale,
                          raw_indices, base_predictor):
    """Keep the pairing guard while accounting for float32 affine roundoff.

    LSTM kernels/batch shapes may differ between saving and reconstruction.
    Near zero, a relative tolerance on the denormalized forecast alone misses
    cancellation of much larger location/scale terms. Allow a small numerical
    budget tied to those terms, without fitting a correction to the forecasts
    or changing them. This is a practical tolerance, not a rounding proof.
    """
    location, scale = np.asarray(location, dtype=float), np.asarray(scale, dtype=float)
    strict_tolerance = 1e-5 + 1e-4 * np.abs(predictions)
    tolerance = strict_tolerance.copy()
    roundoff_factor = 8 if base_predictor == "lstm" else 0
    if roundoff_factor:
        tolerance += roundoff_factor * np.finfo(np.float32).eps * (
            np.abs(location) + np.abs(model_outputs * scale) + scale
        )
    error = np.abs(reconstructed - predictions)
    finite = np.isfinite(reconstructed) & np.isfinite(model_outputs)
    matches = finite & (error <= tolerance)
    diagnostics = {
        "max_abs_error": float(error.max()), "mean_abs_error": float(error.mean()),
        "max_abs_error_over_scale": float((error / scale).max()),
        "strict_mismatch_count": int((~finite | (error > strict_tolerance)).sum()),
        "failed_count": int((~matches).sum()), "sample_count": len(predictions),
        "rtol": 1e-4, "atol": 1e-5, "float32_roundoff_factor": roundoff_factor,
        "max_allowed_error": float(tolerance.max()),
    }
    if not matches.all():
        first = int(np.flatnonzero(~matches)[0])
        checkpoint_field = "model_path" if base_predictor == "lstm" else "head_path"
        raise ValueError(
            f"CT-SSF {base_predictor} features/head do not reconstruct heldout_predictions: "
            f"{diagnostics['failed_count']}/{len(predictions)} rows failed; "
            f"max_abs_error={diagnostics['max_abs_error']:.9g}, "
            f"max_abs_error_over_scale={diagnostics['max_abs_error_over_scale']:.9g}. "
            f"First failing heldout index={first}, raw index={int(raw_indices[first])}: "
            f"saved={predictions[first]:.9g}, reconstructed={reconstructed[first]:.9g}, "
            f"allowed_error={tolerance[first]:.9g}. "
            f"Check that data_path and {checkpoint_field} come from the same base-predictor run."
        )
    return diagnostics


def prepare_sequence(item, head, artifact, config):
    base_predictor = artifact.get("base_predictor", "chronos")
    if base_predictor not in {"chronos", "lstm"}:
        raise ValueError("Unsupported CT-SSF base predictor.")
    if "ctssf" not in item:
        raise ValueError("Missing CT-SSF features. Export Chronos with --save-ctssf or load the saved LSTM checkpoint.")
    cache = item["ctssf"]
    if cache.get("format_version") != FEATURE_FORMAT_VERSION or cache.get("head_sha256") != artifact["head_sha256"]:
        raise ValueError("CT-SSF feature format/head fingerprint mismatch.")
    if base_predictor == "lstm" and cache.get("checkpoint_sha256") != artifact["checkpoint_sha256"]:
        raise ValueError("CT-SSF LSTM feature/checkpoint fingerprint mismatch.")
    arrays = []
    for name in ("heldout_y", "heldout_predictions"):
        array = np.asarray(item[name], dtype=np.float64)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError(f"{name} must have shape [T] or [T, 1].")
        arrays.append(array)
    y, pred = arrays
    n = len(y)
    if n == 0 or pred.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Targets/predictions must be matching, nonempty finite scalar sequences.")
    features = np.asarray(cache["features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != artifact["architecture"]["input_dim"] or not np.isfinite(features).all():
        raise ValueError("Invalid CT-SSF feature matrix.")
    fields = {}
    for name in ("feature_indices", "output_indices", "forecast_origin", "lead", "raw_target_indices", "location", "scale"):
        values = np.asarray(cache[name])
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid or misaligned CT-SSF field: {name}.")
        if name not in {"location", "scale"} and (values.dtype.kind not in "iu" or (values < 0).any()):
            raise ValueError(f"CT-SSF {name} must contain nonnegative integer indices.")
        fields[name] = values
    mapping, offset = fields["feature_indices"], fields["output_indices"]
    raw, origin, lead = (fields[k] for k in ("raw_target_indices", "forecast_origin", "lead"))
    provenance = artifact["provenance"]
    if mapping.max() >= len(features) or offset.max() >= head.output_patch_size:
        raise ValueError("CT-SSF feature/output index is out of range.")
    if (fields["scale"] <= 0).any() or (lead < 1).any():
        raise ValueError("CT-SSF scale and forecast lead must be positive.")
    if not np.array_equal(raw, origin + lead - 1) or not np.all(np.diff(raw) == 1):
        raise ValueError("CT-SSF target indices do not follow forecast origins/leads.")
    if not np.array_equal(offset, (lead - 1) % head.output_patch_size):
        raise ValueError("CT-SSF output indices disagree with forecast leads.")
    window, horizon = provenance.get("window_length"), provenance.get("prediction_length")
    if not isinstance(window, int) or not isinstance(horizon, int) or min(window, horizon) < 1:
        raise ValueError("Head provenance is missing the forecast window/horizon.")
    if provenance.get("uses_future_covariates") is not False:
        raise ValueError("These experiments require past-only covariates.")
    raw_start = len(item["train_y"]) if base_predictor == "lstm" else window
    if base_predictor == "lstm" and (horizon != 1 or raw_start < window):
        raise ValueError("LSTM requires one-step forecasts and sufficient training history.")
    expected_origin = raw_start + (np.arange(n) // horizon) * horizon
    if raw[0] != raw_start or not np.array_equal(origin, expected_origin) or not np.array_equal(lead, np.arange(n) % horizon + 1):
        raise ValueError("CT-SSF chronology does not match the saved forecast setup.")
    # Each feature vector must belong to exactly one (origin, output patch).
    pairs = np.stack([origin, (lead - 1) // head.output_patch_size], axis=1)
    unique_pairs, expected_mapping = np.unique(pairs, axis=0, return_inverse=True)
    if len(unique_pairs) != len(features) or not np.array_equal(mapping, expected_mapping):
        raise ValueError("CT-SSF feature mapping does not match forecast blocks.")
    expanded = features[mapping]
    device = next(head.parameters()).device
    reconstructed = np.empty(n, dtype=np.float64)
    model_outputs = np.empty(n, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n, 512):
            sl = slice(start, start + 512)
            z = torch.as_tensor(expanded[sl], device=device)
            idx = torch.as_tensor(offset[sl], device=device)
            loc = torch.as_tensor(fields["location"][sl], device=device)
            scl = torch.as_tensor(fields["scale"][sl], device=device)
            values = head.select(z, idx)
            model_outputs[sl] = values.cpu().numpy()
            reconstructed[sl] = head.to_response(values, loc, scl).cpu().numpy()
    reconstruction = _check_reconstruction(
        reconstructed, model_outputs, pred, fields["location"], fields["scale"], raw, base_predictor,
    )
    ratios = [float(config[name]) for name in ("train_ratio", "valid_ratio", "calibration_ratio", "test_ratio")]
    if not all(math.isfinite(r) and r > 0 for r in ratios) or not math.isclose(sum(ratios), 1, abs_tol=1e-10):
        raise ValueError("Positive train/valid/calibration/test ratios must sum to one.")
    boundaries = [math.floor(np.nextafter(n * sum(ratios[:i]), np.inf)) for i in (1, 2, 3)]
    train_end, valid_end, test_start = boundaries
    if not 2 <= train_end < valid_end < test_start < n:
        raise ValueError("Sequence is too short for nonempty chronological CT-SSF folds.")
    # The query uses only forecast-time information, including which output
    # coordinate is forecast. It never uses heldout_x at a future timestamp.
    inputs = np.concatenate([expanded, np.eye(head.output_patch_size, dtype=np.float32)[offset]], axis=1)
    return {
        "y": y, "predictions": pred, "features": expanded, "attention_inputs": inputs,
        "prediction_reconstruction": reconstruction,
        **fields,
        "boundaries": {"train_end": train_end, "valid_end": valid_end, "calibration_start": valid_end, "test_start": test_start, "test_end": n},
    }
