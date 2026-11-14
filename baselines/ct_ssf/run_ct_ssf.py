"""Prequential CT-SSF adapters for saved Chronos and LSTM base predictors."""

import hashlib
from pathlib import Path
import pickle
import time
import warnings

import numpy as np
from omegaconf import OmegaConf
import torch

from base_predictor.chronos_features import load_chronos_head
from baselines.ct_ssf.attention import fit_attention
from baselines.ct_ssf.bounds import HeadBounder
from baselines.ct_ssf.data import prepare_sequence
from baselines.ct_ssf.scores import latent_scores, weighted_quantile
from utils.experiment_config import load_experiment_config
from utils.utils import save_data


IMPLEMENTATION_VERSION = 2
UPSTREAM_COMMIT = "28320c61374c93746db8170b641c29cbf313d7bc"
INVERSION_REFERENCE_COMMIT = "85843cbc4d54470881b2ee958d692c5d47448375"


def target_quantiles(config):
    pairs = [tuple(map(float, pair)) for pair in config["model"]["target_quantiles"]]
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("Specify nonempty, distinct target_quantiles pairs.")
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("Each interval needs two quantile levels.")
        low, high = sorted(pair)
        if not 0 < low < high < 1 or not np.isclose(low + high, 1):
            raise ValueError("CT-SSF presets support symmetric nominal coverage pairs only.")
    return pairs


def _metrics(lower, upper, y, predictions, indices, pair, radius, feature_scores, reconstructed):
    alpha = 1 - abs(pair[1] - pair[0])
    coverage = (y >= lower) & (y <= upper)
    feature_coverage = feature_scores <= radius
    width = upper - lower
    winkler = width.copy()
    # Masked assignment avoids 0 * infinity producing NaN.
    below, above = y < lower, y > upper
    winkler[below] += (2 / alpha) * (lower[below] - y[below])
    winkler[above] += (2 / alpha) * (y[above] - upper[above])
    return {
        "lower_interval": lower.tolist(), "upper_interval": upper.tolist(),
        "lower_residual_quantile": (lower - predictions).tolist(),
        "upper_residual_quantile": (upper - predictions).tolist(),
        "target_y": y.tolist(), "target_predictions": predictions.tolist(),
        "target_indices": indices.tolist(), "latent_radius": radius.tolist(),
        "coverage": coverage.tolist(), "interval_width": width.tolist(), "winkler_score": winkler.tolist(),
        "avg_coverage": float(coverage.mean()), "avg_delta_coverage": float(coverage.mean() - (1 - alpha)),
        "feature_coverage": feature_coverage.tolist(), "avg_feature_coverage": float(feature_coverage.mean()),
        "avg_surrogate_output_coverage": float(((reconstructed >= lower) & (reconstructed <= upper)).mean()),
        "avg_interval_width": float(width.mean()), "avg_winkler_score": float(winkler.mean()),
        "infinite_interval_count": int(np.isinf(width).sum()),
    }


def _relative_squared_error(error, target):
    numerator, denominator = np.square(error).sum(), np.square(target).sum()
    return float(numerator / denominator) if denominator > 0 else (0.0 if numerator == 0 else float("inf"))


def _inversion_diagnostics(errors, details, targets, begin, end):
    """Report reconstruction fidelity separately from feature/output coverage."""
    residual = errors[begin:end].astype(float)
    y = targets[begin:end].astype(float)
    response_error = details["reconstructed_targets"][begin:end] - y
    # For a zero target: exact reconstruction passes, nonzero error fails.
    row_passes = np.square(response_error) < 0.01 * np.square(y)
    row_passes |= (y == 0) & (response_error == 0)
    return {
        "mean_abs_model_scale_error": float(residual.mean()),
        "max_abs_model_scale_error": float(residual.max()),
        "relative_squared_model_scale_error": _relative_squared_error(residual, details["model_targets"][begin:end].astype(float)),
        "relative_squared_response_error": _relative_squared_error(response_error, y),
        "fraction_response_rows_below_1pct": float(row_passes.mean()),
    }


def evaluate_sequence(key, item, config, head, artifact, checkpoint_dir=None):
    started = time.perf_counter()
    base_predictor = config.get("base_predictor", "chronos")
    if base_predictor not in {"chronos", "lstm"} or base_predictor != artifact.get("base_predictor", "chronos"):
        raise ValueError("CT-SSF base_predictor does not match the loaded head.")
    method = f"ct_ssf_{base_predictor}_adapted"
    model_config = config["model"]
    allowed = {"target_quantiles", "calibration_size", "rolling_calibration", "weighting", "bound_method", "inverse_steps", "inverse_lr", "inverse_optimizer", "batch_size"}
    if set(model_config) - allowed:
        raise ValueError(f"Unknown CT-SSF model settings: {sorted(set(model_config) - allowed)}")
    pairs = target_quantiles(config)
    batch_size = int(model_config.get("batch_size", 256))
    memory_size = int(model_config.get("calibration_size", 500))
    if min(memory_size, batch_size) < 1:
        raise ValueError("Calibration and execution batch sizes must be positive.")
    rolling = model_config.get("rolling_calibration", True)
    if not isinstance(rolling, bool):
        raise ValueError("rolling_calibration must be boolean.")
    weighting = model_config.get("weighting", "learned_attention")
    if weighting not in {"learned_attention", "uniform"}:
        raise ValueError("weighting must be learned_attention or uniform (FeatureCP ablation).")
    identity = f"{int(config.get('seed', 2026))}\0{type(key).__name__}\0{key}".encode()
    seed = int.from_bytes(hashlib.sha256(identity).digest()[:4], "little")
    torch.manual_seed(seed)
    data = prepare_sequence(item, head, artifact, config["data"])
    boundaries = data["boundaries"]
    train_end, valid_end, start = (boundaries[name] for name in ("train_end", "valid_end", "test_start"))
    n = len(data["y"])
    device = next(head.parameters()).device
    # Inversions are independent, stateless per-row operations. Bulk execution
    # is only an optimization: at time t the calibrator reads scores strictly
    # before t. No training, quantile, or model selection sees future scores.
    scores, errors, inverse = latent_scores(
        head, data["features"], data["output_indices"], data["y"], data["location"], data["scale"],
        steps=model_config.get("inverse_steps", 200), learning_rate=model_config.get("inverse_lr", 0.01),
        optimizer=model_config.get("inverse_optimizer", "adam"), batch_size=batch_size,
        return_details=True, semantic_end=valid_end if weighting == "learned_attention" else 0,
    )
    attention_info, attention_checkpoint = {}, None
    if weighting == "learned_attention":
        attention, attention_info = fit_attention(data["attention_inputs"], inverse["semantic_errors"], train_end, valid_end, config.get("attention", {}), device)
        encoded = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                encoded.append(attention(torch.as_tensor(data["attention_inputs"][i:i + batch_size], device=device)))
        encoded = torch.cat(encoded)
        if checkpoint_dir is not None:
            attention_checkpoint = f"attention_{hashlib.sha256(identity).hexdigest()[:16]}.pt"
            torch.save({
                "method": method, "base_predictor": base_predictor, "input_dim": data["attention_inputs"].shape[1],
                "implementation_version": IMPLEMENTATION_VERSION,
                "options": dict(config.get("attention", {})), "head_sha256": artifact["head_sha256"],
                "base_checkpoint_sha256": artifact.get("checkpoint_sha256"),
                "state_dict": {k: v.cpu() for k, v in attention.state_dict().items()},
                "seed": seed, "boundaries": boundaries, **attention_info,
            }, Path(checkpoint_dir) / attention_checkpoint)
    # Displacements are no longer needed after fitting. Never retain a full
    # test-feature displacement matrix merely to compute scalar diagnostics.
    del inverse["semantic_errors"]
    radii = {pair: [] for pair in pairs}
    effective_sizes, calibration_sizes = [], []
    for t in range(start, n):
        end = t if rolling else start
        begin = max(valid_end, end - memory_size)
        if weighting == "learned_attention":
            with torch.no_grad():
                masses, self_mass = attention.weights(encoded[t], encoded[begin:end])
                weights, test_weight = masses.cpu().numpy(), float(self_mass)
        else:
            weights, test_weight = np.ones(end - begin), 1.0
        for pair in pairs:
            radii[pair].append(weighted_quantile(scores[begin:end], weights, 1 - abs(pair[1] - pair[0]), test_weight))
        effective_sizes.append(float(weights.sum() ** 2 / max(np.square(weights).sum(), 1e-300)))
        calibration_sizes.append(end - begin)
        # Only the NEXT iteration can read scores[t], after y[t] is observed.
    bounder = HeadBounder(head, model_config.get("bound_method", "crown"))
    results = {}
    for pair in pairs:
        radius = np.asarray(radii[pair])
        lower, upper = [], []
        for i in range(start, n, batch_size):
            sl = slice(i, i + batch_size)
            lo, hi = bounder(
                torch.as_tensor(data["features"][sl], device=device), data["output_indices"][sl],
                radius[i - start:i - start + batch_size], data["location"][sl], data["scale"][sl],
            )
            lower.extend(lo.cpu().tolist())
            upper.extend(hi.cpu().tolist())
        results[pair] = _metrics(np.asarray(lower), np.asarray(upper), data["y"][start:].astype(float), data["predictions"][start:].astype(float), np.arange(start, n), pair, radius, scores[start:], inverse["reconstructed_targets"][start:])
    diagnostics = {}
    for name, begin, end in (("training", 0, train_end), ("validation", train_end, valid_end), ("calibration", valid_end, start), ("test", start, n)):
        diagnostics[name] = _inversion_diagnostics(errors, inverse, data["y"], begin, end)
    if diagnostics["calibration"]["relative_squared_response_error"] >= 0.01:
        warnings.warn(f"{key!r}: latent inversion has appreciable residual error; inspect inverse_diagnostics before interpreting coverage.")
    return {
        "evaluation_results": results,
        "metadata": {
            "method": method, "base_predictor": base_predictor, "split": "test", "boundaries": boundaries,
            "implementation_version": IMPLEMENTATION_VERSION,
            "head_sha256": artifact["head_sha256"], "head_provenance": artifact["provenance"],
            "prediction_reconstruction": data["prediction_reconstruction"],
            "base_checkpoint_sha256": artifact.get("checkpoint_sha256"),
            "upstream_commit": UPSTREAM_COMMIT, "inversion_reference_commit": INVERSION_REFERENCE_COMMIT,
            "gradient_weighting": "uniform_upstream_latent_inversion", "weighting": weighting,
            "bound_method": bounder.method, "interval_scale": "original_response", "sequence_seed": seed,
            "attention_checkpoint": attention_checkpoint, "attention_training": attention_info,
            "inverse_diagnostics": diagnostics, "calibration_size": calibration_sizes,
            "inverse_settings": {"steps": model_config.get("inverse_steps", 200), "learning_rate": model_config.get("inverse_lr", 0.01), "optimizer": model_config.get("inverse_optimizer", "adam"), "selection": "fixed_config_no_test_tuning"},
            "effective_calibration_size": effective_sizes,
            "forecast_origin": data["forecast_origin"][start:].tolist(), "lead": data["lead"][start:].tolist(),
            "raw_target_indices": data["raw_target_indices"][start:].tolist(),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }


def summarize(log, pairs):
    summary = {}
    for pair in pairs:
        result = {}
        for metric in ("avg_coverage", "avg_delta_coverage", "avg_interval_width", "avg_winkler_score"):
            values = np.asarray([item["evaluation_results"][pair][metric] for item in log.values()])
            result[f"{metric}_mean"] = float(values.mean())
            # A dispersion of infinite widths is undefined; retain NaN explicitly.
            result[f"{metric}_std"] = float(values.std()) if np.isfinite(values).all() else float("nan")
        result["infinite_interval_count"] = sum(item["evaluation_results"][pair]["infinite_interval_count"] for item in log.values())
        summary[pair] = result
    return summary


def run_ct_ssf(config_path):
    config = load_experiment_config(config_path)
    plain = OmegaConf.to_container(config, resolve=True)
    threads = int(plain.get("num_threads", 4))
    if threads < 1:
        raise ValueError("num_threads must be positive.")
    torch.set_num_threads(threads)
    device = plain.get("device", "cpu")
    base_predictor = plain.get("base_predictor", "chronos")
    if base_predictor == "chronos":
        head, artifact = load_chronos_head(plain["data"]["head_path"], device=device)
    elif base_predictor == "lstm":
        from baselines.ct_ssf.lstm import attach_lstm_features, load_lstm_backbone
        backbone, head, checkpoint, artifact = load_lstm_backbone(plain["data"]["model_path"], device=device)
    else:
        raise ValueError("base_predictor must be 'chronos' or 'lstm'.")
    with open(plain["data"]["data_path"], "rb") as stream:
        data = pickle.load(stream)
    if not data:
        raise ValueError("Prediction artifact has no sequences.")
    output = Path(plain["saving_dir"])
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "resolved_config.yaml", resolve=True)
    log = {}
    for key, item in data.items():
        print(f"CT-SSF {base_predictor} adaptation: {key}", flush=True)
        if base_predictor == "lstm":
            item = attach_lstm_features(item, backbone, checkpoint, artifact, batch_size=int(plain["model"].get("batch_size", 256)))
        log[key] = evaluate_sequence(key, item, plain, head, artifact, checkpoint_dir=output)
        save_data(output / "log.pkl", log)
    summary = summarize(log, target_quantiles(plain))
    save_data(output / "summary_results.pkl", summary)
    print(summary, flush=True)
    if plain.get("plotting", {}).get("plotting", False):
        from utils.plotting import plot_cp_prediction_intervals
        plot_cp_prediction_intervals(log, target_quantiles(plain), plain["plotting"].get("plotting_seq_len", 200), str(output / "plots"))
    return log
