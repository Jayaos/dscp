"""Score saved QR/IQN/Local-CP logs: python -m conditional_coverage.evaluate."""

import argparse
from collections.abc import Mapping
import csv
import hashlib
import json
from pathlib import Path
import pickle
import tempfile

import numpy as np
from omegaconf import OmegaConf
import yaml

from .alignment import align_result
from .artifacts import load_group_artifact
from .config import REPO_ROOT, file_sha256, resolve_path
from .data import _method_boundary, observed_sequence
from .evaluation_config import load_evaluation_config, validate_evaluation_config
from .metrics import CACE_DEFINITION, aggregate_sequences, evaluate_sequence


def _load_pickle(path):
    payload = Path(path).read_bytes()
    return pickle.loads(payload), hashlib.sha256(payload).hexdigest()


def _load_run(entry, source_hash):
    log_path = Path(entry["run_dir"]) / "log.pkl"
    log, log_hash = _load_pickle(log_path)
    if not isinstance(log, Mapping) or not log:
        raise ValueError(f"{log_path}: expected a nonempty mapping of sequences.")
    saved_config = Path(entry["run_dir"]) / "resolved_config.yaml"
    config_path = saved_config if saved_config.is_file() else (
        Path(entry["config_path"]) if "config_path" in entry else None
    )
    audit = {
        "name": entry["name"], "method": entry["method"], "run_dir": entry["run_dir"],
        "log_path": str(log_path), "log_sha256": log_hash,
        "allow_chronological_fallback": entry["allow_chronological_fallback"],
        "source_validation": "logged_targets",
    }
    method_config = None
    if config_path is not None:
        loaded = OmegaConf.load(config_path)
        if not OmegaConf.is_dict(loaded) or "data" not in loaded:
            raise ValueError(f"{config_path}: expected a method configuration with data settings.")
        horizon = loaded.get("model", {}).get("prediction_step", 1)
        if isinstance(horizon, bool) or horizon != 1:
            raise ValueError(f"{entry['name']}: only one-step evaluation is supported.")
        method_config = {
            "data": OmegaConf.to_container(loaded.data, resolve=True),
            "model": {"prediction_step": 1},
        }
        audit.update(config_path=str(config_path), config_sha256=file_sha256(config_path))
        configured_source = method_config["data"].get("data_path")
        if configured_source is not None:
            source = resolve_path(configured_source, REPO_ROOT)
            audit["configured_source_path"] = str(source)
            if source.is_file():
                if file_sha256(source) != source_hash:
                    raise ValueError(f"{entry['name']}: method source artifact differs from shared groups.")
                audit["source_validation"] = "source_checksum_and_logged_targets"
            else:
                audit["configured_source_unavailable"] = True
    return {"entry": entry, "log": log, "config": method_config, "audit": audit}


def _select_sequences(config, groups, source, runs):
    requested = config.get("sequences")
    keys = list(groups) if requested is None else requested
    selected, skipped = [], []
    for key in keys:
        missing = []
        if key not in groups:
            missing.append("groups")
        if key not in source:
            missing.append("source")
        for run in runs:
            if key not in run["log"]:
                missing.append(run["entry"]["name"])
        if missing:
            if requested is not None or "source" in missing:
                raise ValueError(f"Sequence {key!r} is missing from {missing}.")
            skipped.append({"sequence_key": key, "missing_from": missing})
        else:
            selected.append(key)
    if not selected:
        raise ValueError("No sequences are shared by the groups and all selected runs.")
    return selected, skipped


def _write_report(output_dir, report, config):
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Evaluation output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temporary:
        staging = Path(temporary).resolve()
        if staging.parent != destination.parent:
            raise ValueError("Evaluation staging directory escaped its intended parent.")
        with (staging / "evaluation.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        with (staging / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        for name, rows in (
            ("summary", report["summary"]), ("sequence_metrics", report["sequence_metrics"]),
            ("group_metrics", report["group_metrics"]),
        ):
            fields = list(dict.fromkeys(field for row in rows for field in row))
            with (staging / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        if destination.exists():
            raise FileExistsError(f"Evaluation output already exists: {destination}")
        staging.rename(destination)
    return destination


def evaluate_saved_runs(config, *, dry_run=False, progress=print):
    """Read existing artifacts, align observations, and report conditional coverage.

    No encoder, conformal model, or cluster is fitted. Dry runs perform the same
    validation and metric calculations but create no files.
    """
    config = validate_evaluation_config(config)
    output = Path(config["output_dir"])
    if not dry_run and output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    manifest, groups = load_group_artifact(config["groups_dir"])
    k = manifest.get("k")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("Grouping manifest must identify its single positive integer k.")
    if manifest.get("horizon", 1) != 1:
        raise ValueError("Only one-step grouping artifacts are supported.")
    if any(set(record["groups"]) != {k} for record in groups.values()):
        raise ValueError("Grouping manifest k disagrees with saved partitions.")
    source_info = manifest.get("source", {})
    data_path = config.get("data_path", source_info.get("path"))
    if not data_path or not source_info.get("sha256"):
        raise ValueError("Grouping manifest must identify its source path and checksum.")
    source, source_hash = _load_pickle(data_path)
    if source_hash != source_info["sha256"]:
        raise ValueError("Evaluation source artifact differs from the source used to build groups.")
    if not isinstance(source, Mapping) or not source:
        raise ValueError("Source artifact must be a nonempty sequence mapping.")
    runs = [_load_run(entry, source_hash) for entry in config["runs"]]
    keys, skipped = _select_sequences(config, groups, source, runs)
    report = {
        "schema_version": 1, "dry_run": dry_run, "output_dir": str(output),
        "groups_dir": config["groups_dir"], "grouping_id": manifest.get("grouping_id"),
        "groups_manifest_sha256": file_sha256(Path(config["groups_dir"]) / "manifest.json"),
        "source": {"path": str(data_path), "sha256": source_hash},
        "dataset": manifest.get("dataset"), "base_predictor": manifest.get("base_predictor"),
        "k": k, "nominal_coverage": config["nominal_coverage"],
        "policies": {
            "cace_definition": CACE_DEFINITION,
            "coverage": "saved_per_observation_coverage_on_available_predictions",
            "group_membership": "unchanged_shared_groups_joined_by_original_target_index",
            "empty_groups": "null_coverage_excluded_from_cace_and_wuc_denominator",
            "dataset_aggregation": "equal_weight_mean_of_per_sequence_cace_and_wuc",
            "sequences": "intersection_across_groups_and_all_runs_unless_explicitly_requested",
        },
        "sequences": keys, "skipped_sequences": skipped,
        "runs": [run["audit"] for run in runs], "alignment": [],
        "summary": [], "sequence_metrics": [], "group_metrics": [],
    }
    ys = {key: observed_sequence(source[key])[1] for key in keys}
    for run in runs:
        entry = run["entry"]
        rows = []
        for key in keys:
            y = ys[key]
            expected = None
            if run["config"] is not None:
                boundary = _method_boundary(len(y), entry["method"], run["config"], key)
                expected = np.arange(boundary, len(y), dtype=np.int64)
                if np.any(groups[key]["test_indices"] < boundary):
                    raise ValueError(
                        f"{entry['name']}/{key}: run's test split does not cover the shared groups."
                    )
            aligned = align_result(
                run["log"][key], config["nominal_coverage"], y,
                expected_test_indices=expected,
                allow_chronological_fallback=entry["allow_chronological_fallback"],
                quantile_pair=entry.get("quantile_pair"), context=f"{entry['name']}/{key}",
            )
            metrics = evaluate_sequence(groups[key], aligned, k, config["nominal_coverage"])
            identity = {
                "name": entry["name"], "method": entry["method"],
                "sequence_key": key, "sequence_key_type": type(key).__name__,
                "nominal_coverage": config["nominal_coverage"],
                "lower_quantile": aligned["quantile_pair"][0],
                "upper_quantile": aligned["quantile_pair"][1],
            }
            report["alignment"].append({
                **identity, "alignment": aligned["alignment"],
                "full_test_count": aligned["full_test_count"],
                "excluded_count": aligned["excluded_count"],
            })
            row = {**identity, "alignment": aligned["alignment"], **metrics["sequence"]}
            rows.append(row)
            report["sequence_metrics"].append(row)
            report["group_metrics"].extend({**identity, **group} for group in metrics["groups"])
        summary = {
            "name": entry["name"], "method": entry["method"],
            "dataset": manifest.get("dataset"), "base_predictor": manifest.get("base_predictor"),
            "k": k, "nominal_coverage": config["nominal_coverage"],
            **aggregate_sequences(rows),
        }
        report["summary"].append(summary)
        progress(f"{entry['name']}: {len(rows)} sequences, CACE={summary['cace']}, WUC={summary['wuc']}")
    if not dry_run:
        destination = _write_report(output, report, config)
        progress(f"Saved conditional coverage evaluation to {destination}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Saved-results evaluation YAML.")
    parser.add_argument("--output-dir", type=Path, help="New report directory; never overwritten.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and calculate without writing.")
    parser.add_argument("--allow-chronological-fallback", action="store_true",
                        help="Allow chronological alignment when saved target indices are absent.")
    args = parser.parse_args(argv)
    try:
        config = load_evaluation_config(args.config)
        if args.output_dir is not None:
            config["output_dir"] = str(args.output_dir.expanduser().resolve())
        if args.allow_chronological_fallback:
            config["allow_chronological_fallback"] = True
            for run in config["runs"]:
                run["allow_chronological_fallback"] = True
        report = evaluate_saved_runs(config, dry_run=args.dry_run,
                                     progress=(lambda _: None) if args.dry_run else print)
        if args.dry_run:
            print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return report
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
