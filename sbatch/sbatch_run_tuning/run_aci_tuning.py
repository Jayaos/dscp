"""Tune ACI using DSCP-QR's shared coverage-filter/Winkler-ranking workflow."""

import argparse
from pathlib import Path
import pickle
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from baselines.aci.run_aci import validate_config
from baselines.aci.tuning import tune_hyperparameters, validate_tuning_config
from utils.experiment_config import load_experiment_config


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be a nonnegative integer.")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--grid-config", required=True, type=Path)
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--sequence-key", default=None)
    parser.add_argument("--sequence-index", type=nonnegative_int, default=0)
    parser.add_argument("--top-k", type=positive_int, default=3)
    parser.add_argument("--seed", type=nonnegative_int, default=2026,
                        help="Recorded for parity with QR jobs; ACI is deterministic.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config and print paths without loading the artifact or writing files.")
    return parser


def resolve_inputs(base_config_path, grid_config_path, save_dir, *, sequence_key=None,
                   sequence_index=0, top_k=3):
    base_config_path = Path(base_config_path).expanduser().resolve()
    grid_config_path = Path(grid_config_path).expanduser().resolve()
    config = validate_config(load_experiment_config(base_config_path))
    grid_file = OmegaConf.to_container(load_experiment_config(grid_config_path), resolve=True)
    if not isinstance(grid_file, dict) or set(grid_file) - {"grid", "tuning"}:
        raise ValueError("Grid config must contain grid and optional tuning mappings.")
    grid = grid_file.get("grid")
    if not isinstance(grid, dict) or not grid or set(grid) - {"model.gamma", "model.window_size"}:
        raise ValueError("ACI grid supports model.gamma and model.window_size only; keep data splits and coverage fixed.")
    tuning = grid_file.get("tuning", {})
    if not isinstance(tuning, dict):
        raise ValueError("Grid tuning settings must be a mapping.")
    tuning = {**tuning, "sequence_key": sequence_key,
              "sequence_index": sequence_index, "top_k": top_k}
    for parameter, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid {parameter} must be a nonempty list.")
        tuning[parameter.split(".")[1] + "_grid"] = values
    config["tuning"] = tuning
    config = validate_tuning_config(config)
    artifact = Path(config["data"]["data_path"]).expanduser()
    config["data"]["data_path"] = str((artifact if artifact.is_absolute() else REPO_ROOT / artifact).resolve())
    save_dir = Path(save_dir).expanduser().resolve()
    config["saving_dir"] = str(save_dir / "final_run")
    return config, base_config_path, grid_config_path, save_dir


def run_tuning(base_config_path, grid_config_path, save_dir, *, sequence_key=None,
               sequence_index=0, top_k=3, seed=2026, dry_run=False):
    config, base_path, grid_path, output = resolve_inputs(
        base_config_path, grid_config_path, save_dir,
        sequence_key=sequence_key, sequence_index=sequence_index, top_k=top_k,
    )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    print(f"[aci tuning] base config: {base_path}")
    print(f"[aci tuning] artifact: {config['data']['data_path']}")
    print(f"[aci tuning] output: {output}")
    print("[aci tuning] Shared configuration across selected sequences; independent online states.")
    print("[aci tuning] Coverage includes full/empty sets; Winkler uses finite nonempty intervals.")
    print("[aci tuning] ACI is deterministic; seed is recorded but does not affect candidates.")
    if dry_run:
        print(f"Artifact available: {Path(config['data']['data_path']).is_file()}")
        print(OmegaConf.to_yaml(OmegaConf.create(config)))
        return config
    with Path(config["data"]["data_path"]).open("rb") as stream:
        data = pickle.load(stream)
    payload = tune_hyperparameters(data, config)
    payload.update({"base_config_path": str(base_path), "grid_config_path": str(grid_path),
                    "seed": seed, "execution": {"mode": "serial", "deterministic": True}})
    output.mkdir(parents=True, exist_ok=True)
    for trial in payload["all_trials"]:
        folder = output / f"trial_{trial['trial_index']:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        trial["resolved_config"]["saving_dir"] = str(folder / "final_run")
        OmegaConf.save(OmegaConf.create(trial["resolved_config"]), folder / "resolved_config.yaml")
        with (folder / "result.pkl").open("wb") as stream:
            pickle.dump(trial, stream, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[aci tuning] trial {trial['trial_index']:04d}: {trial['grid_values']} "
              f"eligible={trial['eligible']} Winkler={trial['result']['selection_score']:.6g}")
    with (output / "tuning_results.pkl").open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[aci tuning] {payload['num_eligible_trials']}/{payload['num_trials']} eligible trials.")
    print(payload["selection_message"])
    for rank, trial in enumerate(payload["top_trials"], 1):
        print(f"[aci tuning] rank {rank}: {output / ('trial_%04d' % trial['trial_index']) / 'resolved_config.yaml'}")
    return payload


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_tuning(args.base_config, args.grid_config, args.save_dir,
                          sequence_key=args.sequence_key, sequence_index=args.sequence_index,
                          top_k=args.top_k, seed=args.seed, dry_run=args.dry_run)
    except (ValueError, TypeError, OSError, OmegaConfBaseException) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
