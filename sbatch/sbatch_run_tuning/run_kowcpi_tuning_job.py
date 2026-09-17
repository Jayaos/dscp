"""Dispatch one dataset/base-predictor KOWCPI tuning job from a Slurm array."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sbatch.sbatch_run_kowcpi.run_kowcpi_job import DATASET_ARTIFACTS, TASKS


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return parsed


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Must be a nonnegative integer.")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=tuple(DATASET_ARTIFACTS))
    parser.add_argument(
        "--task-id", type=int, choices=range(len(TASKS)), required=True,
        help="0: LR, 1: LSTM, 2: Chronos",
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "results" / "tuning" / "kowcpi",
    )
    parser.add_argument("--base-config", type=Path, help="Override the dataset's base template.")
    parser.add_argument("--grid-config", type=Path, help="Override the dataset's tuning YAML.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-cores", type=_positive_int, default=1)
    parser.add_argument("--top-k", type=_positive_int, default=3)
    parser.add_argument("--sequence-key", default=None)
    parser.add_argument("--sequence-index", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print resolved paths/configuration and command without writing or running tuning.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    from omegaconf import OmegaConf

    predictor = TASKS[args.task_id]
    artifact_dir, artifact_name = DATASET_ARTIFACTS[args.dataset]
    artifact_path = (
        REPO_ROOT / "data" / artifact_dir / predictor
        / f"{predictor}_{artifact_name}_data.pkl"
    )
    config_dir = REPO_ROOT / "configs" / "kowcpi_configs"
    template_name = (
        "kowcpi_lstm_sapflux_config.yaml"
        if args.dataset == "sapflux"
        else "kowcpi_chronos_air_config.yaml"
    )
    template_path = (args.base_config or config_dir / template_name).expanduser().resolve()
    grid_path = (
        args.grid_config or config_dir / f"kowcpi_{args.dataset}_tuning_config.yaml"
    ).expanduser().resolve()
    for path in (template_path, grid_path):
        if not path.is_file():
            parser.error(f"Configuration does not exist: {path}")

    output_dir = args.output_root.expanduser().resolve() / args.dataset / predictor
    config_path = output_dir / "resolved_base_config.yaml"
    config = OmegaConf.load(template_path)
    config.data.data_path = str(artifact_path)
    config.model.prediction_step = 1
    config.seed = args.seed
    # A selected trial config can also be used by the normal evaluation launcher.
    config.saving_dir = str(REPO_ROOT / "results" / "kowcpi" / args.dataset / predictor)

    command = [
        sys.executable, "-u", "-m", "sbatch.sbatch_run_tuning.run_kowcpi_tuning",
        "--base-config", str(config_path),
        "--grid-config", str(grid_path),
        "--save-dir", str(output_dir),
        "--seed", str(args.seed),
        "--num-cores", str(args.num_cores),
        "--top-k", str(args.top_k),
        "--sequence-index", str(args.sequence_index),
    ]
    if args.sequence_key is not None:
        command.extend(["--sequence-key", args.sequence_key])

    print(
        f"KOWCPI tuning task {args.task_id}: dataset={args.dataset}, "
        f"base_predictor={predictor}, seed={args.seed}, num_cores={args.num_cores}",
        flush=True,
    )
    print(f"Base template: {template_path}", flush=True)
    print(f"Tuning grid: {grid_path}", flush=True)
    print(f"Tuning output: {output_dir}", flush=True)
    print(OmegaConf.to_yaml(config, resolve=True), flush=True)
    print(f"Command: {shlex.join(command)}", flush=True)
    if args.dry_run:
        return

    if not artifact_path.is_file():
        parser.error(
            f"Prediction artifact does not exist: {artifact_path}. "
            "Run the corresponding base predictor before this tuning job."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=config_path, resolve=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
