"""Run one KOWCPI/base-predictor combination from a Slurm array."""

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = ("lr", "lstm", "chronos")
DATASET_ARTIFACTS = {
    "air": ("air-10_prediction", "air-10"),
    "solar": ("solar_prediction", "nsdb-60m"),
    "sapflux": ("sapflux-solo3-large", "sapflux-solo3-large"),
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=tuple(DATASET_ARTIFACTS))
    parser.add_argument(
        "--task-id",
        type=int,
        choices=range(len(TASKS)),
        required=True,
        help="0: LR, 1: LSTM, 2: Chronos",
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "results" / "kowcpi"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--num-cores",
        type=int,
        default=1,
        help="Number of independent sequences to process in parallel.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration without writing files or running KOWCPI.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_cores < 1:
        parser.error("--num-cores must be a positive integer.")

    from omegaconf import OmegaConf

    predictor = TASKS[args.task_id]
    artifact_dir, artifact_name = DATASET_ARTIFACTS[args.dataset]
    artifact_path = (
        REPO_ROOT / "data" / artifact_dir / predictor
        / f"{predictor}_{artifact_name}_data.pkl"
    )
    template_name = (
        "kowcpi_lstm_sapflux_config.yaml"
        if args.dataset == "sapflux"
        else "kowcpi_chronos_air_config.yaml"
    )
    template_path = REPO_ROOT / "configs" / "kowcpi_configs" / template_name
    output_dir = args.output_root.expanduser().resolve() / args.dataset / predictor

    config = OmegaConf.load(template_path)
    config.model.prediction_step = 1
    config.data.data_path = str(artifact_path)
    config.saving_dir = str(output_dir)
    config.seed = args.seed

    print(
        f"KOWCPI task {args.task_id}: dataset={args.dataset}, "
        f"base_predictor={predictor}, seed={args.seed}, num_cores={args.num_cores}",
        flush=True,
    )
    print(f"Configuration template: {template_path}", flush=True)
    print(OmegaConf.to_yaml(config, resolve=True), flush=True)
    if args.dry_run:
        return

    if not artifact_path.is_file():
        parser.error(
            f"Prediction artifact does not exist: {artifact_path}. "
            "Run the corresponding base predictor before this KOWCPI job."
        )

    import random

    import numpy as np
    import torch

    from baselines.kowcpi.run_kowcpi import run_kowcpi

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(config=config, f=config_path, resolve=True)
    print(f"Saved resolved configuration: {config_path}", flush=True)

    run_kowcpi(str(config_path), num_cores=args.num_cores)


if __name__ == "__main__":
    main()
