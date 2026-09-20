"""Run one SPCI/base-predictor combination from a Slurm array."""

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
        "--output-root", type=Path, default=REPO_ROOT / "results" / "spci"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration without writing files or training.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    from omegaconf import OmegaConf
    from utils.experiment_config import load_experiment_config, resolve_job_saving_dir

    predictor = TASKS[args.task_id]
    artifact_dir, artifact_name = DATASET_ARTIFACTS[args.dataset]
    artifact_path = (
        REPO_ROOT / "data" / artifact_dir / predictor
        / f"{predictor}_{artifact_name}_data.pkl"
    )
    template_name = f"spci_{predictor}_{args.dataset}_config.yaml"
    template_path = REPO_ROOT / "configs" / "spci_configs" / template_name
    config = load_experiment_config(template_path)
    config.model.prediction_step = 1
    config.data.data_path = str(artifact_path)
    config.seed = args.seed
    output_dir = resolve_job_saving_dir(config, args.output_root)

    print(
        f"SPCI task {args.task_id}: dataset={args.dataset}, "
        f"base_predictor={predictor}, seed={args.seed}",
        flush=True,
    )
    print(f"Configuration template: {template_path}", flush=True)
    print(OmegaConf.to_yaml(config, resolve=True), flush=True)
    if args.dry_run:
        return

    if not artifact_path.is_file():
        parser.error(
            f"Prediction artifact does not exist: {artifact_path}. "
            "Run the corresponding base predictor before this SPCI job."
        )

    import random

    import numpy as np
    import torch

    from baselines.spci.run_spci import run_spci_experiment

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(config=config, f=config_path, resolve=True)
    print(f"Saved resolved configuration: {config_path}", flush=True)

    run_spci_experiment(str(config_path))


if __name__ == "__main__":
    main()
