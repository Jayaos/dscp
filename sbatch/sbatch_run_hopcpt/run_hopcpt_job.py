"""Run one HopCPT/base-predictor combination from a Slurm array."""

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = ("lr", "lstm", "chronos")
DATASET_ARTIFACTS = {
    "air": ("air-10_prediction", "air-10"),
    "solar": ("solar_prediction", "nsdb-60m"),
    "sapflux": ("sapflux-solo3-large", "sapflux-solo3-large"),
}


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


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
        "--output-root", type=Path, default=REPO_ROOT / "results" / "hopcpt"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--num-gpus",
        type=_positive_int,
        default=1,
        help="Number of visible GPUs to use for independent sequence workers.",
    )
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
    template_name = f"hopcpt_{predictor}_{args.dataset}_config.yaml"
    template_path = REPO_ROOT / "configs" / "hopcpt_configs" / template_name
    config = load_experiment_config(template_path)
    config.model.prediction_step = 1
    config.data.data_path = str(artifact_path)
    # CUDA indices are local to the GPUs made visible to this Slurm task.
    config.device = 0
    config.parallel.enabled = args.num_gpus > 1
    config.parallel.devices = list(range(args.num_gpus))
    cpu_budget = int(os.environ.get("SLURM_CPUS_PER_TASK", args.num_gpus))
    config.parallel.threads_per_worker = max(1, cpu_budget // args.num_gpus)
    config.seed = args.seed
    output_dir = resolve_job_saving_dir(config, args.output_root)

    print(
        f"HopCPT task {args.task_id}: dataset={args.dataset}, "
        f"base_predictor={predictor}, seed={args.seed}, num_gpus={args.num_gpus}",
        flush=True,
    )
    print(f"Configuration template: {template_path}", flush=True)
    print(OmegaConf.to_yaml(config, resolve=True), flush=True)
    if args.dry_run:
        return

    if not artifact_path.is_file():
        parser.error(
            f"Prediction artifact does not exist: {artifact_path}. "
            "Run the corresponding base predictor before this HopCPT job."
        )

    # Workers inherit these limits when spawned. Set them before loading NumPy/Torch.
    os.environ["OMP_NUM_THREADS"] = str(config.parallel.threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(config.parallel.threads_per_worker)

    import torch

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if available_gpus < args.num_gpus:
        parser.error(
            f"Requested {args.num_gpus} GPUs, but only {available_gpus} CUDA GPUs "
            "are visible. Match --num-gpus to the Slurm GPU allocation."
        )

    import random

    import numpy as np

    from baselines.hopcpt.run_hopcpt import run_hopcpt

    torch.set_num_threads(config.parallel.threads_per_worker)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(config=config, f=config_path, resolve=True)
    print(f"Saved resolved configuration: {config_path}", flush=True)

    run_hopcpt(str(config_path))


if __name__ == "__main__":
    main()
