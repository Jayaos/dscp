"""Run one Local-CP encoder/base-predictor combination from a Slurm array."""

import argparse
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = tuple(
    (encoder, predictor)
    for encoder in ("rnn", "transformer")
    for predictor in ("lr", "lstm", "chronos")
)
DATASET_ARTIFACTS = {
    "air": ("air-10_prediction", "air-10"),
    "solar": ("solar_prediction", "nsdb-60m"),
    "sapflux": ("sapflux-solo3-large", "sapflux-solo3-large"),
}


def positive_bandwidth(value):
    try:
        bandwidth = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("bandwidth must be finite and positive") from exc
    if isinstance(value, bool) or not math.isfinite(bandwidth) or bandwidth <= 0:
        raise argparse.ArgumentTypeError("bandwidth must be finite and positive")
    return bandwidth


def positive_workers(value):
    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("cpu-workers must be a positive integer") from exc
    if workers < 1:
        raise argparse.ArgumentTypeError("cpu-workers must be a positive integer")
    return workers


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=tuple(DATASET_ARTIFACTS))
    parser.add_argument(
        "--task-id",
        type=int,
        choices=range(len(TASKS)),
        required=True,
        help="0: RNN/LR, 1: RNN/LSTM, 2: RNN/Chronos, "
        "3: Transformer/LR, 4: Transformer/LSTM, 5: Transformer/Chronos",
    )
    parser.add_argument(
        "--kernel",
        choices=("softmax", "epanechnikov"),
        default=None,
        help="Select the matching kernel templates and override model.kernel; "
        "the default without a YAML setting is softmax.",
    )
    parser.add_argument(
        "--bandwidth",
        type=positive_bandwidth,
        default=None,
        help="Override model.bandwidth from the Epanechnikov template with a "
        "positive chord-distance bandwidth.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Override the results root; Epanechnikov defaults separate bandwidths and seeds.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--cpu-workers", type=positive_workers, default=None,
        help="Run independent sequences on CPU with this many single-threaded workers.",
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

    encoder, predictor = TASKS[args.task_id]
    artifact_dir, artifact_name = DATASET_ARTIFACTS[args.dataset]
    artifact_path = (
        REPO_ROOT
        / "data"
        / artifact_dir
        / predictor
        / f"{predictor}_{artifact_name}_data.pkl"
    )
    if args.kernel == "epanechnikov":
        template_dir = "lcp_epanechnikov_kernel_configs"
        template_name = f"lcp_{encoder}_epanechnikov_{predictor}_{args.dataset}_config.yaml"
    else:
        template_dir = "lcp_configs"
        template_name = f"lcp_{encoder}_{predictor}_{args.dataset}_config.yaml"
    template_path = REPO_ROOT / "configs" / template_dir / template_name
    config = load_experiment_config(template_path)
    config.base_predictor = predictor
    kernel = args.kernel if args.kernel is not None else config.model.get("kernel", "softmax")
    if kernel not in ("softmax", "epanechnikov"):
        parser.error(f"Invalid model.kernel in {template_path}: {kernel!r}.")
    config.model.kernel = kernel
    if kernel == "epanechnikov":
        bandwidth = args.bandwidth if args.bandwidth is not None else config.model.get("bandwidth")
        if bandwidth is None:
            parser.error("Epanechnikov requires --bandwidth or model.bandwidth in the template.")
        try:
            bandwidth = positive_bandwidth(bandwidth)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        config.model.bandwidth = bandwidth
        config.model.similarity_fn = "cos_similarity"
    elif args.bandwidth is not None:
        parser.error("--bandwidth applies only to --kernel epanechnikov.")
    config.model.prediction_step = 1
    config.data.data_path = str(artifact_path)
    config.device = config.get("device", 0)
    if args.cpu_workers is not None:
        config.device = "cpu"
        config.cpu_workers = args.cpu_workers
    config.seed = args.seed
    output_root = args.output_root
    if output_root is None:
        method = "local_cp_cpu" if args.cpu_workers is not None else "local_cp"
        output_root = REPO_ROOT / "results" / method
        if kernel == "epanechnikov":
            method = "local_cp_epanechnikov_cpu" if args.cpu_workers is not None else "local_cp_epanechnikov"
            output_root = (
                REPO_ROOT / "results" / method
                / f"bandwidth_{bandwidth}" / f"seed_{args.seed}"
            )
    output_dir = resolve_job_saving_dir(config, output_root)

    print(
        f"Local-CP task {args.task_id}: dataset={args.dataset}, "
        f"base_predictor={predictor}, encoder={encoder}, kernel={kernel}, seed={args.seed}",
        flush=True,
    )
    print(f"Configuration template: {template_path}", flush=True)
    print(OmegaConf.to_yaml(config, resolve=True), flush=True)
    if args.dry_run:
        return

    if not artifact_path.is_file():
        parser.error(
            f"Prediction artifact does not exist: {artifact_path}. "
            "Run the corresponding base predictor before this Local-CP job."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(config=config, f=config_path, resolve=True)
    print(f"Saved resolved configuration: {config_path}", flush=True)

    if args.cpu_workers is not None:
        from dscp.local_cp_parallel import run_cpu_local_cp

        run_cpu_local_cp(str(config_path), encoder=encoder, workers=args.cpu_workers)
        return

    import random

    import numpy as np
    import torch

    from dscp.run_local_cp import run_rnn_local_cp, run_transformer_local_cp

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    runner = run_rnn_local_cp if encoder == "rnn" else run_transformer_local_cp
    runner(str(config_path))


if __name__ == "__main__":
    main()
