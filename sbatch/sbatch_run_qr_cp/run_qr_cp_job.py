"""Run one QR-CP encoder/base-predictor combination from a Slurm array."""

import argparse
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
        "--head-type",
        choices=("nondecreasing", "independent"),
        default="nondecreasing",
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "results" / "qr_cp"
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

    encoder, predictor = TASKS[args.task_id]
    artifact_dir, artifact_name = DATASET_ARTIFACTS[args.dataset]
    artifact_path = (
        REPO_ROOT / "data" / artifact_dir / predictor
        / f"{predictor}_{artifact_name}_data.pkl"
    )
    if args.dataset == "sapflux":
        template_name = f"qr_{encoder}_lstm_sapflux_config.yaml"
    else:
        template_name = f"qr_{encoder}_{predictor}_air_config.yaml"
    template_path = REPO_ROOT / "configs" / "qr_cp_configs" / template_name
    output_dir = (
        args.output_root.expanduser().resolve()
        / args.dataset / predictor / encoder / args.head_type
    )

    config = OmegaConf.load(template_path)
    config.base_predictor = predictor
    config.model.head_type = args.head_type
    config.model.prediction_step = 1
    config.data.data_path = str(artifact_path)
    config.saving_dir = str(output_dir)
    config.device = config.get("device", 0)
    config.seed = args.seed

    print(
        f"QR-CP task {args.task_id}: dataset={args.dataset}, "
        f"base_predictor={predictor}, encoder={encoder}, "
        f"head_type={args.head_type}, seed={args.seed}",
        flush=True,
    )
    print(f"Configuration template: {template_path}", flush=True)
    print(OmegaConf.to_yaml(config, resolve=True), flush=True)
    if args.dry_run:
        return

    if not artifact_path.is_file():
        parser.error(
            f"Prediction artifact does not exist: {artifact_path}. "
            "Run the corresponding base predictor before this QR-CP job."
        )

    import random

    import numpy as np
    import torch

    from dscp.run_qr_cp import (
        run_rnn_quantile_regression_cp,
        run_transformer_quantile_regression_cp,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(config=config, f=config_path, resolve=True)
    print(f"Saved resolved configuration: {config_path}", flush=True)

    runner = (
        run_rnn_quantile_regression_cp
        if encoder == "rnn"
        else run_transformer_quantile_regression_cp
    )
    runner(str(config_path))


if __name__ == "__main__":
    main()
