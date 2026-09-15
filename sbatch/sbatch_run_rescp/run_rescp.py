"""Run a ResCP configuration or one of the dataset/predictor presets."""

import argparse
from numbers import Integral
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf


DATASETS = ("air", "solar", "sapflux")
BASE_PREDICTORS = ("lr", "lstm", "chronos")


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def _seed(value):
    parsed = int(value)
    if not 0 <= parsed < 2**32:
        raise argparse.ArgumentTypeError("Seed must be an integer in [0, 2**32).")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run residual-only ResCP on saved base-predictor forecasts."
    )
    parser.add_argument("config_path", nargs="?", type=Path, help="Experiment YAML file.")
    parser.add_argument("--dataset", choices=DATASETS, help="Choose a checked-in preset.")
    parser.add_argument("--base-predictor", choices=BASE_PREDICTORS)
    parser.add_argument("--num-cores", "--num_cores", type=_positive_int, default=1)
    parser.add_argument("--seed", type=_seed, default=None, help="Override the config seed.")
    parser.add_argument("--data-path", type=Path, help="Override the saved forecast artifact.")
    parser.add_argument("--output-dir", type=Path, help="Override the experiment output directory.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the resolved config and artifact availability without evaluating or writing files.",
    )
    return parser


def resolve_config(args):
    """Load a standalone preset and apply overrides before resolving interpolation."""
    has_preset = args.dataset is not None or args.base_predictor is not None
    if args.config_path is not None and has_preset:
        raise ValueError("Use a config path or --dataset with --base-predictor.")
    if args.config_path is None:
        if args.dataset is None or args.base_predictor is None:
            raise ValueError("Provide a config path, or both --dataset and --base-predictor.")
        config_path = (
            REPO_ROOT / "configs" / "rescp_configs"
            / "rescp_{}_{}_config.yaml".format(args.base_predictor, args.dataset)
        )
    else:
        config_path = args.config_path.resolve()
    config = OmegaConf.load(config_path)
    if args.seed is not None:
        config.seed = args.seed
    seed = config.get("seed", 2026)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    config.seed = int(seed)
    if args.data_path is not None:
        config.data.data_path = str(args.data_path)
    if args.output_dir is not None:
        config.saving_dir = str(args.output_dir)

    for node, field in ((config.data, "data_path"), (config, "saving_dir")):
        path = Path(str(node[field])).expanduser()
        node[field] = str(path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve())
    return config_path, config


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config_path, config = resolve_config(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    artifact_path = Path(config.data.data_path)
    if args.dry_run:
        print("Configuration: {}".format(config_path))
        print("Artifact available: {}".format(artifact_path.is_file()))
        print("Sequence workers: {}".format(args.num_cores))
        print(OmegaConf.to_yaml(config, resolve=True))
        return config
    if not artifact_path.is_file():
        parser.error("Saved forecast artifact does not exist: {}".format(artifact_path))

    from baselines.rescp.run_rescp import run_rescp

    output_dir = Path(config.saving_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    launch_config_path = output_dir / "launch_config.yaml"
    OmegaConf.save(config=config, f=launch_config_path, resolve=True)
    return run_rescp(launch_config_path, num_cores=args.num_cores)


if __name__ == "__main__":
    main()
