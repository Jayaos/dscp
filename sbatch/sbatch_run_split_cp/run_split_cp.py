"""Run a SplitCP configuration or one of the dataset/predictor presets."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from utils.experiment_config import load_experiment_config


DATASETS = ("air", "solar", "sapflux")
BASE_PREDICTORS = ("lr", "lstm", "chronos")


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run fixed-calibration SplitCP on saved base-predictor forecasts."
    )
    parser.add_argument("config_path", nargs="?", type=Path, help="Experiment YAML file.")
    parser.add_argument("--dataset", choices=DATASETS, help="Choose a checked-in preset.")
    parser.add_argument("--base-predictor", choices=BASE_PREDICTORS)
    parser.add_argument(
        "--num-cores", "--num_cores", type=_positive_int, default=None,
        help="Override the number of sequence workers configured in YAML.",
    )
    parser.add_argument("--data-path", type=Path, help="Override the saved forecast artifact.")
    parser.add_argument("--output-dir", type=Path, help="Override the experiment output directory.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and display the resolved config without evaluating or writing files.",
    )
    return parser


def resolve_config(args):
    """Load a preset/config, apply overrides, and resolve artifact/output paths."""
    has_preset = args.dataset is not None or args.base_predictor is not None
    if args.config_path is not None and has_preset:
        raise ValueError("Use a config path or --dataset with --base-predictor.")
    if args.config_path is None:
        if args.dataset is None or args.base_predictor is None:
            raise ValueError("Provide a config path, or both --dataset and --base-predictor.")
        config_path = (
            REPO_ROOT / "configs" / "split_cp_configs"
            / "split_cp_{}_{}_config.yaml".format(args.base_predictor, args.dataset)
        )
    else:
        config_path = args.config_path.expanduser().resolve()
    config = load_experiment_config(config_path)
    if args.num_cores is not None:
        config.num_cores = args.num_cores
    if args.data_path is not None:
        config.data.data_path = str(args.data_path)
    if args.output_dir is not None:
        config.saving_dir = str(args.output_dir)

    for node, field in ((config.data, "data_path"), (config, "saving_dir")):
        path = Path(str(node[field])).expanduser()
        node[field] = str(path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve())

    from baselines.split_cp.run_split_cp import validate_config

    return config_path, OmegaConf.create(validate_config(config))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config_path, config = resolve_config(args)
    except (ValueError, TypeError, OSError, OmegaConfBaseException) as exc:
        parser.error(str(exc))

    artifact_path = Path(config.data.data_path)
    if args.dry_run:
        print("Configuration: {}".format(config_path))
        print("Artifact available: {}".format(artifact_path.is_file()))
        print("Sequence workers: {}".format(config.num_cores))
        print(OmegaConf.to_yaml(config, resolve=True))
        return config
    if not artifact_path.is_file():
        parser.error("Saved forecast artifact does not exist: {}".format(artifact_path))

    from baselines.split_cp.run_split_cp import run_split_cp

    return run_split_cp(config)


if __name__ == "__main__":
    main()
