"""Run the CT-SSF Chronos adaptation from a preset or an explicit YAML config."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf
from utils.experiment_config import load_experiment_config


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", nargs="?", type=Path)
    parser.add_argument("--dataset", choices=("air", "solar", "sapflux"))
    parser.add_argument("--base-predictor", choices=("chronos",), default="chronos")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--head-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", help="cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths/config without loading artifacts or writing files.")
    return parser


def resolve_config(args):
    if (args.config_path is None) == (args.dataset is None):
        raise ValueError("Provide either a config path or --dataset.")
    path = args.config_path or REPO_ROOT / "configs" / "ct_ssf_configs" / f"ct_ssf_chronos_{args.dataset}_config.yaml"
    config = load_experiment_config(path)
    for field in ("seed", "device", "num_threads"):
        if getattr(args, field) is not None:
            config[field] = getattr(args, field)
    if not 0 <= config.seed < 2**32 or config.num_threads < 1:
        raise ValueError("seed must be in [0, 2**32) and num_threads must be positive.")
    for field in ("data_path", "head_path"):
        if getattr(args, field) is not None:
            config.data[field] = str(getattr(args, field))
    if args.output_dir is not None:
        config.saving_dir = str(args.output_dir)
    for node, field in ((config.data, "data_path"), (config.data, "head_path"), (config, "saving_dir")):
        value = Path(str(node[field])).expanduser()
        node[field] = str(value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve())
    return config


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = resolve_config(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        print(f"Forecast artifact available: {Path(config.data.data_path).is_file()}")
        print(f"Head artifact available: {Path(config.data.head_path).is_file()}")
        return config
    for field in ("data_path", "head_path"):
        if not Path(config.data[field]).is_file():
            parser.error(f"Missing {field}: {config.data[field]}. Run the matching Chronos --save-ctssf job first.")
    from baselines.ct_ssf.run_ct_ssf import run_ct_ssf
    output = Path(config.saving_dir)
    output.mkdir(parents=True, exist_ok=True)
    launch = output / "launch_config.yaml"
    OmegaConf.save(config, launch, resolve=True)
    return run_ct_ssf(launch)


if __name__ == "__main__":
    main()
