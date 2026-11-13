"""Run CT-SSF on Chronos or saved LSTM forecasts, from a preset or YAML config."""

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
    parser.add_argument("--base-predictor", choices=("chronos", "lstm"), help="Preset predictor (default: chronos); must match an explicit YAML.")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--head-path", type=Path)
    parser.add_argument("--model-path", type=Path, help="LSTM checkpoint saved by --save-model.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", help="cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths/config without loading artifacts or writing files.")
    return parser


def resolve_config(args):
    if (args.config_path is None) == (args.dataset is None):
        raise ValueError("Provide either a config path or --dataset.")
    selected = args.base_predictor or "chronos"
    path = args.config_path or REPO_ROOT / "configs" / "ct_ssf_configs" / f"ct_ssf_{selected}_{args.dataset}_config.yaml"
    config = load_experiment_config(path)
    base_predictor = config.get("base_predictor", "chronos")
    if base_predictor not in {"chronos", "lstm"} or (args.base_predictor and args.base_predictor != base_predictor):
        raise ValueError("--base-predictor must match the config's base_predictor.")
    config.base_predictor = base_predictor
    checkpoint_field = "model_path" if base_predictor == "lstm" else "head_path"
    incompatible = "head_path" if base_predictor == "lstm" else "model_path"
    if getattr(args, incompatible) is not None:
        raise ValueError(f"--{incompatible.replace('_', '-')} is incompatible with {base_predictor}.")
    for field in ("seed", "device", "num_threads"):
        if getattr(args, field) is not None:
            config[field] = getattr(args, field)
    if not 0 <= config.seed < 2**32 or config.num_threads < 1:
        raise ValueError("seed must be in [0, 2**32) and num_threads must be positive.")
    for field in ("data_path", checkpoint_field):
        if getattr(args, field) is not None:
            config.data[field] = str(getattr(args, field))
    if args.output_dir is not None:
        config.saving_dir = str(args.output_dir)
    for node, field in ((config.data, "data_path"), (config.data, checkpoint_field), (config, "saving_dir")):
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
        checkpoint_field = "model_path" if config.base_predictor == "lstm" else "head_path"
        print(f"Checkpoint artifact available: {Path(config.data[checkpoint_field]).is_file()}")
        return config
    checkpoint_field = "model_path" if config.base_predictor == "lstm" else "head_path"
    for field in ("data_path", checkpoint_field):
        if not Path(config.data[field]).is_file():
            flag = "LSTM --save-model" if config.base_predictor == "lstm" else "Chronos --save-ctssf"
            parser.error(f"Missing {field}: {config.data[field]}. Run the matching {flag} job first.")
    from baselines.ct_ssf.run_ct_ssf import run_ct_ssf
    output = Path(config.saving_dir)
    output.mkdir(parents=True, exist_ok=True)
    launch = output / "launch_config.yaml"
    OmegaConf.save(config, launch, resolve=True)
    return run_ct_ssf(launch)


if __name__ == "__main__":
    main()
