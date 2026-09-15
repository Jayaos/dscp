"""Run DistMatch on saved forecasts; help and dry runs avoid numerical imports."""

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
        description="Run residual-only DistMatch on saved base-predictor forecasts."
    )
    parser.add_argument("config_path", type=Path, help="Experiment YAML file.")
    parser.add_argument(
        "--num-cores", "--num_cores", type=_positive_int, default=None,
        help="Override parallel sequence workers; omitted uses config.num_cores.",
    )
    parser.add_argument(
        "--threads-per-worker", type=_positive_int, default=None,
        help="Override the CPU thread limit inside each sequence worker.",
    )
    parser.add_argument("--seed", type=_seed, default=None, help="Override the config seed.")
    parser.add_argument("--data-path", type=Path, help="Override the saved forecast artifact.")
    parser.add_argument("--output-dir", type=Path, help="Override the experiment output directory.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and artifact paths, then print the plan without evaluating or writing files.",
    )
    return parser


def _resolve_path(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty filesystem path.")
    path = Path(value).expanduser()
    return str(path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve())


def validate_cpu_allocation(config, allocated_cpus):
    """Reject oversubscription of a Slurm allocation without changing the config."""
    if allocated_cpus is None:
        return
    try:
        allocation = int(allocated_cpus)
    except (TypeError, ValueError) as exc:
        raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer.") from exc
    if allocation < 1:
        raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer.")
    requested = config.num_cores * config.threads_per_worker
    if requested > allocation:
        raise ValueError(
            f"Configuration requests {requested} CPUs "
            f"({config.num_cores} workers * {config.threads_per_worker} threads), "
            f"but SLURM_CPUS_PER_TASK={allocation}. "
            f"Submit with sbatch --cpus-per-task={requested} or change "
            "num_cores/threads_per_worker in the configuration."
        )


def resolve_config(args):
    from omegaconf import OmegaConf

    config_path = args.config_path.expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"Experiment configuration does not exist: {config_path}")
    config = OmegaConf.load(config_path)
    if not OmegaConf.is_dict(config):
        raise ValueError("Experiment configuration must be a YAML mapping.")
    for field in ("seed", "num_cores", "threads_per_worker"):
        value = getattr(args, field)
        if value is not None:
            config[field] = value
    if args.data_path is not None:
        OmegaConf.update(config, "data.data_path", str(args.data_path))
    if args.output_dir is not None:
        config.saving_dir = str(args.output_dir)
    from baselines.distmatch.config import validate_config

    config = OmegaConf.create(validate_config(config))
    config.data.data_path = _resolve_path(config.data.get("data_path"), "data.data_path")
    config.saving_dir = _resolve_path(config.get("saving_dir"), "saving_dir")
    if config.matching.get("cache_dir") is not None:
        config.matching.cache_dir = _resolve_path(config.matching.cache_dir, "matching.cache_dir")
    if Path(config.saving_dir).is_file():
        raise ValueError(f"saving_dir points to an existing file: {config.saving_dir}")
    if config.matching.get("cache_dir") is not None and Path(config.matching.cache_dir).is_file():
        raise ValueError(f"matching.cache_dir points to an existing file: {config.matching.cache_dir}")
    return config_path, config


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from omegaconf import OmegaConf
        from omegaconf.errors import OmegaConfBaseException
    except ImportError as exc:
        parser.error(f"DistMatch configuration requires OmegaConf; install envs/env-distmatch.yml. {exc}")
    try:
        config_path, config = resolve_config(args)
        validate_cpu_allocation(config, os.environ.get("SLURM_CPUS_PER_TASK"))
        artifact_path = Path(config.data.data_path)
        if not artifact_path.is_file():
            raise ValueError(f"Saved forecast artifact does not exist: {artifact_path}")
    except (ValueError, OSError, OmegaConfBaseException) as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(f"Configuration: {config_path}")
        print(f"Artifact available: {artifact_path}")
        print(f"Sequence workers: {config.num_cores}")
        print(f"Threads per worker: {config.threads_per_worker}")
        print(OmegaConf.to_yaml(config, resolve=True))
        return config

    from baselines.distmatch.run_distmatch import run_distmatch

    output_dir = Path(config.saving_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    launch_config_path = output_dir / "launch_config.yaml"
    OmegaConf.save(config=config, f=launch_config_path, resolve=True)
    return run_distmatch(launch_config_path, num_cores=args.num_cores)


if __name__ == "__main__":
    main()
