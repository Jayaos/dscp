import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the KOWCPI baseline from a YAML config file."
    )
    parser.add_argument(
        "config_path",
        help="Path to the KOWCPI YAML config file.",
    )
    parser.add_argument(
        "--num-cores",
        "--num_cores",
        type=int,
        default=1,
        help="Number of independent sequences to process in parallel.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from baselines.kowcpi.run_kowcpi import run_kowcpi

    run_kowcpi(args.config_path, num_cores=args.num_cores)


if __name__ == "__main__":
    main()
