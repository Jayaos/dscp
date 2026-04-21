import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.nexcp.run_nexcp import run_nexcp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the NexCP baseline from a YAML config file."
    )
    parser.add_argument(
        "config_path",
        help="Path to the NexCP YAML config file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_nexcp(args.config_path)


if __name__ == "__main__":
    main()
