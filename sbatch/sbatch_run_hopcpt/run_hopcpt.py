import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.hopcpt.run_hopcpt import run_hopcpt, run_hopcpt_sequence_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the HOPCPT baseline from a YAML config file."
    )
    parser.add_argument(
        "config_path",
        help="Path to the HOPCPT YAML config file.",
    )
    parser.add_argument(
        "--sequence-batch",
        action="store_true",
        help="Train one HopCPT model with independent sequences stacked as the batch axis.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.sequence_batch:
        run_hopcpt_sequence_batch(args.config_path)
    else:
        run_hopcpt(args.config_path)


if __name__ == "__main__":
    main()
