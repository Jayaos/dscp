import argparse
from baselines.kowcpi.run_kowcpi import run_kowcpi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the KOWCPI baseline from a YAML config file."
    )
    parser.add_argument(
        "config_path",
        help="Path to the KOWCPI YAML config file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_kowcpi(args.config_path)


if __name__ == "__main__":
    main()
