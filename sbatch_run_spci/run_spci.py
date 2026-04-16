import argparse
from baselines.spci.run_spci import run_spci_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SPCI baseline from a YAML config file."
    )
    parser.add_argument(
        "config_path",
        help="Path to the SPCI YAML config file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_spci_experiment(args.config_path)


if __name__ == "__main__":
    main()
