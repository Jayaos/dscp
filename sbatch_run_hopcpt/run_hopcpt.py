import argparse
from baselines.hopcpt.run_hopcpt import run_hopcpt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the HOPCPT baseline from a YAML config file."
    )
    parser.add_argument(
        "config_path",
        help="Path to the HOPCPT YAML config file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_hopcpt(args.config_path)


if __name__ == "__main__":
    main()
