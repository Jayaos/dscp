import argparse

from dscp.run_local_cp import run_rnn_local_cp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local-CP RNN model from a YAML config file."
    )
    parser.add_argument(
        "config_path",
        help="Path to the local-CP RNN YAML config file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_rnn_local_cp(args.config_path)


if __name__ == "__main__":
    main()
