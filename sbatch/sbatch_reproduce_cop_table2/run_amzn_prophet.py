"""Reproduce COP Table 2 on the authors' saved AMZN Prophet predictions."""

import argparse
import json
from pathlib import Path
import sys

# This entrypoint only reports to the console, including when invoked directly.
sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce ACI, LQT, and COP on AMZN / Prophet using the COP "
            "paper Table 2 protocol. Results are printed to the run log only."
        )
    )
    parser.add_argument(
        "--data-path", type=Path, default=Path("data/amzn_prophet.npz"),
        help="Saved Prophet forecasts; relative paths resolve from the repository root.",
    )
    parser.add_argument(
        "--targets-path", type=Path, default=Path("data/amzn_prophet_targets.csv"),
        help="Aligned AMZN observations; relative paths resolve from the repository root.",
    )
    parser.add_argument(
        "--method", choices=("all", "aci", "lqt", "cop"), default="all",
        help="Run all three methods (default), or a single method.",
    )
    parser.add_argument(
        "--lqt-order", type=int, choices=(1, 2), default=1,
        help="LQT order: 1 is the empirically recovered Table 2 setup (default); 2 is the released AMZN YAML.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load and validate the inputs and print metadata without running methods.",
    )
    return parser


def resolve_inputs(args):
    paths = []
    for label, value in (("Forecast", args.data_path), ("Target", args.targets_path)):
        path = value.expanduser()
        path = (path if path.is_absolute() else REPO_ROOT / path).resolve()
        if not path.is_file():
            raise ValueError("{} input file does not exist: {}".format(label, path))
        paths.append(path)
    return tuple(paths)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        data_path, targets_path = resolve_inputs(args)
        methods = None if args.method == "all" else (args.method,)
        print("AMZN / Prophet: COP paper Table 2 reproduction")
        print("Forecast input: {}".format(data_path))
        print("Target input: {}".format(targets_path))
        print("Methods: {}".format("ACI, LQT, COP" if methods is None else args.method.upper()))
        print("LQT order: {} (bias=5)".format(args.lqt_order))
        print("Results are printed to this log only.")

        if args.dry_run:
            from baselines.cop_table2.run_reproduction import load_amzn_data

            data = load_amzn_data(data_path, targets_path)
            print("Validated observations: {}".format(len(data["y"])))
            print("Input metadata:")
            print(json.dumps(data["metadata"], indent=2, sort_keys=True, default=str))
            print("Dry run complete; no algorithms executed.")
            return data

        from baselines.cop_table2.run_reproduction import run_reproduction

        return run_reproduction(data_path, targets_path, methods=methods, lqt_order=args.lqt_order)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
