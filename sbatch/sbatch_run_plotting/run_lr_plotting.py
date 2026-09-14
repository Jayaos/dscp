"""Plot held-out forecasts from linear-regression base predictors."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from utils.plotting import plot_darts_predictions
from utils.utils import load_data


REPO_ROOT = Path(__file__).resolve().parents[2]
LR_ARTIFACTS = {
    "air-10": Path("data/air-10_prediction/lr/lr_air-10_data.pkl"),
    "nsdb-60m": Path("data/solar_prediction/lr/lr_nsdb-60m_data.pkl"),
    "sapflux-solo3-large": Path(
        "data/sapflux_prediction/lr/lr_sapflux-solo3-large_data.pkl"
    ),
}


def _positive_int(raw_value):
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot true and predicted held-out values from LR artifacts."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(LR_ARTIFACTS),
        default=list(LR_ARTIFACTS),
        help="Datasets to plot. All three are plotted by default.",
    )
    parser.add_argument(
        "--plot-len",
        type=_positive_int,
        default=1000,
        help="Number of held-out time steps to show in each plot.",
    )
    parser.add_argument(
        "--n-seqs",
        type=_positive_int,
        default=2,
        help="Number of series to plot from each dataset (default: 2).",
    )
    parser.add_argument(
        "--all-series",
        action="store_true",
        help="Plot every series, overriding --n-seqs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional output root. Relative paths are resolved from the repository "
            "root, with one subdirectory created per dataset. By default, plots "
            "are saved beside each artifact under a plots directory."
        ),
    )
    return parser


def _resolve_output_dir(dataset, artifact_path, output_root):
    if output_root is None:
        return artifact_path.parent / "plots"

    root = output_root if output_root.is_absolute() else REPO_ROOT / output_root
    return root / dataset


def _validate_prediction_data(prediction_data, artifact_path):
    if not isinstance(prediction_data, dict) or not prediction_data:
        raise ValueError(
            "Expected a non-empty dictionary in LR artifact: {}".format(
                artifact_path
            )
        )

    required_fields = {"heldout_y", "heldout_predictions"}
    for series_name, series_data in prediction_data.items():
        if not isinstance(series_data, dict):
            raise ValueError(
                "Series {!r} in {} is not a dictionary.".format(
                    series_name, artifact_path
                )
            )
        missing_fields = required_fields.difference(series_data)
        if missing_fields:
            raise ValueError(
                "Series {!r} in {} is missing: {}".format(
                    series_name,
                    artifact_path,
                    ", ".join(sorted(missing_fields)),
                )
            )

        target_length = len(series_data["heldout_y"])
        prediction_length = len(series_data["heldout_predictions"])
        if target_length == 0 or target_length != prediction_length:
            raise ValueError(
                "Series {!r} in {} has {} targets and {} predictions; "
                "expected equal, non-zero lengths.".format(
                    series_name,
                    artifact_path,
                    target_length,
                    prediction_length,
                )
            )


def main():
    parser = build_parser()
    args = parser.parse_args()

    artifact_paths = {
        dataset: REPO_ROOT / LR_ARTIFACTS[dataset] for dataset in args.datasets
    }
    missing_artifacts = [
        path for path in artifact_paths.values() if not path.is_file()
    ]
    if missing_artifacts:
        parser.error(
            "Missing LR artifact(s):\n  {}".format(
                "\n  ".join(str(path) for path in missing_artifacts)
            )
        )

    for dataset, artifact_path in artifact_paths.items():
        print("Loading {} LR predictions from {}".format(dataset, artifact_path))
        prediction_data = load_data(str(artifact_path))
        _validate_prediction_data(prediction_data, artifact_path)

        n_seqs = len(prediction_data) if args.all_series else args.n_seqs
        n_seqs = min(n_seqs, len(prediction_data))
        output_dir = _resolve_output_dir(
            dataset, artifact_path, args.output_root
        )

        plot_darts_predictions(
            prediction_data,
            plot_len=args.plot_len,
            n_seqs=n_seqs,
            save_dir=str(output_dir),
        )
        print(
            "Saved {} {} plot(s) to {}".format(
                n_seqs, dataset, output_dir
            )
        )


if __name__ == "__main__":
    main()
