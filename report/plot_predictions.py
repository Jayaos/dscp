"""Save one PDF per sequence from a base predictor's *_data.pkl artifact.

Example (from the repository root):
    python report/plot_predictions.py data/solar_prediction/lr/lr_nsdb-60m_data.pkl --plot-len 1000 --save-dir report/plots/solar

Works with Sapflux, solar, and air artifacts containing heldout_y and
heldout_predictions. Requires NumPy and Matplotlib.
"""

import argparse
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
import pickle
import re


def _positive_int(raw_value):
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _as_vector(values, sequence_id, field):
    import numpy as np

    context = f"Sequence {sequence_id!r}, {field}"
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected numeric values.") from exc
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{context}: expected a nonempty 1-D array or column vector.")
    if not np.isfinite(array).all():
        raise ValueError(f"{context}: contains NaN or infinite values.")
    return array


def _filename_part(value):
    # Keep series identifiers safe as filenames on Windows and POSIX.
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")[:100] or "sequence"


def plot_predictions(data_path, plot_len, save_dir):
    """Plot the first plot_len held-out steps for every series; return PDF paths.

    Lists, 1-D arrays, and (N, 1) arrays are accepted. Targets and predictions
    must have equal lengths before slicing. Saved values retain their scale.
    """
    if isinstance(plot_len, bool) or not isinstance(plot_len, Integral) or plot_len <= 0:
        raise ValueError("plot_len must be a positive integer.")

    data_path = Path(data_path)
    save_dir = Path(save_dir)
    with data_path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, Mapping) or not data:
        raise ValueError("Expected a nonempty dictionary of sequences in the data pickle.")

    # Validate every sequence before writing plots, without silently truncating
    # mismatched targets and predictions.
    sequences = []
    for sequence_id, record in data.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"Sequence {sequence_id!r}: expected a dictionary.")
        missing = {"heldout_y", "heldout_predictions"}.difference(record)
        if missing:
            raise ValueError(f"Sequence {sequence_id!r}: missing {', '.join(sorted(missing))}.")
        true_y = _as_vector(record["heldout_y"], sequence_id, "heldout_y")
        predicted_y = _as_vector(record["heldout_predictions"], sequence_id, "heldout_predictions")
        if len(true_y) != len(predicted_y):
            raise ValueError(
                f"Sequence {sequence_id!r}: {len(true_y)} targets and "
                f"{len(predicted_y)} predictions; expected equal lengths."
            )
        sequences.append((sequence_id, true_y[:plot_len], predicted_y[:plot_len]))

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import numpy as np

    save_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = data_path.stem.removesuffix("_data")
    saved_paths = []
    for index, (sequence_id, true_y, predicted_y) in enumerate(sequences, start=1):
        steps = np.arange(len(true_y))
        fig, ax = plt.subplots(figsize=(12, 4))
        try:
            ax.plot(
                steps, true_y, label="True y", color="tab:blue", linewidth=1.5,
                marker="o" if len(steps) == 1 else None,
            )
            ax.plot(
                steps, predicted_y, label="Predicted y", color="tab:orange",
                linewidth=1.5, linestyle="--", marker="x" if len(steps) == 1 else None,
            )
            ax.set_title(f"{artifact_name}: {sequence_id}")
            ax.set_xlabel("Held-out step (0-based)")
            ax.set_ylabel("Value")
            ax.set_xlim((-0.5, 0.5) if len(steps) == 1 else (0, len(steps) - 1))
            ax.grid(alpha=0.2)
            ax.legend()
            fig.tight_layout()
            # The index also prevents collisions between sanitized identifiers.
            filename = (
                f"{_filename_part(artifact_name)}_{index:03d}_"
                f"{_filename_part(sequence_id)}_len{len(steps)}.pdf"
            )
            output_path = save_dir / filename
            fig.savefig(output_path, format="pdf", bbox_inches="tight")
            saved_paths.append(output_path)
        finally:
            plt.close(fig)

    return saved_paths


def build_parser():
    parser = argparse.ArgumentParser(
        description="Save true-versus-predicted held-out values as one PDF per sequence."
    )
    parser.add_argument("data_path", type=Path, help="Path to a base predictor's *_data.pkl file.")
    parser.add_argument(
        "--plot-len", type=_positive_int, required=True,
        help="Number of initial held-out steps to plot (capped at each sequence's length).",
    )
    parser.add_argument(
        "--save-dir", "--saving-dir", dest="save_dir", type=Path, required=True,
        help="Directory for the PDFs; created if needed.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        saved_paths = plot_predictions(args.data_path, args.plot_len, args.save_dir)
    except (OSError, ValueError, EOFError, pickle.UnpicklingError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(saved_paths)} sequence PDFs to {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()
