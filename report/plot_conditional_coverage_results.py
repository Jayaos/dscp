"""Plot signed group coverage errors across sequences from evaluation CSVs.

Example:
    python report/plot_conditional_coverage_results.py results/conditional_coverage/air/lstm/evaluation/k3

Saves one box plot per named run, dataset, base predictor, and K.
Requires Matplotlib for plotting; only the two evaluation CSVs are read.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import textwrap

if __package__:
    from .inspect_conditional_coverage_results import _read_csv
else:
    from inspect_conditional_coverage_results import _read_csv


SUMMARY_FIELDS = (
    "name", "method", "dataset", "base_predictor", "k", "nominal_coverage", "mean_sequence_coverage",
)
GROUP_FIELDS = ("name", "method", "sequence_key", "sequence_key_type", "group", "coverage")


def _fraction(value, context, *, nominal=False):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected a numeric fraction.") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{context}: expected a finite fraction in [0, 1].")
    if nominal and not 0 < number < 1:
        raise ValueError(f"{context}: nominal coverage must be between zero and one.")
    return number


def load_plot_data(results_dir):
    """Return one plot record per run, preserving typed sequence identities.

    ``errors`` is a sequence-major list, with groups in numeric order 0..K-1.
    Missing coverage is None. Group labels remain local to each sequence.
    ``average_coverage_error`` is the saved mean sequence coverage minus nominal.
    Validate all rows before any plots are written.
    """
    results_dir = Path(results_dir).expanduser().resolve()
    summaries = _read_csv(results_dir / "summary.csv", SUMMARY_FIELDS)
    groups = _read_csv(results_dir / "group_metrics.csv", GROUP_FIELDS)
    runs = {}
    by_run = {}
    for row in summaries:
        identity = (row["name"], row["method"])
        if identity in runs:
            raise ValueError(f"summary.csv: duplicate run {identity!r}; use separate evaluation directories.")
        try:
            k = int(row["k"])
        except ValueError as exc:
            raise ValueError(f"summary.csv: run {identity!r} has invalid K {row['k']!r}.") from exc
        if k <= 0:
            raise ValueError(f"summary.csv: run {identity!r} requires positive K.")
        nominal = _fraction(row["nominal_coverage"], f"summary.csv: {identity!r}", nominal=True)
        mean_coverage = row["mean_sequence_coverage"].strip()
        average_error = (
            _fraction(mean_coverage, f"summary.csv: {identity!r}, mean_sequence_coverage") - nominal
            if mean_coverage else None
        )
        runs[identity] = {field: row[field] for field in SUMMARY_FIELDS}
        runs[identity].update(k=k, nominal_coverage=nominal, average_coverage_error=average_error)
        by_run[identity] = {}

    for row in groups:
        identity = (row["name"], row["method"])
        if identity not in runs:
            raise ValueError(f"group_metrics.csv: run {identity!r} has no matching summary.csv row.")
        run = runs[identity]
        sequence = (row["sequence_key_type"], row["sequence_key"])
        context = f"group_metrics.csv: run {identity!r}, sequence {sequence!r}"
        try:
            group = int(row["group"])
        except ValueError as exc:
            raise ValueError(f"{context}: invalid group ID {row['group']!r}.") from exc
        if not 0 <= group < run["k"]:
            raise ValueError(f"{context}: group ID {group} must be in [0, K).")
        values = by_run[identity].setdefault(sequence, {})
        if group in values:
            raise ValueError(f"{context}: duplicate group {group}.")
        nominal = run["nominal_coverage"]
        if row.get("nominal_coverage"):
            group_nominal = _fraction(row["nominal_coverage"], context, nominal=True)
            if not math.isclose(group_nominal, nominal, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{context}: nominal coverage disagrees with summary.csv.")
        values[group] = (
            _fraction(row["coverage"], f"{context}, group {group}") - nominal
            if row["coverage"].strip() else None
        )

    plots = []
    for identity, run in runs.items():
        sequences = by_run[identity]
        if not sequences:
            raise ValueError(f"group_metrics.csv: no groups for run {identity!r}.")
        for sequence, values in sequences.items():
            if len(values) != run["k"]:
                raise ValueError(f"group_metrics.csv: run {identity!r}, sequence {sequence!r}: "
                                 f"expected all {run['k']} group rows, including empty groups.")
        plots.append({
            **run, "sequences": list(sequences),
            "errors": [[values[group] for group in range(run["k"])] for values in sequences.values()],
        })
    return plots


def _filename(run, results_dir, file_format):
    parts = [run["dataset"], run["base_predictor"], run["method"], run["name"], f"k{run['k']}"]
    safe_parts = [re.sub(r"[^A-Za-z0-9_.-]+", "_", part).strip("._")[:32] or "unknown"
                  for part in parts]
    # Include the source directory to keep separate evaluations from colliding.
    identity = [str(results_dir), *parts, run["nominal_coverage"]]
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
    return "_".join([*safe_parts, digest, "signed_coverage_error"]) + f".{file_format}"


def plot_conditional_coverage_results(results_dir, save_dir=None, file_format="pdf"):
    """Save one box plot per run and return paths; default to <results_dir>/plots.

    Each sequence's box summarizes its available group signed coverage errors,
    weighting groups equally. Individual group errors are overlaid as dots.
    Missing groups are omitted; entirely unavailable sequences are labeled n/a.
    Input CSVs remain unchanged.
    """
    if file_format not in ("pdf", "png"):
        raise ValueError("file_format must be 'pdf' or 'png'.")
    results_dir = Path(results_dir).expanduser().resolve()
    plots = load_plot_data(results_dir)
    destination = (Path(save_dir).expanduser().resolve() if save_dir is not None
                   else results_dir / "plots")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for run in plots:
        n_sequences, k = len(run["sequences"]), run["k"]
        fig, ax = plt.subplots(figsize=(max(10, min(40, 3 + 0.9 * n_sequences)), 6))
        try:
            samples = [[error for error in errors if error is not None] for errors in run["errors"]]
            positions = [index for index, values in enumerate(samples) if values]
            if positions:
                ax.boxplot(
                    [samples[index] for index in positions], positions=positions,
                    widths=0.5, patch_artist=True, manage_ticks=False, whis=1.5,
                    showfliers=False,  # Every observation is already shown as a dot.
                    boxprops={"facecolor": "#d9e8f5", "edgecolor": "#35658b"},
                    medianprops={"color": "#bd4b20", "linewidth": 1.8},
                    whiskerprops={"color": "#35658b"}, capprops={"color": "#35658b"},
                    zorder=3,
                )
            for index, values in enumerate(samples):
                if values:
                    offsets = ([0] if len(values) == 1 else
                               [-0.12 + 0.24 * group / (len(values) - 1) for group in range(len(values))])
                    ax.scatter([index + offset for offset in offsets], values,
                               color="#244865", s=22, alpha=0.85, zorder=4)
                else:
                    ax.annotate("n/a", (index, 0), xytext=(0, 7), textcoords="offset points",
                                ha="center", va="bottom", fontsize=9, color="#777777")
            ax.axhline(0, color="#303030", linewidth=1, zorder=2)
            reference_errors = [-0.025, -0.05]
            for value in reference_errors:
                ax.axhline(value, color="grey", linestyle=":", alpha=0.5, zorder=2)
            average_error = run["average_coverage_error"]
            if average_error is not None:
                ax.axhline(average_error, color="red", linestyle=":", alpha=0.5,
                           label="Average coverage error", zorder=4)
            labels = [f"{key!r} ({kind})" if kind == "str" else f"{key} ({kind})"
                      for kind, key in run["sequences"]]
            ax.set_xticks(range(n_sequences))
            ax.set_xticklabels([textwrap.fill(label, 28) + f"\nn={len(values)}/{k}"
                                for label, values in zip(labels, samples)],
                               rotation=45, ha="right", fontsize=9)
            ax.set_xlim(-0.6, n_sequences - 0.4)
            finite_errors = [value for errors in run["errors"] for value in errors if value is not None]
            limits = [0, *finite_errors, *reference_errors]
            if average_error is not None:
                limits.append(average_error)
            low, high = min(limits), max(limits)
            padding = max(0.02, (high - low) * 0.15)
            ax.set_ylim(low - padding, high + padding)
            ax.set_ylabel("Coverage error")
            title = (f"{run['dataset'] or 'Unknown dataset'} | {run['base_predictor'] or 'Unknown predictor'} | "
                     f"{run['method']} | {run['name']}")
            ax.set_title(textwrap.fill(title, 110) +
                         f"\nK = {k} | Nominal coverage = {run['nominal_coverage']:.6g}", pad=14)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            path = destination / _filename(run, results_dir, file_format)
            fig.savefig(path, format=file_format, dpi=180, bbox_inches="tight")
            paths.append(path)
        finally:
            plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, help="Directory containing summary.csv and group_metrics.csv.")
    parser.add_argument("--save-dir", type=Path, help="Plot directory; default: <results_dir>/plots.")
    parser.add_argument("--format", dest="file_format", choices=("pdf", "png"), default="pdf",
                        help="Image format (default: pdf).")
    args = parser.parse_args(argv)
    try:
        paths = plot_conditional_coverage_results(args.results_dir, args.save_dir, args.file_format)
    except (OSError, UnicodeError, csv.Error, ValueError, ImportError) as exc:
        parser.exit(1, f"Error plotting {args.results_dir}: {exc}\n")
    for path in paths:
        print(f"Saved {path}")
    return paths


if __name__ == "__main__":
    main()
