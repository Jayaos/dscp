"""Inspect saved conditional coverage summary and per-sequence group CSVs.

Examples:
    python report/inspect_conditional_coverage_results.py results/conditional_coverage/air/lstm/evaluation/k3
    python report/inspect_conditional_coverage_results.py path/to/air/evaluation path/to/solar/evaluation
"""

import argparse
import csv
from pathlib import Path


CONTEXT_FIELDS = ("dataset", "base_predictor", "k", "nominal_coverage")
SUMMARY_REQUIRED = (
    "name", "method", *CONTEXT_FIELDS, "cace", "wuc",
    "mean_sequence_coverage", "pooled_coverage",
)
GROUP_REQUIRED = (
    "name", "method", "sequence_key", "sequence_key_type", "group",
    "coverage", "group_size", "evaluated_count", "missing_count", "covered_count",
)
SUMMARY_METRICS = (
    ("CACE", "cace"), ("Mean WUC", "wuc"),
    ("Worst WUC", "worst_sequence_wuc"),
    ("Mean seq coverage", "mean_sequence_coverage"),
    ("Pooled coverage", "pooled_coverage"),
)
SUMMARY_COUNTS = (
    ("Sequences", "sequence_count"), ("Used seqs", "evaluated_sequence_count"),
    ("Unavailable seqs", "unavailable_sequence_count"),
    ("Evaluated", "evaluated_count"), ("Missing", "missing_count"),
    ("Used groups", "evaluated_group_count"), ("Empty groups", "empty_group_count"),
)
GROUP_COLUMNS = (
    ("Group", "group"), ("Coverage", "coverage"),
    ("Covered", "covered_count"), ("Evaluated", "evaluated_count"),
    ("Missing", "missing_count"), ("Size", "group_size"),
    ("Undercoverage", "undercoverage"), ("Signed error", "coverage_error"),
    ("Abs error", "absolute_coverage_error"),
)
FRACTION_FIELDS = {
    "cace", "wuc", "worst_sequence_wuc", "mean_sequence_coverage",
    "pooled_coverage", "coverage", "undercoverage", "coverage_error", "absolute_coverage_error",
}


def _read_csv(path, required):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}, line {reader.line_num}: row does not match CSV header.")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no result rows found.")
    return rows


def _value(value, field=None):
    if value is None or value == "":
        return "n/a"
    if field in FRACTION_FIELDS:
        return f"{float(value):.6g}"
    return str(value)


def _print_table(rows, columns):
    cells = [[_value(row.get(field), field) for _, field in columns] for row in rows]
    widths = [max(len(label), *(len(row[i]) for row in cells))
              for i, (label, _) in enumerate(columns)]
    print("  " + "  ".join(label.ljust(width) for (label, _), width in zip(columns, widths)))
    print("  " + "  ".join("-" * width for width in widths))
    for row in cells:
        print("  " + "  ".join(value.ljust(width) for value, width in zip(row, widths)))


def _print_context(row):
    print(f"Dataset: {_value(row['dataset'])}; base predictor: {_value(row['base_predictor'])}; "
          f"K: {_value(row['k'])}; nominal coverage: {_value(row['nominal_coverage'])}")


def inspect_results(results_dir):
    """Print both CSV reports and return their rows with original string values.

    Group identities stay local to a run and typed sequence key. Dataset and
    predictor identity come from the matching summary row. Only the two CSVs
    are needed; saved model logs and grouping artifacts are not loaded.
    """
    results_dir = Path(results_dir).expanduser().resolve()
    summary = _read_csv(results_dir / "summary.csv", SUMMARY_REQUIRED)
    groups = _read_csv(results_dir / "group_metrics.csv", GROUP_REQUIRED)

    runs = {}
    contexts = {}
    for row in summary:
        identity = (row["name"], row["method"])
        if identity in runs:
            raise ValueError(f"summary.csv: duplicate run {identity!r}; use separate evaluation directories.")
        runs[identity] = row
        contexts.setdefault(tuple(row[field] for field in CONTEXT_FIELDS), []).append(row)

    by_run = {identity: {} for identity in runs}
    seen = set()
    for row in groups:
        identity = (row["name"], row["method"])
        if identity not in runs:
            raise ValueError(f"group_metrics.csv: run {identity!r} has no matching summary.csv row.")
        sequence = (row["sequence_key_type"], row["sequence_key"])
        try:
            group_id = int(row["group"])
        except ValueError as exc:
            raise ValueError(f"group_metrics.csv: invalid group ID {row['group']!r}.") from exc
        if group_id < 0:
            raise ValueError("group_metrics.csv: group IDs must be nonnegative.")
        key = (identity, sequence, group_id)
        if key in seen:
            raise ValueError(f"group_metrics.csv: duplicate group row for {key!r}.")
        seen.add(key)
        by_run[identity].setdefault(sequence, []).append(row)
    for identity, sequences in by_run.items():
        if not sequences:
            raise ValueError(f"group_metrics.csv: no groups for run {identity!r}.")

    print(f"Results directory: {results_dir}")
    print("Coverage and errors are fractions (0.9 = 90%); n/a means unavailable.")
    print("Summary CACE and WUC average sequences equally; mean seq coverage also weights sequences equally.")
    print("Pooled coverage weights evaluated observations; worst WUC is the largest sequence WUC.")
    print("Group labels are local to each sequence. Coverage uses evaluated predictions only.")
    print("Signed error = coverage - nominal coverage; negative means undercoverage, positive means overcoverage.")
    print("\nDataset summary (summary.csv)")
    for rows in contexts.values():
        _print_context(rows[0])
        _print_table(rows, (("Run", "name"), ("Method", "method"), *SUMMARY_METRICS))
        print("  Availability:")
        _print_table(rows, (("Run", "name"), ("Method", "method"), *SUMMARY_COUNTS))
        definitions = {}
        for row in rows:
            definitions.setdefault(row.get("cace_definition") or "not saved", []).append(row["name"])
        for definition, names in definitions.items():
            print(f"  CACE definition: {definition} (runs: {', '.join(names)})")

    print("\nPer-sequence group coverage (group_metrics.csv)")
    for identity, sequences in by_run.items():
        print()
        _print_context(runs[identity])
        print(f"Run: {identity[0]}; method: {identity[1]}")
        for (key_type, key), rows in sequences.items():
            label = repr(key) if key_type == "str" else key
            print(f"  Sequence: {label} ({key_type})")
            display_rows = []
            for row in sorted(rows, key=lambda row: int(row["group"])):
                nominal = row.get("nominal_coverage") or runs[identity]["nominal_coverage"]
                error = float(row["coverage"]) - float(nominal) if row["coverage"] and nominal else None
                display_rows.append({**row, "coverage_error": error})
            _print_table(display_rows, GROUP_COLUMNS)

    return {"summary": summary, "group_metrics": groups}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dirs", type=Path, nargs="+",
                        help="Evaluation directories containing summary.csv and group_metrics.csv.")
    args = parser.parse_args(argv)
    reports = []
    for directory in args.results_dirs:
        try:
            if reports:
                print()
            reports.append(inspect_results(directory))
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            parser.exit(1, f"Error reading {directory}: {exc}\n")
    return reports


if __name__ == "__main__":
    main()
