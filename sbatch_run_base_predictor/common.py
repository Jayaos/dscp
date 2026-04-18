import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_TYPE_CHOICES = ["air-10", "air-25", "nsdb-60m", "nsdb-30m", "toy"]


def default_data_dir(data_type: str) -> Path:
    if data_type in {"air-10", "air-25"}:
        return REPO_ROOT / "data" / "bejing_air_quality"

    if data_type in {"nsdb-60m", "nsdb-30m"}:
        return REPO_ROOT / "data" / "nsdb_2018-2020"

    if data_type == "toy":
        return REPO_ROOT / "example"

    raise ValueError(f"Unsupported data type: {data_type}")


def default_save_dir(data_type: str, predictor_name: str) -> Path:
    return REPO_ROOT / "data" / data_type / predictor_name


def parse_device(raw_device: str):
    try:
        return int(raw_device)
    except ValueError:
        return raw_device


def add_shared_data_args(
    parser: argparse.ArgumentParser,
    predictor_name: str,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "data_type",
        choices=DATA_TYPE_CHOICES,
        help="Dataset type to load with BasePredictorData.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional dataset path override. Defaults are inferred from data_type.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help=f"Optional output directory override. Defaults to data/<data_type>/{predictor_name}.",
    )
    return parser
