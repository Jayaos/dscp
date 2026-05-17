import argparse
from pathlib import Path

from base_predictor.chronos_predictor import ChronosPredictor
from base_predictor.data import BasePredictorData


REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_data_dir(data_type: str) -> Path:
    if data_type in {"air-10", "air-25"}:
        return REPO_ROOT / "data" / "bejing_air_quality"

    if data_type in {"nsdb-60m", "nsdb-30m"}:
        return REPO_ROOT / "data" / "nsdb_2018-2020"

    raise ValueError(f"Unsupported data type: {data_type}")


def _default_save_dir(data_type: str) -> Path:
    return REPO_ROOT / "data" / data_type / "chronos"


def _parse_device(raw_device: str):
    try:
        return int(raw_device)
    except ValueError:
        return raw_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Chronos base predictor and save its outputs."
    )
    parser.add_argument(
        "data_type",
        choices=["air-10", "air-25", "nsdb-60m", "nsdb-30m", "toy"],
        help="Dataset type to load with BasePredictorData.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional dataset directory override. Defaults are inferred from data_type.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional output directory override. Defaults to data/<data_type>/chronos.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=100,
        help="Chronos context window length.",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=10,
        help="Chronos forecast horizon per rollout step.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to Chronos2Pipeline.from_pretrained(device_map=...).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir is not None else _default_data_dir(args.data_type)
    save_dir = args.save_dir if args.save_dir is not None else _default_save_dir(args.data_type)
    device = _parse_device(args.device)

    print(f"Data type: {args.data_type}")
    print(f"Data dir: {data_dir}")
    print(f"Save dir: {save_dir}")
    print(f"Window length: {args.window_length}")
    print(f"Prediction length: {args.prediction_length}")
    print(f"Device: {device}")

    base_predictor_data = BasePredictorData()
    base_predictor_data.load_data(args.data_type, str(data_dir))

    predictor = ChronosPredictor(base_predictor_data, device)
    predictor.predict(args.window_length, args.prediction_length)
    predictor.save(str(save_dir))


if __name__ == "__main__":
    main()
