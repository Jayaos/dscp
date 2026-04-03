import argparse

from base_predictor.data import BasePredictorData
from base_predictor.lightgbm_predictor import LightGBMPredictor
from sbatch_run_base_predictor.common import (
    add_shared_data_args,
    default_data_dir,
    default_save_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the LightGBM base predictor and save its outputs."
    )
    add_shared_data_args(parser, "lightgbm")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.33,
        help="Fraction of each series used for training before heldout evaluation.",
    )
    parser.add_argument(
        "--past-window",
        type=int,
        default=50,
        help="Number of past target and covariate steps used for prediction.",
    )
    parser.add_argument(
        "--prediction-step",
        type=int,
        default=1,
        help="Forecast output chunk length.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir is not None else default_data_dir(args.data_type)
    save_dir = args.save_dir if args.save_dir is not None else default_save_dir(args.data_type, "lightgbm")

    print(f"Data type: {args.data_type}")
    print(f"Data dir: {data_dir}")
    print(f"Save dir: {save_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Past window: {args.past_window}")
    print(f"Prediction step: {args.prediction_step}")

    base_predictor_data = BasePredictorData()
    base_predictor_data.load_data(args.data_type, str(data_dir))

    predictor = LightGBMPredictor(base_predictor_data, args.train_ratio)
    predictor.fit_predict(args.past_window, args.prediction_step)
    predictor.save(str(save_dir))


if __name__ == "__main__":
    main()
