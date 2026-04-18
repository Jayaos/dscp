import argparse

from base_predictor.data import BasePredictorData
from base_predictor.ridge_predictor import RidgeRegressionPredictor
from sbatch_run_base_predictor.common import (
    add_shared_data_args,
    default_data_dir,
    default_save_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ridge regression base predictor and save its outputs."
    )
    add_shared_data_args(parser, "ridge")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.33,
        help="Fraction of each series used for training before heldout evaluation.",
    )
    parser.add_argument(
        "--min-alpha",
        type=float,
        default=0.0001,
        help="Smallest RidgeCV alpha value.",
    )
    parser.add_argument(
        "--max-alpha",
        type=float,
        default=10.0,
        help="Largest RidgeCV alpha value.",
    )
    parser.add_argument(
        "--num-alphas",
        type=int,
        default=10,
        help="Number of evenly spaced RidgeCV alpha values.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir is not None else default_data_dir(args.data_type)
    save_dir = args.save_dir if args.save_dir is not None else default_save_dir(args.data_type, "ridge")

    print(f"Data type: {args.data_type}")
    print(f"Data dir: {data_dir}")
    print(f"Save dir: {save_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Alpha range: [{args.min_alpha}, {args.max_alpha}]")
    print(f"Num alphas: {args.num_alphas}")

    base_predictor_data = BasePredictorData()
    base_predictor_data.load_data(args.data_type, str(data_dir))

    predictor = RidgeRegressionPredictor(
        base_predictor_data,
        args.train_ratio,
        args.min_alpha,
        args.max_alpha,
        args.num_alphas,
    )
    predictor.fit_predict()
    predictor.save(str(save_dir))


if __name__ == "__main__":
    main()
