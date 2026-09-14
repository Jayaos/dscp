import argparse

from base_predictor.data import BasePredictorData
from base_predictor.lstm_predictor import LSTMPredictor
from sbatch_run_base_predictor.common import (
    add_shared_data_args,
    default_data_dir,
    default_save_dir,
    parse_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the LSTM base predictor and save its outputs."
    )
    add_shared_data_args(parser, "lstm")
    parser.add_argument(
        "--predictor-train-ratio",
        type=float,
        default=0.33,
        help="Fraction of each series used before heldout splitting inside the predictor.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=100,
        help="Number of past covariate/target steps used for one-step prediction.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        help="Input embedding dimension.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="LSTM hidden dimension.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of LSTM layers.",
    )
    parser.add_argument(
        "--fit-train-ratio",
        type=float,
        default=0.9,
        help=(
            "Chronological inner-training fraction applied to each sequence's "
            "fitting prefix; the remaining tail is used for validation."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training and heldout inference.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=1,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=3,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed used when shuffling sequence windows.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Torch device passed through to the predictor.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir is not None else default_data_dir(args.data_type)
    save_dir = args.save_dir if args.save_dir is not None else default_save_dir(args.data_type, "lstm")
    device = parse_device(args.device)

    print(f"Data type: {args.data_type}")
    print(f"Data dir: {data_dir}")
    print(f"Save dir: {save_dir}")
    print(f"Predictor train ratio: {args.predictor_train_ratio}")
    print(f"Window length: {args.window_length}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Num layers: {args.num_layers}")
    print(f"Fit train ratio: {args.fit_train_ratio}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max epoch: {args.max_epoch}")
    print(f"Early stop: {args.early_stop}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    base_predictor_data = BasePredictorData()
    base_predictor_data.load_data(args.data_type, str(data_dir))

    predictor = LSTMPredictor(
        base_predictor_data,
        args.embedding_dim,
        args.hidden_dim,
        args.num_layers,
        args.predictor_train_ratio,
        args.window_length,
    )
    predictor.fit_predict(
        args.fit_train_ratio,
        args.batch_size,
        args.learning_rate,
        args.max_epoch,
        args.early_stop,
        seed=args.seed,
        device=device,
    )
    predictor.save(str(save_dir))


if __name__ == "__main__":
    main()
