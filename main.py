
import argparse
import os
from preprocess import preprocess
from pretrain import pretrain

def main():
    parser = argparse.ArgumentParser(description="Mahjong AI Pre-training CLI")
    parser.add_argument(
        "action",
        choices=["preprocess", "pretrain", "all"],
        help="Action to perform: 'preprocess' data, 'pretrain' model, or 'all'.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Path to the directory to save models.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Proportion of the dataset to include in the test split.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.1,
        help="Proportion of the dataset to include in the validation split.",
    )

    args = parser.parse_args()

    raw_logs_dir = os.path.join(args.data_dir, "raw_logs")
    processed_dir = os.path.join(args.data_dir, "processed")

    if args.action == "preprocess" or args.action == "all":
        print("--- Starting Preprocessing ---")
        preprocess(
            raw_logs_dir=raw_logs_dir,
            processed_dir=processed_dir,
            test_size=args.test_size,
            validation_size=args.validation_size,
        )
        print("--- Preprocessing Finished ---")

    if args.action == "pretrain" or args.action == "all":
        print("--- Starting Pre-training ---")
        pretrain(
            processed_dir=processed_dir,
            model_dir=args.model_dir,
        )
        print("--- Pre-training Finished ---")


if __name__ == "__main__":
    main()

