import argparse
import os
from preprocess import preprocess
from pretrain import pretrain

def main():
    parser = argparse.ArgumentParser(description="麻雀AIの事前学習・実行CLI")
    parser.add_argument(
        "action",
        choices=["preprocess", "pretrain", "all"],
        help="実行するアクション: 'preprocess' (データ前処理), 'pretrain' (モデル学習), or 'all' (両方)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="データディレクトリへのパス",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="モデルを保存するディレクトリへのパス",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="データセット全体に対するテストデータの割合",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.1,
        help="データセット全体に対する検証データの割合",
    )

    args = parser.parse_args()

    raw_logs_dir = os.path.join(args.data_dir, "raw_logs")
    processed_dir = os.path.join(args.data_dir, "processed")

    if args.action == "preprocess" or args.action == "all":
        print("--- データの前処理を開始します ---")
        preprocess(
            raw_logs_dir=raw_logs_dir,
            processed_dir=processed_dir,
            test_size=args.test_size,
            validation_size=args.validation_size,
        )
        print("--- データの前処理が完了しました ---")

    if args.action == "pretrain" or args.action == "all":
        print("--- モデルの事前学習を開始します ---")
        pretrain(
            processed_dir=processed_dir,
            model_dir=args.model_dir,
        )
        print("--- モデルの事前学習が完了しました ---")


if __name__ == "__main__":
    main()