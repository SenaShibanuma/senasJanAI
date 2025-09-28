# -*- coding: utf-8 -*-
import os
import pickle
import time
from .transformer_parser import TransformerParser
from .train_transformer import preprocess_data # ベクトル化関数をインポート

def generate_and_save_data(data_dir, output_path):
    """
    【改良版】指定されたディレクトリからログファイルを解析し、前処理済みのデータを
    指定されたパスに保存する。os.scandir() を使用して大量のファイルによるタイムアウトを回避する。
    """
    print("--- Starting Data Generation (Robust Version) ---")
    
    if not os.path.isdir(data_dir):
        print(f"Error: The specified data directory '{data_dir}' does not exist or is not a directory.")
        return

    # 1. ログファイルのパスを効率的に取得
    log_files = []
    print(f"Scanning for log files in '{data_dir}'...")
    try:
        # os.listdir() の代わりに os.scandir() を使用してタイムアウトを回避
        for entry in os.scandir(data_dir):
            if entry.is_file() and (entry.name.endswith('.mjlog') or entry.name.endswith('.gz')):
                log_files.append(entry.path)
    except OSError as e:
        print(f"Error while scanning directory '{data_dir}': {e}")
        print("Please check if the directory is accessible and not corrupted.")
        return

    if not log_files:
        print(f"Error: No log files (.mjlog or .mjlog.gz) found in '{data_dir}'.")
        return

    print(f"Successfully found {len(log_files)} log files to process.")
    # 意図的に少し待機し、Driveの同期を安定させる
    time.sleep(2)

    # 2. パーサーを実行して生データを抽出
    parser = TransformerParser()
    training_data = parser.run(log_files)

    if not training_data:
        print("Error: No training data could be extracted from the log files.")
        return
    
    print(f"\nSuccessfully parsed {len(training_data)} data points.")
    
    # 3. データをベクトル形式に前処理
    print("Preprocessing and vectorizing data for the model...")
    processed_dataset = preprocess_data(training_data)
    
    # 4. データをファイルに保存 (常に上書き)
    print(f"Saving processed dataset to '{output_path}'...")
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_path, 'wb') as f:
        pickle.dump(processed_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("--- Data Generation Complete ---")
    print(f"Dataset saved successfully. Total samples: {len(processed_dataset[2])}")


def main():
    DATA_DIR = 'data'
    PROCESSED_DATA_PATH = 'processed_data/training_dataset_transformer.pkl'
    generate_and_save_data(DATA_DIR, PROCESSED_DATA_PATH)

if __name__ == '__main__':
    main()