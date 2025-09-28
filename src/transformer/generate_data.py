# -*- coding: utf-8 -*-
import os
import pickle
import time
import numpy as np
from .transformer_parser import TransformerParser
from .train_transformer import preprocess_data # ベクトル化関数をインポート

def generate_and_save_data(data_dir, output_path):
    """
    【メモリ対策版】ファイルを1つずつ処理し、即座にベクトル化することで、
    大量のファイルを扱う際のメモリ不足を回避する。
    """
    print("--- Starting Data Generation (Memory Safe Version) ---")
    
    if not os.path.isdir(data_dir):
        print(f"Error: The specified data directory '{data_dir}' does not exist or is not a directory.")
        return

    # 1. ログファイルのパスをスキャン
    log_files = []
    print(f"Scanning for log files in '{data_dir}'...")
    try:
        for entry in os.scandir(data_dir):
            if entry.is_file() and (entry.name.endswith('.mjlog') or entry.name.endswith('.gz')):
                log_files.append(entry.path)
    except OSError as e:
        print(f"Error while scanning directory '{data_dir}': {e}")
        return

    if not log_files:
        print(f"Error: No log files (.mjlog or .mjlog.gz) found in '{data_dir}'.")
        return

    total_files = len(log_files)
    print(f"Found {total_files} log files. Starting iterative processing...")
    time.sleep(2)

    parser = TransformerParser()
    
    # 最終的なベクトルデータを格納するリスト
    all_contexts = []
    all_choices = []
    all_labels = []
    
    # 2. ファイルを1つずつループ処理
    for i, filepath in enumerate(log_files):
        # 100ファイルごとに進捗を表示
        if (i + 1) % 100 == 0:
            filename = os.path.basename(filepath)
            print(f"Processing file {i+1}/{total_files}: {filename}")

        # a) 単一のファイルを解析
        try:
            raw_data_points = parser.parse_log_file(filepath)
        except Exception:
            # XML解析エラーなどが発生した場合はスキップ
            continue

        if not raw_data_points:
            continue

        # b) 即座にベクトル化
        contexts, choices, labels = preprocess_data(raw_data_points)

        # c) 結果をリストに追加（numpy配列をpythonリストに変換して追加）
        if labels.size > 0:
            all_contexts.extend(list(contexts))
            all_choices.extend(list(choices))
            all_labels.extend(list(labels))

    # 3. 最終確認
    if not all_labels:
        print("\nError: No training data could be extracted from the log files after processing.")
        print("This might be due to issues in the parser logic or incompatible log file formats.")
        return

    print(f"\nSuccessfully processed all files. Aggregating final dataset containing {len(all_labels)} samples...")
    
    # 4. 全てのベクトルデータをNumPy配列に結合
    final_contexts_np = np.array(all_contexts, dtype=np.float32)
    final_choices_np = np.array(all_choices, dtype=np.float32)
    final_labels_np = np.array(all_labels, dtype=np.int32)

    processed_dataset = (final_contexts_np, final_choices_np, final_labels_np)
    
    # 5. データセットを保存
    print(f"Saving processed dataset to '{output_path}'...")
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_path, 'wb') as f:
        pickle.dump(processed_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("--- Data Generation Complete ---")
    print(f"Dataset saved successfully. Total samples: {len(final_labels_np)}")


def main():
    DATA_DIR = 'data'
    PROCESSED_DATA_PATH = 'processed_data/training_dataset_transformer.pkl'
    generate_and_save_data(DATA_DIR, PROCESSED_DATA_PATH)

if __name__ == '__main__':
    main()