# -*- coding: utf--8 -*-
import os
import pickle
import time
import numpy as np
import glob
from .transformer_parser import TransformerParser
from .train_transformer import preprocess_data

def generate_data_in_chunks(data_dir, chunk_dir, chunk_size=1000):
    """
    【最終版：チャンク処理】データをチャンクに分割して処理し、
    中間ファイルをディスクに保存することでメモリ不足を完全に回避する。
    """
    print("--- Starting Data Generation (Chunked Processing) ---")
    
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return False

    # 一時的なチャンク保存ディレクトリを作成
    os.makedirs(chunk_dir, exist_ok=True)
    # 古いチャンクファイルが残っていれば削除
    for old_chunk in glob.glob(os.path.join(chunk_dir, "*.pkl")):
        os.remove(old_chunk)

    log_files = []
    print(f"Scanning for log files in '{data_dir}'...")
    try:
        for entry in os.scandir(data_dir):
            if entry.is_file() and (entry.name.endswith('.mjlog') or entry.name.endswith('.gz')):
                log_files.append(entry.path)
    except OSError as e:
        print(f"Error while scanning directory '{data_dir}': {e}")
        return False

    if not log_files:
        print(f"Error: No log files found in '{data_dir}'.")
        return False

    total_files = len(log_files)
    print(f"Found {total_files} log files. Starting to process in chunks of {chunk_size} files...")
    time.sleep(2)

    parser = TransformerParser()
    chunk_num = 1
    
    # チャンクごとのデータを保持するリスト
    chunk_contexts = []
    chunk_choices = []
    chunk_labels = []

    for i, filepath in enumerate(log_files):
        try:
            raw_data_points = parser.parse_log_file(filepath)
            if raw_data_points:
                contexts, choices, labels = preprocess_data(raw_data_points)
                if labels.size > 0:
                    chunk_contexts.extend(list(contexts))
                    chunk_choices.extend(list(choices))
                    chunk_labels.extend(list(labels))
        except Exception as e:
            # 個別ファイルの解析エラーは無視して次に進む
            # print(f"Warning: Skipping file {os.path.basename(filepath)} due to error: {e}")
            continue

        # チャンクサイズに達したか、最後のファイルであれば、チャンクを保存
        if (i + 1) % chunk_size == 0 or (i + 1) == total_files:
            if chunk_labels:
                print(f"Saving chunk {chunk_num}... ({len(chunk_labels)} samples)")
                chunk_dataset = (
                    np.array(chunk_contexts, dtype=np.float32),
                    np.array(chunk_choices, dtype=np.float32),
                    np.array(chunk_labels, dtype=np.int32)
                )
                chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_num}.pkl")
                with open(chunk_path, 'wb') as f:
                    pickle.dump(chunk_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # 次のチャンクのためにメモリを解放
                chunk_contexts, chunk_choices, chunk_labels = [], [], []
                chunk_num += 1
            
            # 定期的な進捗報告
            print(f"--- Progress: {i+1} / {total_files} files processed ---")

    print("\n--- All chunks have been generated successfully. ---")
    return True

def merge_chunks(chunk_dir, output_path):
    """
    ディスクに保存された全てのチャンクファイルを1つのデータセットに結合する。
    """
    print("\n--- Starting Chunk Merging ---")
    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.pkl")))
    
    if not chunk_files:
        print("Error: No chunk files found to merge.")
        return

    print(f"Found {len(chunk_files)} chunks to merge.")

    # 最初のチャンクをロードして初期化
    with open(chunk_files[0], 'rb') as f:
        final_contexts, final_choices, final_labels = pickle.load(f)

    # 2番目以降のチャンクを順番に結合
    for chunk_file in chunk_files[1:]:
        with open(chunk_file, 'rb') as f:
            contexts, choices, labels = pickle.load(f)
            final_contexts = np.vstack((final_contexts, contexts))
            final_choices = np.vstack((final_choices, choices))
            final_labels = np.hstack((final_labels, labels))
        print(f"Merged {os.path.basename(chunk_file)}. Total samples: {len(final_labels)}")

    # 最終データセットを保存
    print(f"\nSaving final merged dataset to '{output_path}'...")
    final_dataset = (final_contexts, final_choices, final_labels)
    with open(output_path, 'wb') as f:
        pickle.dump(final_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 一時的なチャンクファイルを削除
    print("Cleaning up temporary chunk files...")
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    os.rmdir(chunk_dir)

    print("--- Merging Complete ---")
    print(f"Final dataset saved. Total samples: {len(final_labels)}")


def main():
    DATA_DIR = '/content/data' # デフォルトはローカル
    PROCESSED_DATA_PATH = 'processed_data/training_dataset_transformer.pkl'
    CHUNK_DIR = 'processed_data/chunks'

    if generate_data_in_chunks(DATA_DIR, CHUNK_DIR):
        merge_chunks(CHUNK_DIR, PROCESSED_DATA_PATH)

if __name__ == '__main__':
    main()