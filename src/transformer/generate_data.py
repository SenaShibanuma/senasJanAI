# -*- coding: utf--8 -*-
import os
import pickle
import time
import numpy as np
import glob
from .transformer_parser import TransformerParser
from .vectorizer import preprocess_data

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
    total_samples_generated = 0
    
    # チャンクごとのデータを保持するリスト
    chunk_contexts = []
    chunk_choices = []
    chunk_labels = []

    for i, filepath in enumerate(log_files):
        try:
            raw_data_points = parser.parse_log_file(filepath)
            if raw_data_points:
                contexts, choices, labels, masks = preprocess_data(raw_data_points)
                if labels.size > 0:
                    chunk_contexts.extend(list(contexts))
                    chunk_choices.extend(list(choices))
                    chunk_labels.extend(list(labels))
        except Exception as e:
            print(f" [!] Warning: Skipping file {os.path.basename(filepath)} due to a parsing error: {e}")
            continue

        # チャンクサイズに達したか、最後のファイルであれば、チャンクを保存
        if (i + 1) % chunk_size == 0 or (i + 1) == total_files:
            # ▼▼▼【ロギング改善】ここから ▼▼▼
            if chunk_labels:
                num_samples_in_chunk = len(chunk_labels)
                total_samples_generated += num_samples_in_chunk
                print(f"  -> Saving chunk {chunk_num}... ({num_samples_in_chunk} samples found)")
                
                chunk_dataset = (
                    np.array(chunk_contexts),
                    np.array(chunk_choices),
                    np.array(chunk_labels)
                )
                chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_num}.pkl")
                with open(chunk_path, 'wb') as f:
                    pickle.dump(chunk_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                chunk_contexts, chunk_choices, chunk_labels = [], [], []
                chunk_num += 1
            else:
                # チャンクが空だった場合のメッセージを追加
                print(f"  -> No samples found in this chunk, skipping save.")
            
            print(f"--- Progress: {i+1} / {total_files} files processed ---")
            # ▲▲▲【ロギング改善】ここまで ▲▲▲

    print("\n--- Chunk generation finished. ---")
    print(f"  Total chunks created: {chunk_num - 1}")
    print(f"  Total data samples generated: {total_samples_generated}")
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
        contexts_list, choices_list, labels_list = pickle.load(f)

    # 2番目以降のチャンクを順番に結合
    for chunk_file in chunk_files[1:]:
        print(f"  -> Merging {os.path.basename(chunk_file)}...")
        with open(chunk_file, 'rb') as f:
            contexts, choices, labels = pickle.load(f)
            contexts_list = np.vstack((contexts_list, contexts))
            choices_list = np.vstack((choices_list, choices))
            labels_list = np.hstack((labels_list, labels))
        print(f"     Total samples so far: {len(labels_list)}")

    # 最終データセットを保存
    print(f"\nSaving final merged dataset to '{output_path}'...")
    masks = []
    for choices in choices_list:
        mask = np.zeros(choices.shape[0], dtype='float32')
        num_actual_choices = np.count_nonzero(np.sum(choices, axis=1))
        mask[:num_actual_choices] = 1.0
        masks.append(mask)
    
    masks_array = np.array(masks)
    final_dataset = (contexts_list, choices_list, labels_list, masks_array)
    
    with open(output_path, 'wb') as f:
        pickle.dump(final_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("\n--- Merging Complete ---")
    print(f"  Final dataset saved. Total samples: {len(labels_list)}")
    print(f"  Dataset structure (shapes):")
    print(f"    Contexts: {contexts_list.shape}")
    print(f"    Choices:  {choices_list.shape}")
    print(f"    Labels:   {labels_list.shape}")
    print(f"    Masks:    {masks_array.shape}")

    # 一時的なチャンクファイルを削除
    print("\nCleaning up temporary chunk files...")
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    os.rmdir(chunk_dir)
    print("  -> Temporary files deleted.")


def main():
    start_time = time.time()
    
    DATA_DIR = '/content/data'
    PROCESSED_DATA_PATH = 'processed_data/training_dataset_transformer.pkl'
    CHUNK_DIR = 'processed_data/chunks'

    if generate_data_in_chunks(DATA_DIR, CHUNK_DIR):
        merge_chunks(CHUNK_DIR, PROCESSED_DATA_PATH)
    
    end_time = time.time()
    print(f"\n--- Total execution time: {end_time - start_time:.2f} seconds ---")

if __name__ == '__main__':
    main()

