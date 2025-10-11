# -*- coding: utf--8 -*-
import os
import pickle
import time
import numpy as np
import tensorflow as tf
import glob
import sys
from multiprocessing import Pool, cpu_count

# 循環参照を避けるため、パーサーとベクトライザを直接インポートしない
# from .transformer_parser import TransformerParser
# from .vectorizer import vectorize_event, vectorize_choice, MAX_CONTEXT_LENGTH, MAX_CHOICES

# 代わりに、ワーカープロセス内で動的にインポートする
def initialize_worker():
    """ワーカープロセスごとに必要なモジュールをインポートする"""
    global TransformerParser, vectorize_event, vectorize_choice, MAX_CONTEXT_LENGTH, MAX_CHOICES
    from .transformer_parser import TransformerParser
    from .vectorizer import vectorize_event, vectorize_choice, MAX_CONTEXT_LENGTH, MAX_CHOICES

def process_file_chunk(args):
    """
    ワーカープロセス用の関数。
    ファイルのリストとパス情報を受け取り、結果を「ローカル」のチャンクファイルとして保存する。
    """
    file_paths, chunk_index, local_chunk_dir = args
    parser = TransformerParser()
    contexts_list, choices_list, labels_list = [], [], []
    processed_paths_in_chunk = []

    for filepath in file_paths:
        try:
            training_data = parser.parse_log_file(filepath)
            if not training_data: 
                processed_paths_in_chunk.append(filepath)
                continue
            for data_point in training_data:
                player_pov = data_point.get('player_pov')
                if player_pov is None: continue
                try:
                    label_index = data_point['choices'].index(data_point['label'])
                except ValueError: continue
                context_vecs = [vectorize_event(e, player_pov) for e in data_point['context']]
                choices_vecs = [vectorize_choice(c) for c in data_point['choices']]
                contexts_list.append(context_vecs)
                choices_list.append(choices_vecs)
                labels_list.append(label_index)
            processed_paths_in_chunk.append(filepath)
        except Exception:
            continue
            
    if labels_list:
        padded_contexts = tf.keras.preprocessing.sequence.pad_sequences(contexts_list, maxlen=MAX_CONTEXT_LENGTH, dtype='float32', padding='post', truncating='post')
        padded_choices = tf.keras.preprocessing.sequence.pad_sequences(choices_list, maxlen=MAX_CHOICES, dtype='float32', padding='post')
        labels_array = np.array(labels_list, dtype=np.int32)
        
        # チャンクファイルを一意のIDでローカルに保存
        local_chunk_path = os.path.join(local_chunk_dir, f"chunk_{os.getpid()}_{chunk_index}.pkl")
        with open(local_chunk_path, 'wb') as f:
            pickle.dump((padded_contexts, padded_choices, labels_array), f)

    return processed_paths_in_chunk

def main():
    start_time = time.time()
    
    # --- 全ての処理をColabの高速なローカルストレージで完結させる ---
    DATA_DIR = '/content/data' 
    PROCESSED_DIR = '/content/processed_data'
    CHUNK_DIR = os.path.join(PROCESSED_DIR, 'chunks')
    TRACKER_FILE = os.path.join(PROCESSED_DIR, 'processed_files.log')
    
    os.makedirs(CHUNK_DIR, exist_ok=True)
    
    # --- ファイルリストの比較 ---
    all_log_files = set(glob.glob(os.path.join(DATA_DIR, "*.mjlog"))) | set(glob.glob(os.path.join(DATA_DIR, "*.gz")))
    processed_files = set()
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, 'r') as f:
            processed_files = set(line.strip() for line in f)

    files_to_process = sorted(list(all_log_files - processed_files))
    
    if files_to_process:
        num_cores = cpu_count()
        print(f"--- Resumable Data Generation (Local Storage Only) ---")
        print(f"Total: {len(all_log_files)} | Processed: {len(processed_files)} | Remaining: {len(files_to_process)}")
        print(f"Starting parallel processing on {num_cores} CPU cores...")

        sub_chunk_size = 100
        file_chunks = [files_to_process[i:i + sub_chunk_size] for i in range(0, len(files_to_process), sub_chunk_size)]
        tasks = [(chunk, i, CHUNK_DIR) for i, chunk in enumerate(file_chunks)]

        processed_count = 0
        progbar = tf.keras.utils.Progbar(len(files_to_process), unit_name="file")

        # initializerを設定して、各ワーカープロセスでモジュールをインポート
        with Pool(processes=num_cores, initializer=initialize_worker) as pool, open(TRACKER_FILE, 'a') as tracker:
            for processed_paths in pool.imap_unordered(process_file_chunk, tasks):
                for path in processed_paths:
                    tracker.write(path + '\n')
                processed_count += len(processed_paths)
                progbar.update(processed_count)
        print(f"\n--- Data Generation task finished for this session. ---")
    else:
        print("--- All files have been processed. You can now proceed to the 'Sync to Drive' step in your notebook. ---")

    end_time = time.time()
    print(f"\nTotal execution time for this run: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()

