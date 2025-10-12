# -*- coding: utf-8 -*-
import os
import numpy as np
import shutil

def combine_batches(temp_dir, processed_dir, dataset_name):
    """
    指定された一時ディレクトリ内のバッチ.npzファイルを結合し、
    メモリマップ可能な.npyファイルとして保存する。
    """
    batch_files = sorted(
        [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.npz')],
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )
    
    if not batch_files:
        print(f"警告: {temp_dir} にバッチファイルが見つかりません。スキップします。")
        return

    print(f"'{dataset_name}' データセットのバッチを結合しています... ({len(batch_files)}個のファイル)")

    # 最初のバッチファイルから配列の形状とキーを決定する
    with np.load(batch_files[0]) as data:
        keys = list(data.keys())
        shapes = {key: data[key].shape[1:] for key in keys}

    # 全サンプル数を計算
    total_samples = 0
    for f in batch_files:
        with np.load(f) as data:
            # いずれか一つのキーを使ってサンプル数を加算（すべての配列で同じはず）
            total_samples += len(data[keys[0]])

    print(f" -> 合計サンプル数: {total_samples}")

    # 出力用のメモリマップファイルを作成
    output_arrays = {}
    for key in keys:
        output_path = os.path.join(processed_dir, f"{dataset_name}_{key}.npy")
        output_arrays[key] = np.memmap(
            output_path, 
            dtype=np.float32, 
            mode='w+', 
            shape=(total_samples, *shapes[key])
        )

    # バッチを順番に書き込んでいく
    current_pos = 0
    for f in batch_files:
        with np.load(f) as data:
            batch_size = len(data[keys[0]])
            for key in keys:
                output_arrays[key][current_pos:current_pos + batch_size] = data[key]
            current_pos += batch_size
    
    # 変更をディスクに書き込む
    for arr in output_arrays.values():
        arr.flush()

    print(f" -> {processed_dir} に .npy ファイルを保存しました。")


def main():
    PROCESSED_DIR = 'processed_data'
    
    # train, validation, test の各データセットに対して処理を実行
    for dataset_name in ['train', 'validation', 'test']:
        temp_dir = os.path.join(PROCESSED_DIR, f'temp_{dataset_name}')
        
        if not os.path.exists(temp_dir):
            print(f"一時ディレクトリ '{temp_dir}' が見つかりません。スキップします。")
            continue
        
        combine_batches(temp_dir, PROCESSED_DIR, dataset_name)
        
        # 変換が終わったら一時ディレクトリを削除
        try:
            shutil.rmtree(temp_dir)
            print(f" -> 一時ディレクトリ {temp_dir} をクリーンアップしました。")
        except OSError as e:
            print(f" -> 一時ディレクトリのクリーンアップ中にエラー: {e}")
            
    print("\nすべての処理が完了しました。")

if __name__ == '__main__':
    main()
