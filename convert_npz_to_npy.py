# -*- coding: utf-8 -*-
import os
import glob
import numpy as np

def convert_npz_to_memmap(npz_path, output_dir, batch_size=5000):
    """
    巨大な.npzファイルをチャンク（塊）ごとに読み込み、
    指定されたディレクトリ内に、単純な名前の.npyファイルとして変換・保存する。
    """
    if not os.path.exists(npz_path):
        print(f"エラー: 入力ファイルが見つかりません: {npz_path}")
        return

    print(f"'{npz_path}' を開いています...")
    # .npzファイルを遅延読み込みモードで開く
    npz_file = np.load(npz_path)
    
    keys = list(npz_file.keys())
    print(f" -> ファイル内の配列: {keys}")

    # 最初の配列から全体のサンプル数を取得
    if not keys:
        print(" -> ファイル内に配列がありません。スキップします。")
        npz_file.close()
        return
        
    total_samples = npz_file[keys[0]].shape[0]
    print(f" -> 合計サンプル数: {total_samples}")

    # 出力用のメモリマップファイルを作成
    output_arrays = {}
    for key in keys:
        # ★修正点: ファイル名を配列名のみにする（例: 'context.npy'）
        simple_key = key.replace('_input', '')
        output_path = os.path.join(output_dir, f"{simple_key}.npy")
        
        shape = (total_samples, *npz_file[key].shape[1:])
        
        # 'w+' モードで書き込み可能な空のファイルを作成
        output_arrays[key] = np.memmap(
            output_path, 
            dtype=npz_file[key].dtype, 
            mode='w+', 
            shape=shape
        )
        print(f" -> 出力ファイルを作成: {output_path}")

    # データをバッチ単位でコピー
    print(f"\nバッチサイズ {batch_size} でデータの変換を開始します...")
    for i in range(0, total_samples, batch_size):
        start_index = i
        end_index = min(i + batch_size, total_samples)
        
        print(f" -> サンプル {start_index} から {end_index-1} を処理中...", end='\r')
        
        for key in keys:
            # ここで.npzファイルから一部だけを読み込む
            chunk = npz_file[key][start_index:end_index]
            # メモリマップファイルに書き込む
            output_arrays[key][start_index:end_index] = chunk

    # 変更をディスクに確実に書き込む
    for arr in output_arrays.values():
        arr.flush()
        
    npz_file.close()
    print("\n\n変換が正常に完了しました。")

def main():
    # --- 設定項目 ---
    # 変換したい.npzファイルが保存されているディレクトリ
    SOURCE_DIR = 'data/processed'
    # ----------------
    
    # SOURCE_DIR内のすべての.npzファイルを検索
    npz_files = glob.glob(os.path.join(SOURCE_DIR, '*.npz'))
    
    if not npz_files:
        print(f"エラー: '{SOURCE_DIR}' 内に.npzファイルが見つかりません。")
        return

    print(f"'{SOURCE_DIR}' 内で {len(npz_files)} 個の .npz ファイルが見つかりました。変換を開始します。")
    
    for npz_path in npz_files:
        print("-" * 50)
        
        # ★修正点: train, test ごとにサブディレクトリを作成して、そこに結果を保存する
        # 例: "train.npz" -> "data/processed/train/" ディレクトリを作成
        basename = os.path.splitext(os.path.basename(npz_path))[0]
        output_subdir = os.path.join(SOURCE_DIR, basename)
        os.makedirs(output_subdir, exist_ok=True)
        print(f"出力先フォルダ: {output_subdir}")
        
        convert_npz_to_memmap(npz_path, output_subdir)

    print("\n" + "="*50)
    print("すべての変換処理が完了しました。")
    print(f"'{SOURCE_DIR}' 内に train, test, validation フォルダが作成されているか確認してください。")
    print("今後は、これらのフォルダ内の .npy ファイルを使って学習を進めてください。")

if __name__ == '__main__':
    main()

