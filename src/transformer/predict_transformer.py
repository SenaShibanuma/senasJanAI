# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from .train_transformer import vectorize_event, vectorize_choice, MAX_CONTEXT_LENGTH, MAX_CHOICES

def main():
    """
    学習済みのTransformerモデルを読み込み、
    与えられた盤面状況と選択肢から最適な行動を予測する。
    """
    MODEL_PATH = 'models/senas_jan_ai_transformer_v1.keras'
    print(f"--- Loading a trained Transformer model from {MODEL_PATH}... ---")

    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Please run train_transformer.py first.")
        return
        
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("\n--- Preparing sample data for prediction ---")
    
    # --- 予測したい状況をここに設定 ---
    
    # 1. 文脈(Context)をイベントシーケンスとして作成
    # (実際のパーサーから取得する必要があるが、ここではダミーを作成)
    sample_context = [
        {'event_id': 'INIT', 'round': 0, 'honba': 0, 'dora_indicator': 0},
        {'event_id': 'DRAW', 'player': 0, 'tile': 10},
        # ... 多くのイベントが続く ...
    ]
    
    # 2. その瞬間に可能な「選択肢(Action Space)」のリストを作成
    sample_choices_str = [
        "DISCARD_27_False", # 捨てる: 東 (赤ではない)
        "DISCARD_30_False", # 捨てる: 白
        "ACTION_RIICHI"
    ]
    
    # --- ここから下は編集不要 ---
    
    # データをモデルの入力形式に変換
    context_vecs = [vectorize_event(e) for e in sample_context]
    choice_vecs = [vectorize_choice(c) for c in sample_choices_str]
    
    padded_context = tf.keras.preprocessing.sequence.pad_sequences(
        [context_vecs], maxlen=MAX_CONTEXT_LENGTH, dtype='float32', padding='post'
    )
    padded_choices = tf.keras.preprocessing.sequence.pad_sequences(
        [choice_vecs], maxlen=MAX_CHOICES, dtype='float32', padding='post'
    )

    # 予測を実行
    predictions = model.predict([padded_context, padded_choices])
    
    # 予測されたスコアを取得 (パディング分を除外)
    scores = predictions[0][:len(sample_choices_str)]
    
    # スコアが高い順にソート
    top_indices = np.argsort(scores)[::-1]

    print("\n--- AI's Recommendations (Ranked) ---")
    for i, choice_idx in enumerate(top_indices):
        choice_str = sample_choices_str[choice_idx]
        score = scores[choice_idx]
        print(f"{i+1}. Action: {choice_str:<25} (Score: {score:.4f})")


if __name__ == '__main__':
    main()
