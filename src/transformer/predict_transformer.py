# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
# ベクトライザから関数と定数をインポート
from .vectorizer import vectorize_context, vectorize_choice, MAX_CONTEXT_LENGTH, MAX_CHOICES

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
    # (例：東1局5巡目、自分がリーチをかけるべきか悩んでいる状況)
    pov_player = 0
    sample_context_events = [
        {'type': 'start_kyoku', 'bakaze': 'E', 'kyoku': 1, 'honba': 0, 'kyotaku': 0, 'oya': 0, 'scores': [25000, 25000, 25000, 25000], 'dora_marker': 4},
        {'type': 'haipai', 'actor': 0, 'pai': ['1m', '2m', '3m', '4p', '5p', '6p', '7s', '8s', '9s', 'N', 'N', 'C', 'C']},
        {'type': 'tsumo', 'actor': 0, 'pai': 'N'},
        {'type': 'dahai', 'actor': 0, 'pai': 'C'},
        # ... (中略) ...
        {'type': 'tsumo', 'actor': 0, 'pai': '6s'},
    ]
    sample_choices_str = [
        "DAHAI_8",   # 2s
        "DAHAI_24",  # 6s (ツモ切り)
        "REACH_8",   # 2s切りリーチ
        "REACH_24"   # 6s切りリーチ
    ]
    
    # --- ここから下は編集不要 ---
    
    # データをベクトル化
    context_vec = vectorize_context(sample_context_events, pov_player)
    choice_vecs = [vectorize_choice(c) for c in sample_choices_str]
    
    # パディングとNumpy配列化
    padded_context = tf.keras.preprocessing.sequence.pad_sequences(
        [context_vec], maxlen=MAX_CONTEXT_LENGTH, dtype='float32', padding='post'
    )
    padded_choices = tf.keras.preprocessing.sequence.pad_sequences(
        [choice_vecs], maxlen=MAX_CHOICES, dtype='float32', padding='post'
    )
    prediction_mask = np.zeros((1, MAX_CHOICES), dtype='float32')
    prediction_mask[0, :len(sample_choices_str)] = 1.0

    # 予測を実行 (入力は3つの要素を持つリスト)
    model_inputs = [padded_context, padded_choices, prediction_mask]
    predictions = model.predict(model_inputs)
    
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
