# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
# 整理されたvectorizerから定数と関数をインポート
from .vectorizer import vectorize_event, vectorize_choice, MAX_CONTEXT_LENGTH, MAX_CHOICES

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
    player_pov_id = 0
    sample_context = [
        {'event_id': 'INIT', 'round': 0, 'honba': 0, 'dora_indicator': 5, 'scores': [25000]*4, 'riichi_sticks': 0, 'remaining_tiles': 70, 'turn_num': 0},
        {'event_id': 'DRAW', 'player': 0, 'tile': 50, 'turn_num': 1, 'scores': [25000]*4, 'riichi_sticks': 0, 'remaining_tiles': 69},
        {'event_id': 'DISCARD', 'player': 0, 'tile': 20, 'is_tedashi': True, 'turn_num': 1, 'scores': [25000]*4, 'riichi_sticks': 0, 'remaining_tiles': 69},
        {'event_id': 'DRAW', 'player': 0, 'tile': 100, 'turn_num': 5, 'scores': [24000, 25000, 26000, 25000], 'riichi_sticks': 1, 'remaining_tiles': 55},
    ]
    sample_choices_str = [
        "DISCARD_27", 
        "DISCARD_30",
        "DISCARD_100",
        "ACTION_RIICHI_30"
    ]
    
    # --- ここから下は編集不要 ---
    
    # データをベクトル化
    context_vecs = [vectorize_event(e, player_pov_id) for e in sample_context]
    choice_vecs = [vectorize_choice(c) for c in sample_choices_str]
    
    # パディング処理
    padded_context = tf.keras.preprocessing.sequence.pad_sequences(
        [context_vecs], maxlen=MAX_CONTEXT_LENGTH, dtype='float32', padding='post'
    )
    padded_choices = tf.keras.preprocessing.sequence.pad_sequences(
        [choice_vecs], maxlen=MAX_CHOICES, dtype='float32', padding='post'
    )

    # ▼▼▼【重要】予測時にもマスクを作成する ▼▼▼
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
