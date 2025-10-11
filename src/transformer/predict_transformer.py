# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
from .vectorizer import vectorize_context, vectorize_choice, MAX_CONTEXT_LENGTH, MAX_CHOICES

def main():
    """
    学習済みのTransformerモデルを読み込み、
    与えられた盤面状況と選択肢から最適な行動を予測する。
    """
    MODEL_PATH = '/content/models/senas_jan_ai_transformer_v1.keras'
    print(f"--- Loading a trained Transformer model from {MODEL_PATH}... ---")

    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Please run training first.")
        return
        
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("\n--- Preparing sample data for prediction ---")
    
    pov_player = 0
    sample_context_events = [
        {'event_id': 'INIT', 'round': 0, 'honba': 0, 'riichi_sticks': 0, 'scores': [25000, 25000, 25000, 25000], 'dora_indicator': 4},
        {'event_id': 'DRAW', 'player': 0, 'tile': 135},{'event_id': 'DISCARD', 'player': 0, 'tile': 134},
        {'event_id': 'DRAW', 'player': 1, 'tile': 50},{'event_id': 'DISCARD', 'player': 1, 'tile': 51},
        {'event_id': 'DRAW', 'player': 0, 'tile': 24},
    ]
    sample_choices_str = ["DISCARD_8", "DISCARD_24", "ACTION_RIICHI_8", "ACTION_RIICHI_24"]
    
    context_vec = vectorize_context(sample_context_events, pov_player)
    choice_vecs = [vectorize_choice(c) for c in sample_choices_str]
    
    padded_context = tf.keras.preprocessing.sequence.pad_sequences(
        [context_vec], maxlen=MAX_CONTEXT_LENGTH, dtype='float32', padding='post'
    )
    padded_choices = tf.keras.preprocessing.sequence.pad_sequences(
        [choice_vecs], maxlen=MAX_CHOICES, dtype='float32', padding='post'
    )
    prediction_mask = np.zeros((1, MAX_CHOICES), dtype='float32')
    prediction_mask[0, :len(sample_choices_str)] = 1.0

    model_inputs = [padded_context, padded_choices, prediction_mask]
    predictions = model.predict(model_inputs)
    raw_scores = predictions[0][:len(sample_choices_str)]
    
    def softmax(x):
        e_x = np.exp(x - np.max(x)); return e_x / e_x.sum(axis=0)

    probabilities = softmax(raw_scores)
    top_indices = np.argsort(probabilities)[::-1]

    print("\n--- AI's Recommendations (Ranked) ---")
    for i, choice_idx in enumerate(top_indices):
        choice_str = sample_choices_str[choice_idx]
        probability = probabilities[choice_idx]
        print(f"{i+1}. Action: {choice_str:<25} (Probability: {probability:.2%})")

if __name__ == '__main__':
    main()

