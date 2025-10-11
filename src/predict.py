# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from mahjong.shanten import Shanten

# --- 定数と牌の変換用データ ---
TILE_TYPES = [
    '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
    '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
    '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
    'E', 'S', 'W', 'N', 'H', 'G', 'C'
]
TILE_MAP = {s: i for i, s in enumerate(TILE_TYPES)}

def parse_hand_string(hand_str):
    tiles = []
    current_suit = ''
    for char in reversed(hand_str):
        if char.isalpha():
            current_suit = char
            if current_suit.upper() in 'ESWNHGC':
                tile_name = current_suit.upper()
                if tile_name in TILE_MAP:
                    tiles.append(TILE_MAP[tile_name])
        elif char.isdigit():
            suit_map = {'m': 'm', 'p': 'p', 's': 's'}
            if current_suit in suit_map:
                tile_name = char + current_suit
                if tile_name in TILE_TYPES:
                    tiles.append(TILE_MAP[tile_name])
    return tiles

def create_feature_tensor_for_prediction(hand_tiles, dora_indicators, rivers, is_riichi_list):
    """予測用の単一の盤面状態から入力テンソルを生成する"""
    NUM_CHANNELS = 15
    NUM_TILE_TYPES = 34
    tensor = np.zeros((NUM_CHANNELS, NUM_TILE_TYPES), dtype=np.float32)

    # ch0: 自分の手牌
    for tile_type in hand_tiles:
        tensor[0, tile_type] += 1
    # ch1: 自分の河
    for tile_type in rivers[0]:
        tensor[1, tile_type] = 1
    # ch3: 自分のリーチ状態
    if is_riichi_list[0]:
        tensor[3, :] = 1

    # 他家3人の情報
    for i in range(1, 4):
        ch_offset = 4 + (i - 1) * 3
        for tile_type in rivers[i]:
            tensor[ch_offset, tile_type] = 1
        if is_riichi_list[i]:
            tensor[ch_offset + 2, :] = 1

    # ch13: ドラ表示牌
    for tile_type in dora_indicators:
         tensor[13, tile_type] = 1
    
    # ch14: シャンテン数
    shanten_calculator = Shanten()
    hand_counts = [0] * 34
    for tile_type in hand_tiles:
        hand_counts[tile_type] += 1
    
    num_tiles = sum(hand_counts)
    if num_tiles % 3 == 2:
        shanten_val = shanten_calculator.calculate_shanten(hand_counts)
        tensor[14, :] = min(shanten_val, 8) / 8.0
             
    return tensor

def main():
    """メインの実行関数"""
    model_path = os.path.join('models', 'senas_jan_ai_v3.keras')
    print("--- Loading a trained model (v3)... ---")
    
    if not os.path.exists(model_path):
        print(f"Error loading model: File not found at {model_path}")
        print("Please run train_model.py to train and save the model first.")
        return

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # --- 予測したい状況をここに設定 ---
    test_hand_str = "123m456p789sEESSN"
    dora_indicators_str = "1m"
    rivers_str = ["", "W", "", "9s"]
    is_riichi_list = [False, False, False, False]
    
    print(f"\nAnalyzing hand: {test_hand_str}")
    
    hand_tiles = parse_hand_string(test_hand_str)
    dora_indicators = parse_hand_string(dora_indicators_str)
    rivers = [parse_hand_string(r) for r in rivers_str]

    feature_tensor = create_feature_tensor_for_prediction(hand_tiles, dora_indicators, rivers, is_riichi_list)
    input_tensor = np.expand_dims(feature_tensor, axis=0)

    predictions = model.predict(input_tensor)
    
    top_5_indices = np.argsort(predictions[0])[::-1][:5]

    print("\n--- AI's Recommendations (Top 5) ---")
    for i, tile_index in enumerate(top_5_indices):
        discard_tile = TILE_TYPES[tile_index]
        confidence = predictions[0][tile_index] * 100
        print(f"{i+1}. Discard: {discard_tile:<4} (Confidence: {confidence:.2f}%)")

if __name__ == '__main__':
    main()

