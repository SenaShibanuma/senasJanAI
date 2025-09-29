# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

# --- グローバル定数 (プロジェクト全体で共有) ---
MAX_CONTEXT_LENGTH = 150
MAX_CHOICES = 50
VECTOR_DIM = 100

def vectorize_event(event, player_pov):
    """単一のイベント辞書を固定長のベクトルに変換する"""
    vec = np.zeros(VECTOR_DIM, dtype=np.float32)
    event_ids = {'GAME_START': 1, 'INIT': 2, 'DRAW': 3, 'DISCARD': 4, 'MELD': 5, 'RIICHI_DECLARED': 6, 'RIICHI_ACCEPTED': 7, 'NEW_DORA': 8, 'AGARI': 9, 'RYUUKYOKU': 10}
    vec[0] = event_ids.get(event.get('event_id'), 0)
    player = event.get('player', -1)
    if player != -1:
        relative_player = (player - player_pov + 4) % 4
        vec[1] = relative_player
    vec[2] = event.get('turn_num', 0) / 30.0
    scores = event.get('scores', [25000]*4)
    for i in range(4): vec[3 + i] = (scores[(player_pov + i) % 4] - 25000) / 10000.0
    vec[7] = event.get('riichi_sticks', 0)
    vec[8] = event.get('remaining_tiles', 70) / 70.0
    event_id = event.get('event_id')
    if event_id == 'INIT':
        vec[10], vec[11], vec[12] = event.get('round', 0), event.get('honba', 0), event.get('dora_indicator', 0)
    elif event_id == 'DRAW' or event_id == 'DISCARD':
        vec[10] = event.get('tile', 0)
        if event_id == 'DISCARD': vec[11] = 1.0 if event.get('is_tedashi') else 0.0
    return vec

def vectorize_choice(choice_str):
    """単一の選択肢文字列を固定長のベクトルに変換する"""
    vec = np.zeros(VECTOR_DIM, dtype=np.float32)
    parts = choice_str.split('_'); action = parts[0]
    if action == 'DISCARD':
        vec[0], vec[1] = 1, int(parts[1])
    elif action == 'ACTION':
        vec[0] = 2
        action_type = parts[1]
        action_type_ids = {'TSUMO': 1, 'RON': 2, 'RIICHI': 3, 'PUNG': 4, 'CHII': 5, 'DAIMINKAN': 6, 'PASS': 7}
        vec[1] = action_type_ids.get(action_type, 0)
        if action_type == 'RIICHI': vec[2] = int(parts[2])
        elif action_type == 'CHII': vec[2], vec[3] = int(parts[2]), int(parts[3])
    return vec

def preprocess_data(training_data):
    """
    パーサーから受け取った生データを、AIモデルが学習できる形式に前処理する。
    マスキング用のデータも同時に生成する。
    """
    contexts, choices_list, labels, masks = [], [], [], []
    for data_point in training_data:
        player_pov = data_point.get('player_pov')
        if player_pov is None: continue
        
        # 正解ラベルが選択肢リスト内に存在することを確認
        try:
            label_index = data_point['choices'].index(data_point['label'])
        except ValueError:
            continue # ラベルが存在しないデータはスキップ

        contexts.append([vectorize_event(e, player_pov) for e in data_point['context']])
        choices_list.append([vectorize_choice(c) for c in data_point['choices']])
        labels.append(label_index)

    # パディング処理
    padded_contexts = tf.keras.preprocessing.sequence.pad_sequences(
        contexts, maxlen=MAX_CONTEXT_LENGTH, dtype='float32', padding='post', truncating='post'
    )
    padded_choices = tf.keras.preprocessing.sequence.pad_sequences(
        choices_list, maxlen=MAX_CHOICES, dtype='float32', padding='post'
    )
    
    # マスク生成: 実際の選択肢がある部分は1.0、パディング部分は0.0
    for choices in choices_list:
        mask = np.zeros(MAX_CHOICES, dtype='float32')
        mask[:len(choices)] = 1.0
        masks.append(mask)

    return padded_contexts, padded_choices, np.array(labels, dtype=np.int32), np.array(masks, dtype=np.float32)
