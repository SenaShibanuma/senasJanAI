# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
import os
import pickle
# データ生成が別ファイルになったため、TransformerParserの直接インポートは不要

# --- グローバル定数 ---
MAX_CONTEXT_LENGTH = 150
MAX_CHOICES = 50
VECTOR_DIM = 100

def vectorize_event(event, player_pov):
    vec = np.zeros(VECTOR_DIM, dtype=np.float32)
    event_ids = {'GAME_START': 1, 'INIT': 2, 'DRAW': 3, 'DISCARD': 4, 'MELD': 5, 'RIICHI_DECLARED': 6, 'RIICHI_ACCEPTED': 7, 'NEW_DORA': 8, 'AGARI': 9, 'RYUKYOKU': 10}
    vec[0] = event_ids.get(event.get('event_id'), 0)
    player = event.get('player', -1)
    if player != -1:
        relative_player = (player - player_pov + 4) % 4
        vec[1] = relative_player
    vec[2] = event.get('turn_num', 0) / 30.0
    scores = event.get('scores', [25000]*4)
    for i in range(4):
        vec[3 + i] = (scores[(player_pov + i) % 4] - 25000) / 10000.0
    vec[7] = event.get('riichi_sticks', 0)
    vec[8] = event.get('remaining_tiles', 70) / 70.0
    event_id = event.get('event_id')
    if event_id == 'INIT':
        vec[10] = event.get('round', 0)
        vec[11] = event.get('honba', 0)
        vec[12] = event.get('dora_indicator', 0)
    elif event_id == 'DRAW':
        vec[10] = event.get('tile', 0)
    elif event_id == 'DISCARD':
        vec[10] = event.get('tile', 0)
        vec[11] = 1.0 if event.get('is_tedashi') else 0.0
    return vec

def vectorize_choice(choice_str):
    vec = np.zeros(VECTOR_DIM, dtype=np.float32)
    parts = choice_str.split('_')
    action = parts[0]
    action_ids = {'DISCARD': 1, 'ACTION': 2}
    if action == 'DISCARD':
        vec[0] = action_ids['DISCARD']
        vec[1] = int(parts[1])
    elif action == 'ACTION':
        action_type = parts[1]
        action_type_ids = {'TSUMO': 1, 'RON': 2, 'RIICHI': 3, 'PUNG': 4, 'CHII': 5, 'DAIMINKAN': 6, 'PASS': 7}
        sub_id = action_type_ids.get(action_type, 0)
        vec[0] = action_ids['ACTION']
        vec[1] = sub_id
        if action_type == 'RIICHI':
            vec[2] = int(parts[2])
        elif action_type == 'CHII':
            vec[2] = int(parts[2])
            vec[3] = int(parts[3])
    return vec

def preprocess_data(training_data):
    contexts = []
    choices_list = []
    labels = []
    for data_point in training_data:
        player_pov = -1
        for event in reversed(data_point['context']):
            if event.get('event_id') == 'DRAW' and 'player' in event:
                player_pov = event['player']
                break
        if player_pov == -1: continue
        context_vecs = [vectorize_event(e, player_pov) for e in data_point['context']]
        contexts.append(context_vecs)
        choice_vecs = [vectorize_choice(c) for c in data_point['choices']]
        choices_list.append(choice_vecs)
        try:
            label_index = data_point['choices'].index(data_point['label'])
            labels.append(label_index)
        except (ValueError, KeyError):
            contexts.pop()
            choices_list.pop()
            continue
    padded_contexts = tf.keras.preprocessing.sequence.pad_sequences(contexts, maxlen=MAX_CONTEXT_LENGTH, dtype='float32', padding='post', truncating='post')
    padded_choices = tf.keras.preprocessing.sequence.pad_sequences(choices_list, maxlen=MAX_CHOICES, dtype='float32', padding='post')
    return padded_contexts, padded_choices, np.array(labels, dtype=np.int32)

def build_parallel_evaluation_transformer(context_len=MAX_CONTEXT_LENGTH, choices_len=MAX_CHOICES, embed_dim=VECTOR_DIM, num_heads=4, ff_dim=256, num_transformer_blocks=2):
    context_input = layers.Input(shape=(context_len, embed_dim), name="context_input")
    choices_input = layers.Input(shape=(choices_len, embed_dim), name="choices_input")
    x = context_input
    for _ in range(num_transformer_blocks):
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        ffn_output = layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = layers.Dense(embed_dim)(ffn_output)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    context_vector = layers.GlobalAveragePooling1D()(x)
    context_repeated = layers.RepeatVector(choices_len)(context_vector)
    merged = layers.Concatenate(axis=-1)([context_repeated, choices_input])
    score_head = layers.Dense(128, activation="relu")(merged)
    score_head = layers.Dropout(0.3)(score_head)
    output_scores = layers.Dense(1, name="output_scores")(score_head)
    output_scores = layers.Reshape((choices_len,))(output_scores)
    model = models.Model(inputs=[context_input, choices_input], outputs=output_scores)
    return model

def load_processed_dataset(processed_data_path):
    """
    【修正版】前処理済みのデータセットをファイルから読み込む
    """
    if not os.path.exists(processed_data_path):
        # ファイルが存在しない場合はエラーメッセージを表示して終了
        raise FileNotFoundError(
            f"Processed dataset not found at '{processed_data_path}'. "
            "Please run 'generate_data.py' first to create the dataset."
        )
        
    print(f"Loading processed dataset from {processed_data_path}...")
    with open(processed_data_path, 'rb') as f:
        return pickle.load(f)

def main():
    """
    【修正版】学習プロセスのみを実行する
    """
    PROCESSED_DATA_PATH = 'processed_data/training_dataset_transformer.pkl'
    MODEL_PATH = 'models/senas_jan_ai_transformer_v1.keras'

    try:
        # データ生成は行わず、常にファイルから読み込む
        contexts, choices, labels = load_processed_dataset(PROCESSED_DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    print(f"\nData ready. Number of samples: {len(labels)}")
    print(f"Contexts shape: {contexts.shape}")
    print(f"Choices shape: {choices.shape}")
    print(f"Labels shape: {labels.shape}")

    print("\n--- Building the Transformer model... ---")
    model = build_parallel_evaluation_transformer()
    model.summary()

    print("\n--- Training the Transformer model... ---")
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss=loss_fn,
                  metrics=['accuracy'])

    callbacks_list = [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        callbacks.ModelCheckpoint(filepath=MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    X_train_ctx, X_val_ctx, X_train_cho, X_val_cho, y_train, y_val = train_test_split(
        contexts, choices, labels, test_size=0.1, random_state=42
    )

    model.fit(
        [X_train_ctx, X_train_cho], 
        y_train,
        epochs=30,
        batch_size=64,
        validation_data=([X_val_ctx, X_val_cho], y_val),
        callbacks=callbacks_list
    )

    print(f"\n--- Finished Training. Best model saved to {MODEL_PATH} ---")

if __name__ == '__main__':
    main()