# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
import os
import pickle
from .transformer_parser import TransformerParser

# --- グローバル定数 ---
MAX_CONTEXT_LENGTH = 150 # 1局の最大イベント数（パディング用）
MAX_CHOICES = 50       # 1回の判断での最大選択肢数（パディング用）
EMBEDDING_DIM = 128    # イベント/アクションを表現するベクトルの次元数

def vectorize_event(event):
    """
    イベント辞書を固定長のベクトルに変換する（簡易版）
    TODO: 仕様書に基づき、より詳細なベクトル化を実装する
    """
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    # 簡単なハッシュを使ってIDをベクトルに割り振る
    vec[hash(event.get('event_id', '')) % EMBEDDING_DIM] = 1.0
    if 'player' in event:
        vec[hash(f"player_{event['player']}") % EMBEDDING_DIM] = 1.0
    if 'tile' in event:
        vec[hash(f"tile_{event['tile']}") % EMBEDDING_DIM] = 1.0
    return vec

def vectorize_choice(choice_str):
    """
    選択肢の文字列を固定長のベクトルに変換する（簡易版）
    TODO: 仕様書に基づき、より詳細なベクトル化を実装する
    """
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vec[hash(choice_str) % EMBEDDING_DIM] = 1.0
    return vec

def preprocess_data(training_data):
    """
    パーサーから受け取ったデータを、Transformerモデルの入力形式に変換・パディングする
    """
    contexts = []
    choices_list = []
    labels = []

    for data_point in training_data:
        context_vecs = [vectorize_event(e) for e in data_point['context']]
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

    padded_contexts = tf.keras.preprocessing.sequence.pad_sequences(
        contexts, maxlen=MAX_CONTEXT_LENGTH, dtype='float32', padding='post', truncating='post'
    )
    padded_choices = tf.keras.preprocessing.sequence.pad_sequences(
        choices_list, maxlen=MAX_CHOICES, dtype='float32', padding='post'
    )
    
    return padded_contexts, padded_choices, np.array(labels, dtype=np.int32)


def build_parallel_evaluation_transformer(
    context_len=MAX_CONTEXT_LENGTH, 
    choices_len=MAX_CHOICES, 
    embed_dim=EMBEDDING_DIM,
    num_heads=4, # 軽量化のためヘッド数を削減
    ff_dim=256, 
    num_transformer_blocks=2): # 軽量化のためブロック数を削減
    """
    「並列評価モデル」のアーキテクチャでTransformerを構築する
    """
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

def prepare_dataset(data_dir, processed_data_path):
    """データセットを準備する（読み込み or 生成）"""
    if os.path.exists(processed_data_path):
        print(f"Loading processed dataset from {processed_data_path}...")
        with open(processed_data_path, 'rb') as f:
            return pickle.load(f)

    print(f"Processed dataset not found. Generating from logs in {data_dir}...")
    parser = TransformerParser()
    log_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mjlog') or f.endswith('.gz')]
    
    if not log_files:
        raise FileNotFoundError(f"No log files found in {data_dir}")

    training_data = parser.run(log_files)
    
    if not training_data:
        raise ValueError("No data could be extracted from the log files.")

    print("Preprocessing data for the model...")
    processed_dataset = preprocess_data(training_data)
    
    print(f"Saving processed dataset to {processed_data_path}...")
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    with open(processed_data_path, 'wb') as f:
        pickle.dump(processed_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return processed_dataset

def main():
    """メインの実行関数"""
    DATA_DIR = 'data'
    PROCESSED_DATA_PATH = 'processed_data/training_dataset_transformer.pkl'
    MODEL_PATH = 'models/senas_jan_ai_transformer_v1.keras'

    try:
        contexts, choices, labels = prepare_dataset(DATA_DIR, PROCESSED_DATA_PATH)
    except (FileNotFoundError, ValueError) as e:
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

