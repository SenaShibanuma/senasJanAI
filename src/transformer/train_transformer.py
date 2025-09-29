# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
import os
import pickle
# ベクトライザから定数をインポート
from .vectorizer import MAX_CONTEXT_LENGTH, MAX_CHOICES, VECTOR_DIM

def build_masked_transformer(context_len=MAX_CONTEXT_LENGTH, choices_len=MAX_CHOICES, embed_dim=VECTOR_DIM, num_heads=4, ff_dim=256, num_transformer_blocks=2):
    """
    選択肢をマスキングする機能を持つTransformerモデルを構築する。
    """
    # --- 3種類の入力層を定義 ---
    context_input = layers.Input(shape=(context_len, embed_dim), name="context_input")
    choices_input = layers.Input(shape=(choices_len, embed_dim), name="choices_input")
    mask_input = layers.Input(shape=(choices_len,), name="mask_input") # マスク用入力 (1.0=有効, 0.0=無効)

    # --- Context Encoder (Transformerブロック) ---
    x = context_input
    for _ in range(num_transformer_blocks):
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        ffn_output = layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = layers.Dense(embed_dim)(ffn_output)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # 全てのイベント情報を集約し、文脈全体を表現する単一のベクトルを生成
    context_vector = layers.GlobalAveragePooling1D()(x)
    
    # --- Score Head (スコア計算部分) ---
    # 文脈ベクトルを選択肢の数だけ複製
    context_repeated = layers.RepeatVector(choices_len)(context_vector)
    # [文脈] と [各選択肢] の情報を結合
    merged = layers.Concatenate(axis=-1)([context_repeated, choices_input])
    
    # 結合された情報から各選択肢のスコアを計算
    score_head = layers.Dense(128, activation="relu")(merged)
    score_head = layers.Dropout(0.3)(score_head)
    output_scores = layers.Dense(1, name="output_scores")(score_head)
    output_scores = layers.Reshape((choices_len,))(output_scores)
    
    # --- マスキング処理 ---
    # マスク入力(0.0)に対応するスコアに非常に大きな負の値を加算する。
    # これにより、softmax関数を適用した際に、その選択肢の確率がほぼ0になる。
    masking_layer = (1.0 - mask_input) * -1e9
    masked_scores = layers.Add()([output_scores, masking_layer])
    
    # 3つの入力を受け取り、マスク済みのスコアを出力するモデルを定義
    model = models.Model(
        inputs=[context_input, choices_input, mask_input], 
        outputs=masked_scores
    )
    return model

def main():
    PROCESSED_DATA_PATH = 'processed_data/training_dataset_transformer.pkl'
    MODEL_PATH = 'models/senas_jan_ai_transformer_v1.keras'
    os.makedirs('models', exist_ok=True)

    # --- 1. データセットの読み込み ---
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Dataset not found at '{PROCESSED_DATA_PATH}'. Please run 'generate_data.py' first.")
        return

    print(f"Loading processed dataset from {PROCESSED_DATA_PATH}...")
    with open(PROCESSED_DATA_PATH, 'rb') as f:
        contexts, choices, labels, masks = pickle.load(f)
        
    print(f"\nData ready. Samples: {len(labels)}, Contexts shape: {contexts.shape}, Choices shape: {choices.shape}, Masks shape: {masks.shape}")

    # --- 2. モデルの構築 ---
    print("\n--- Building Masked Transformer Model... ---")
    model = build_masked_transformer()
    model.summary()

    # --- 3. モデルの学習 ---
    print("\n--- Training Model... ---")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        # from_logits=True: モデルの出力がsoftmax適用前の生スコアであることを示す
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    callbacks_list = [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(filepath=MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # データを訓練用と検証用に分割
    X_train_ctx, X_val_ctx, \
    X_train_cho, X_val_cho, \
    y_train, y_val, \
    X_train_mask, X_val_mask = train_test_split(
        contexts, choices, labels, masks, test_size=0.1, random_state=42
    )
    
    # model.fit に渡す入力をリスト形式でまとめる
    train_inputs = [X_train_ctx, X_train_cho, X_train_mask]
    val_inputs = [X_val_ctx, X_val_cho, X_val_mask]

    model.fit(
        train_inputs, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(val_inputs, y_val),
        callbacks=callbacks_list
    )
    
    print(f"\n--- Finished Training. Best model saved to {MODEL_PATH} ---")

if __name__ == '__main__':
    main()
