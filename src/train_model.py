# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- 定数定義 ---
PROCESSED_DATA_DIR = 'data/processed'
BATCH_SIZE = 128
EPOCHS = 50

# --- 1. データジェネレータ ---
def data_generator(data_dir, batch_size):
    """
    .npyファイルをメモリマップモードで読み込み、バッチごとにデータを生成するジェネレータ
    """
    try:
        context_input = np.load(os.path.join(data_dir, 'context.npy'), mmap_mode='r')
        choices_input = np.load(os.path.join(data_dir, 'choices.npy'), mmap_mode='r')
        mask_input = np.load(os.path.join(data_dir, 'mask.npy'), mmap_mode='r')
        labels = np.load(os.path.join(data_dir, 'labels.npy'), mmap_mode='r')
    except FileNotFoundError:
        print(f"エラー: データファイルが見つかりません。'{data_dir}'に必要な.npyファイルが存在するか確認してください。")
        # 空のジェネレータを返す
        yield
        return


    num_samples = context_input.shape[0]
    indices = np.arange(num_samples)
    
    while True:
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            X_batch = {
                'context_input': context_input[batch_indices],
                'choices_input': choices_input[batch_indices],
                'mask_input': mask_input[batch_indices]
            }
            y_batch = labels[batch_indices]
            
            yield X_batch, y_batch

# --- 2. モデル構築 ---
def build_transformer_model(context_shape, choice_shape):
    """
    データ形式に合わせたTransformerベースのモデルを構築する
    """
    # Encoder part (for context)
    context_input = layers.Input(shape=context_shape, name='context_input')
    
    # 2層のTransformer Encoder Block
    x = context_input
    for _ in range(2):
        attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=100)(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        ffn_output = layers.Dense(128, activation='relu')(x)
        ffn_output = layers.Dense(context_shape[1])(ffn_output)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
    context_vector = layers.GlobalAveragePooling1D()(x)

    # Combining context and choices
    choices_input = layers.Input(shape=choice_shape, name='choices_input')
    
    # コンテキストベクトルを各選択肢にブロードキャスト
    context_repeated = layers.RepeatVector(choice_shape[0])(context_vector)
    
    # コンテキストと選択肢を結合
    merged = layers.Concatenate()([context_repeated, choices_input])
    
    # Decision network
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    output_scores = layers.Dense(1, name='output_scores')(x)
    output_scores = layers.Reshape((choice_shape[0],))(output_scores) # (None, 50, 1) -> (None, 50)

    # Masking
    mask_input = layers.Input(shape=(choice_shape[0],), name='mask_input')
    # マスクを適用: マスクが0の部分を-infにすることでsoftmaxで確率0にする
    masked_output = layers.Add()([output_scores, (1.0 - mask_input) * -1e9])
    
    final_output = layers.Activation('softmax')(masked_output)

    model = models.Model(
        inputs=[context_input, choices_input, mask_input], 
        outputs=final_output
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- 3. メイン実行部分 ---
def main():
    """メインの実行関数"""
    print("--- Step 1: Preparing data generators... ---")
    train_dir = os.path.join(PROCESSED_DATA_DIR, 'train')
    val_dir = os.path.join(PROCESSED_DATA_DIR, 'validation')

    train_generator = data_generator(train_dir, BATCH_SIZE)
    validation_generator = data_generator(val_dir, BATCH_SIZE)
    
    # サンプル数を計算
    num_train_samples = len(np.load(os.path.join(train_dir, 'labels.npy'), mmap_mode='r'))
    num_val_samples = len(np.load(os.path.join(val_dir, 'labels.npy'), mmap_mode='r'))

    print(f"Training samples: {num_train_samples}")
    print(f"Validation samples: {num_val_samples}")

    if num_train_samples == 0:
        print("学習データが見つからないため、処理を終了します。")
        return

    print("\n--- Step 2: Building the model... ---")
    # ジェネレータから最初のバッチを取得して形状を確認
    sample_x, _ = next(train_generator)
    context_shape = sample_x['context_input'].shape[1:]
    choice_shape = sample_x['choices_input'].shape[1:]
    
    model = build_transformer_model(context_shape, choice_shape)
    model.summary()

    print("\n--- Step 3: Training the model... ---")
    
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        train_generator,
        steps_per_epoch=num_train_samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=num_val_samples // BATCH_SIZE,
        callbacks=callbacks
    )

    print("\n--- Step 4: Saving the model... ---")
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'mahjong_transformer_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
