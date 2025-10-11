# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from src.mjlog_parser import MjlogParser

def prepare_dataset(data_dir='data', processed_data_dir='processed_data'):
    """データセットを準備する。なければログから生成する"""
    dataset_path = os.path.join(processed_data_dir, 'training_dataset_ph2_shanten.npz')
    
    if os.path.exists(dataset_path):
        print(f"Loading processed dataset from {dataset_path}...")
        data = np.load(dataset_path)
        return data['features'], data['labels']

    print(f"Processed dataset not found. Generating from logs in {data_dir}...")
    os.makedirs(processed_data_dir, exist_ok=True)
    
    log_files = glob.glob(os.path.join(data_dir, '**', '*.mjlog'), recursive=True)
    if not log_files:
        print(f"No .mjlog files found in {data_dir}")
        return None, None
        
    print(f"Found {len(log_files)} log files.")
    
    parser = MjlogParser()
    all_data = []
    
    for i, file_path in enumerate(log_files):
        print(f"Processing file {i+1}/{len(log_files)}: {os.path.basename(file_path)}")
        try:
            data = parser.parse_log_file(file_path)
            if data:
                all_data.extend(data)
        except Exception as e:
            print(f"  -> Skipping file due to error: {e}")
            continue

    print("File processing finished.")

    if not all_data:
        print("No data could be extracted.")
        return None, None

    features = np.array([item[0] for item in all_data])
    labels = np.array([item[1] for item in all_data])
    
    print(f"Saving processed dataset to {dataset_path}...")
    np.savez_compressed(dataset_path, features=features, labels=labels)

    return features, labels

def build_cnn_model_v3(input_shape):
    """
    より深く、賢いCNNモデル(v3)を構築する
    """
    inputs = layers.Input(shape=input_shape)

    # 畳み込み層を深くする
    # L2正則化を追加して過学習を抑制
    x = layers.Conv1D(128, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x) # Batch Normalizationを追加
    x = layers.Conv1D(128, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x) # プーリング層を追加
    
    x = layers.Conv1D(256, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    
    # 全結合層
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(34, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    """メインの実行関数"""
    print("--- Step 1: Preparing dataset (Phase 2 - Shanten)... ---")
    features, labels = prepare_dataset()

    if features is None or labels is None:
        print("Dataset preparation failed. Exiting.")
        return

    print("\nData ready.")
    print(f"Number of samples: {len(features)}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    print("\n--- Step 2: Building the CNN model (v3)... ---")
    input_shape = (features.shape[1], features.shape[2])
    model = build_cnn_model_v3(input_shape)
    model.summary()

    print("\n--- Step 3: Training the model... ---")
    
    # 学習率を徐々に下げるコールバック
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    # 早期終了コールバック
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=30,  # エポック数を増やす
        batch_size=256, # バッチサイズを調整
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, early_stopping]
    )

    print("\n--- Step 4: Saving the model... ---")
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'senas_jan_ai_v3.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()

