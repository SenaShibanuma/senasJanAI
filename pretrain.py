import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

# srcディレクトリをパスに追加してインポートを許可
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Transformerモデルの定義をインポート
from transformer.train_transformer import build_masked_transformer

def create_dataset(data_dir, batch_size):
    """
    ディレクトリ内の.npyファイルをメモリマップモードで読み込み、
    tf.data.Datasetを作成するヘルパー関数。
    """
    try:
        # 各.npyファイルを個別にメモリマップモードで開く
        # allow_pickle=True は不要なので削除
        context_data = np.load(os.path.join(data_dir, 'context.npy'), mmap_mode='r')
        choices_data = np.load(os.path.join(data_dir, 'choices.npy'), mmap_mode='r')
        mask_data = np.load(os.path.join(data_dir, 'mask.npy'), mmap_mode='r')
        labels_data = np.load(os.path.join(data_dir, 'labels.npy'), mmap_mode='r')
    except FileNotFoundError:
        print(f"エラー: {data_dir} 内に必要な.npyファイルが見つかりません。")
        return None, 0

    num_samples = context_data.shape[0]

    def generator():
        # インデックスの配列をシャッフルして、データをランダムに供給
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for i in indices:
            # 必要なデータだけをディスクから読み込む
            X = (context_data[i], choices_data[i], mask_data[i])
            y = labels_data[i]
            yield X, y

    # ジェネレータからDatasetを作成
    output_signature = (
        (tf.TensorSpec(shape=(150, 100), dtype=tf.float32),
         tf.TensorSpec(shape=(50, 100), dtype=tf.float32),
         tf.TensorSpec(shape=(50,), dtype=tf.float32)),
        tf.TensorSpec(shape=(50,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    # パフォーマンスのための設定
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, num_samples

def pretrain(processed_dir, model_dir):
    """
    前処理済みのデータをロードし、Transformerモデルの学習を行う。
    """
    print("--- データセットを準備しています ---")
    BATCH_SIZE = 16  # GPUメモリに応じて調整してください

    train_data_dir = os.path.join(processed_dir, 'train')
    val_data_dir = os.path.join(processed_dir, 'validation')

    train_dataset, num_train_samples = create_dataset(train_data_dir, BATCH_SIZE)
    val_dataset, num_val_samples = create_dataset(val_data_dir, BATCH_SIZE)

    if not train_dataset or not val_dataset:
        print("データセットの読み込みに失敗しました。'preprocess'を再実行してください。")
        return

    print(f"訓練サンプル数: {num_train_samples}")
    print(f"検証サンプル数: {num_val_samples}")

    print("--- モデルを構築しています ---")
    context_shape = train_dataset.element_spec[0][0].shape
    choices_shape = train_dataset.element_spec[0][1].shape
    model = build_masked_transformer(
        context_len=context_shape[1],
        choices_len=choices_shape[1],
        embed_dim=context_shape[2],
        num_heads=4,
        ff_dim=128,
        num_transformer_blocks=2
    )

    print("--- モデルをコンパイルしています ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()

    print("--- コールバックを設定しています ---")
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, 'tenho_model.weights.h5')
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir=os.path.join(model_dir, 'logs'))
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    print("--- 学習を開始します ---")
    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=val_dataset,
        steps_per_epoch=num_train_samples // BATCH_SIZE,
        validation_steps=num_val_samples // BATCH_SIZE,
        callbacks=[model_checkpoint_callback, tensorboard_callback, early_stopping_callback]
    )

    print("--- 学習が完了しました ---")
    final_model_path = os.path.join(model_dir, 'tenho_model.keras')
    print(f"最終モデルを {final_model_path} に保存しています")
    model.save(final_model_path)

