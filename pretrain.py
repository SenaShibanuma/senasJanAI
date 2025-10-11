
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

# srcディレクトリをパスに追加してインポートを許可
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 本来はmodel.pyにあるべきだが、現状はtrain_transformer.pyからインポート
from transformer.train_transformer import build_masked_transformer

def create_dataset(npz_path, batch_size):
    """
    .npzファイルからtf.data.Datasetを作成するヘルパー関数。
    メモリに全データをロードせず、ジェネレータを使ってバッチごとにデータを供給する。
    """
    # np.loadはmmap_mode='r'を使うと、全データをメモリに読み込まずに済む
    data = np.load(npz_path, mmap_mode='r')
    num_samples = data['context_input'].shape[0]

    def generator():
        for i in range(num_samples):
            X = (data['context_input'][i], data['choices_input'][i], data['mask_input'][i])
            y = data['labels'][i]
            yield X, y

    # ジェネレータからDatasetを作成
    # 出力の型と形状を明示的に指定する必要がある
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
    BATCH_SIZE = 16
    train_npz_path = os.path.join(processed_dir, 'train.npz')
    val_npz_path = os.path.join(processed_dir, 'validation.npz')

    try:
        train_dataset, num_train_samples = create_dataset(train_npz_path, BATCH_SIZE)
        val_dataset, num_val_samples = create_dataset(val_npz_path, BATCH_SIZE)
    except FileNotFoundError:
        print("エラー: 前処理済みデータが見つかりません。先に 'preprocess' を実行してください。")
        return

    print(f"訓練サンプル数: {num_train_samples}")
    print(f"検証サンプル数: {num_val_samples}")

    print("--- モデルを構築しています ---")
    # データセットの形状からモデルのパラメータを設定
    # .element_specでデータセットの形状を取得できる
    context_shape = train_dataset.element_spec[0][0].shape
    choices_shape = train_dataset.element_spec[0][1].shape
    model = build_masked_transformer(
        context_len=context_shape[1], # 150
        choices_len=choices_shape[1], # 50
        embed_dim=context_shape[2], # 100
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
        train_dataset, # NumPy配列の代わりにDatasetオブジェクトを渡す
        epochs=50,
        validation_data=val_dataset, # こちらもDatasetオブジェクト
        steps_per_epoch=num_train_samples // BATCH_SIZE,
        validation_steps=num_val_samples // BATCH_SIZE,
        callbacks=[model_checkpoint_callback, tensorboard_callback, early_stopping_callback]
    )

    print("--- 学習が完了しました ---")

    final_model_path = os.path.join(model_dir, 'tenho_model.keras')
    print(f"最終モデルを {final_model_path} に保存しています")
    model.save(final_model_path)

