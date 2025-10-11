# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
import os
import pickle
from .vectorizer import MAX_CONTEXT_LENGTH, MAX_CHOICES, VECTOR_DIM

def build_masked_transformer(context_len=MAX_CONTEXT_LENGTH, choices_len=MAX_CHOICES, embed_dim=VECTOR_DIM, num_heads=4, ff_dim=256, num_transformer_blocks=2):
    context_input = layers.Input(shape=(context_len, embed_dim), name="context_input")
    choices_input = layers.Input(shape=(choices_len, embed_dim), name="choices_input")
    mask_input = layers.Input(shape=(choices_len,), name="mask_input")
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
    masking_layer = (1.0 - mask_input) * -1e9
    masked_scores = layers.Add()([output_scores, masking_layer])
    model = models.Model(inputs=[context_input, choices_input, mask_input], outputs=masked_scores)
    return model

def main():
    # 全てのパスをColabの高速なローカルストレージに変更
    PROCESSED_DATA_PATH = '/content/processed_data/training_dataset_transformer.pkl'
    MODEL_PATH = '/content/models/senas_jan_ai_transformer_v1.keras'
    os.makedirs('/content/models', exist_ok=True)

    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Dataset not found at '{PROCESSED_DATA_PATH}'.")
        print("Please ensure the data generation and copying steps are completed.")
        return

    print(f"Loading processed dataset from {PROCESSED_DATA_PATH}...")
    with open(PROCESSED_DATA_PATH, 'rb') as f:
        contexts, choices, labels, masks = pickle.load(f)
    print(f"Data ready. Samples: {len(labels)}")

    print("\n--- Building Masked Transformer Model... ---")
    model = build_masked_transformer()
    model.summary()

    print("\n--- Training Model... ---")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    callbacks_list = [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(filepath=MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    X_train_ctx, X_val_ctx, X_train_cho, X_val_cho, y_train, y_val, X_train_mask, X_val_mask = train_test_split(
        contexts, choices, labels, masks, test_size=0.1, random_state=42
    )
    
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

