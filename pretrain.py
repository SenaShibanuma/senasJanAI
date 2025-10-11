
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

# Add src to path to allow imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# It's in train_transformer, not a model file, but that's fine for now.
from transformer.train_transformer import build_masked_transformer

def pretrain(processed_dir, model_dir):
    """
    Loads preprocessed data and trains the Transformer model.
    """
    print("--- Loading Data ---")
    try:
        train_data = np.load(os.path.join(processed_dir, 'train.npz'))
        val_data = np.load(os.path.join(processed_dir, 'validation.npz'))
    except FileNotFoundError:
        print("Error: Preprocessed data not found. Please run the 'preprocess' action first.")
        return

    X_train = [train_data['context_input'], train_data['choices_input'], train_data['mask_input']]
    y_train = train_data['labels']
    X_val = [val_data['context_input'], val_data['choices_input'], val_data['mask_input']]
    y_val = val_data['labels']

    print(f"Training data shape: {X_train[0].shape}")
    print(f"Validation data shape: {X_val[0].shape}")

    print("--- Building Model ---")
    # Parameters from the original train_transformer.py, can be adjusted
    model = build_masked_transformer(
        context_len=X_train[0].shape[1],
        choices_len=X_train[1].shape[1],
        embed_dim=X_train[0].shape[2],
        num_heads=4,
        ff_dim=128,
        num_transformer_blocks=2
    )

    print("--- Compiling Model ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    print("--- Setting up Callbacks ---")
    # Create model directory if it doesn't exist
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
        patience=5, # Stop if val_loss doesn't improve for 5 epochs
        restore_best_weights=True
    )

    print("--- Starting Training ---")
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[model_checkpoint_callback, tensorboard_callback, early_stopping_callback]
    )

    print("--- Training Finished ---")

    # The EarlyStopping callback with restore_best_weights ensures the model has the best weights
    final_model_path = os.path.join(model_dir, 'tenho_model.keras')
    print(f"Saving final model to {final_model_path}")
    model.save(final_model_path)

