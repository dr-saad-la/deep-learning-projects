"""
    Model utility functions
"""
import os


import tensorflow as tf


# The model function
def create_model(input_shape, initial_lr=0.001):
    """Create a model with specified initial learning rate"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model


# Create callbacks
def create_callbacks(checkpoint_dir='cancer_model_checkpoints'):
    """Create training callbacks with enhanced monitoring."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_delta=0.0001,
        min_lr=1e-6,
        cooldown=2,
        mode='min',
        verbose=1
    )

    # Custom callback to log learning rate changes
    class LRLoggingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            print(f'\nEpoch {epoch + 1}: Current learning rate: {lr:.2e}')

    callbacks = [
        lr_reducer,
        LRLoggingCallback(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            min_delta=0.0001,
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(checkpoint_dir, 'training_log.csv')
        )
    ]

    return callbacks


# Train a model with LR On Plateau Callback
def train_with_lr_plateau(initial_lr=0.01, batch_size=32, epochs=200):
    """Train a binary classification model with ReduceLROnPlateau callback.

    Args:
        initial_lr (float): Starting learning rate (default: 0.01)
        batch_size (int): Number of samples per batch (default: 32)
        epochs (int): Maximum number of epochs to train (default: 200)

    Returns:
        tuple: (trained model, training history, test features, test labels)
    """
    # Load and preprocess data with proper validation split
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_preprocess_data()

    # Create model with specified learning rate
    model = create_model((X_train.shape[1],), initial_lr)

    # Create checkpoint directory
    checkpoint_dir = 'cancer_model_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define callbacks with enhanced monitoring
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,            # Reduce LR by half when plateauing
        patience=10,           # Wait for 10 epochs before reducing LR
        min_delta=0.0001,
        min_lr=1e-6,
        cooldown=2,           # Added cooldown period
        mode='min',
        verbose=1
    )

    # Custom callback for learning rate logging
    class LRLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            if not hasattr(logs, 'learning_rate'):
                logs['learning_rate'] = []
            logs['learning_rate'] = lr

    callbacks = [
        lr_reducer,
        LRLogger(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            min_delta=0.0001,
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(checkpoint_dir, 'training_log.csv')
        )
    ]

    # Train model with validation data
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history, X_test, y_test
