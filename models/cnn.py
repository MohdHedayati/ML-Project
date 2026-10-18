import tensorflow as tf
from tensorflow.keras import layers, models

def create(nclasses, input_shape=(49, 49, 3), use_gap=False):
    """
    Creates a convolutional neural network for traffic sign classification.

    Parameters:
    - nclasses (int): Number of output target classes.
    - input_shape (tuple): Spatial and channel dimensions (H, W, C).
    - use_gap (bool): If True, replaces Flatten + Dense(100) with GlobalAveragePooling2D
                      to slash parameter count from ~3.3M to ~110k, reducing overfitting.
    """
    model_layers = [
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Block 3
        layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
    ]

    # Classifier Head
    if use_gap:
        model_layers.extend([
            layers.GlobalAveragePooling2D(),
            layers.Dense(units=128, activation="relu"),
        ])
    else:
        model_layers.extend([
            layers.Flatten(),
            layers.Dense(units=100, activation="relu"),
        ])

    model_layers.extend([
        layers.Dropout(rate=0.5),
        layers.Dense(units=nclasses, activation="softmax")
    ])

    return models.Sequential(model_layers)