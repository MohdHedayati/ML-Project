import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from models.cnn import create
from data_loader.gdata import train_dataset, val_dataset, test_dataset

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "german_cnn.keras")

def train(epochs=10):
    nclasses = 43
    model = create(nclasses)

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("Starting GTSRB model training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    loss, accuracy = model.evaluate(test_dataset)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved successfully to: {MODEL_SAVE_PATH}")
    return model, history

if __name__ == "__main__":
    train()