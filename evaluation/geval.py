import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sn

from data_loader.gdata import test_dataset, test_labels

MODEL_PATH = os.path.join(PROJECT_ROOT, "german_cnn.keras")
OUTPUT_PLOT_PATH = os.path.join(PROJECT_ROOT, "confusion_matrix_german.png")

def evaluate():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Train the model first.")

    print(f"Loading GTSRB model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Use direct test_labels (avoids decoding all test images twice)
    actualY = np.array(test_labels)

    print("Generating predictions on test dataset...")
    predY = model.predict(test_dataset)
    predY = predY.argmax(axis=1)

    # Compute classification metrics
    accuracy = accuracy_score(actualY, predY)
    precision = precision_score(actualY, predY, average="weighted", zero_division=0)
    recall = recall_score(actualY, predY, average="weighted", zero_division=0)
    f1 = f1_score(actualY, predY, average="weighted", zero_division=0)
    f1_macro = f1_score(actualY, predY, average="macro", zero_division=0)

    print("\n--- GTSRB Evaluation Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"F1 Macro:  {f1_macro:.4f}")

    # Generate Confusion Matrix
    confuse = confusion_matrix(actualY, predY)
    plt.figure(figsize=(20, 20))
    sn.heatmap(
        confuse,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("GTSRB Confusion Matrix")
    plt.savefig(OUTPUT_PLOT_PATH, bbox_inches="tight")
    print(f"Confusion matrix plot saved to: {OUTPUT_PLOT_PATH}")
    plt.close()

    print("\nClassification Report:")
    print(classification_report(actualY, predY, zero_division=0))

if __name__ == "__main__":
    evaluate()