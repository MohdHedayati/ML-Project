import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ============================================================
# 1. PATHS (Dynamic resolution relative to project root)
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.environ.get(
    "CTSD_DATASET_ROOT",
    os.path.join(PROJECT_ROOT, "Images", "chinese-traffic-signs", "versions", "2")
)

IMAGE_ROOT = os.path.join(DATASET_ROOT, "images")
ANNOTATION_FILE = os.path.join(DATASET_ROOT, "annotations.csv")

IMG_SIZE = (49, 49)
BATCH_SIZE = 32


# ============================================================
# 2. IMAGE LOADING FUNCTION (Native TensorFlow graph ops for PNG)
# ============================================================

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# ============================================================
# 3. DATASET BUILDER
# ============================================================

def build_datasets(batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle_buffer=1000):
    df = pd.read_csv(ANNOTATION_FILE)

    image_paths = [os.path.join(IMAGE_ROOT, filename) for filename in df["file_name"]]
    labels = df["category"].astype(int).values

    # Step 1: Carve out 20% test set
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    # Step 2: Carve out 20% validation set from training portion
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=0.20,
        random_state=42,
        stratify=train_labels
    )

    def _loader(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

    train_ds = (
        train_ds.map(_loader, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=shuffle_buffer, seed=42, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds.map(_loader, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        test_ds.map(_loader, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, train_labels, val_labels, test_labels


# ============================================================
# 4. MODULE-LEVEL EXPORTS (Backward compatibility)
# ============================================================

train_dataset, val_dataset, test_dataset, train_labels, val_labels, test_labels = build_datasets()


# ============================================================
# 5. VERIFICATION (Runs only when executed directly)
# ============================================================

if __name__ == "__main__":
    df = pd.read_csv(ANNOTATION_FILE)
    print("Annotation data:")
    print(df.head())
    print("\nTotal images:", len(df))
    print("Unique classes:", len(np.unique(df["category"].values)))
    print("Training images:", len(train_labels))
    print("Validation images:", len(val_labels))
    print("Testing images:", len(test_labels))

    for images, labels in train_dataset.take(1):
        print("\nTrain batch:")
        print("Images:", images.shape)
        print("Labels:", labels.shape)
        print("First image shape:", images[0].shape)
        print("First label:", labels[0].numpy())
        print(
            "Pixel range:",
            tf.reduce_min(images).numpy(),
            "to",
            tf.reduce_max(images).numpy()
        )

    for images, labels in val_dataset.take(1):
        print("\nValidation batch:")
        print("Images:", images.shape)
        print("Labels:", labels.shape)

    for images, labels in test_dataset.take(1):
        print("\nTest batch:")
        print("Images:", images.shape)
        print("Labels:", labels.shape)