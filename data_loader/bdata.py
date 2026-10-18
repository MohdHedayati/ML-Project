import os
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

# ============================================================
# 1. PATHS (Dynamic resolution relative to project root)
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.environ.get(
    "BELGIUM_DATASET_ROOT",
    os.path.join(PROJECT_ROOT, "Images", "belgium-ts", "versions", "10")
)

TRAIN_ROOT = os.path.join(DATASET_ROOT, "BelgiumTSC_Training", "Training")
TEST_ROOT = os.path.join(DATASET_ROOT, "BelgiumTSC_Testing", "Testing")

IMG_SIZE = (49, 49)
BATCH_SIZE = 32


# ============================================================
# 2. READ BTSD ANNOTATIONS
# ============================================================

def read_btsd_annotations(root_folder):
    image_paths = []
    labels = []

    if not os.path.exists(root_folder):
        return image_paths, labels

    class_folders = sorted(
        [
            folder
            for folder in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, folder))
        ]
    )

    for class_folder in class_folders:
        class_path = os.path.join(root_folder, class_folder)
        csv_files = [f for f in os.listdir(class_path) if f.lower().endswith(".csv")]
        if not csv_files:
            continue

        csv_path = os.path.join(class_path, csv_files[0])
        df = pd.read_csv(csv_path, sep=";")

        for _, row in df.iterrows():
            filename = str(row["Filename"])
            label = int(row["ClassId"])
            image_path = os.path.join(class_path, filename)
            image_paths.append(image_path)
            labels.append(label)

    return image_paths, labels


# ============================================================
# 3. PPM IMAGE LOADING (via PIL wrapped in tf.py_function)
# ============================================================

def _pil_load_ppm(path, label, img_size=IMG_SIZE):
    path_str = path.numpy().decode("utf-8")
    with Image.open(path_str) as img:
        img_rgb = img.convert("RGB").resize(img_size)
        img_arr = np.array(img_rgb, dtype=np.float32) / 255.0
    return img_arr, np.int32(label)


def tf_load_image(path, label, img_size=IMG_SIZE):
    image, label = tf.py_function(
        func=lambda p, l: _pil_load_ppm(p, l, img_size),
        inp=[path, label],
        Tout=[tf.float32, tf.int32]
    )
    image.set_shape([img_size[0], img_size[1], 3])
    label.set_shape([])
    return image, label


# ============================================================
# 4. DATASET BUILDER
# ============================================================

def build_datasets(batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle_buffer=1000):
    raw_train_paths, raw_train_labels = read_btsd_annotations(TRAIN_ROOT)
    test_paths, test_labels = read_btsd_annotations(TEST_ROOT)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        raw_train_paths,
        raw_train_labels,
        test_size=0.20,
        random_state=42,
        stratify=raw_train_labels
    )

    def _mapper(p, l):
        return tf_load_image(p, l, img_size)

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

    train_ds = (
        train_ds.map(_mapper, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=shuffle_buffer, seed=42, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds.map(_mapper, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        test_ds.map(_mapper, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, train_labels, val_labels, test_labels


# ============================================================
# 5. MODULE-LEVEL EXPORTS (Backward compatibility)
# ============================================================

train_dataset, val_dataset, test_dataset, train_labels, val_labels, test_labels = build_datasets()


# ============================================================
# 6. VERIFICATION (Runs only when executed directly)
# ============================================================

if __name__ == "__main__":
    from collections import Counter
    print("Training samples count:", len(train_labels))
    print("Validation samples count:", len(val_labels))
    print("Testing samples count:", len(test_labels))
    print("Unique classes in train:", len(np.unique(train_labels)))
    print("Unique classes in test:", len(np.unique(test_labels)))

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