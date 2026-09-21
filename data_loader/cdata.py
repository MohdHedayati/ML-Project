import os
import pandas as pd
import numpy as np
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PATHS
# ============================================================

DATASET_ROOT = r"D:\python\Me\Images\chinese-traffic-signs\versions\2"

IMAGE_ROOT = os.path.join(
    DATASET_ROOT,
    "images"
)

ANNOTATION_FILE = os.path.join(
    DATASET_ROOT,
    "annotations.csv"
)


# ============================================================
# 2. LOAD ANNOTATIONS
# ============================================================

df = pd.read_csv(ANNOTATION_FILE)

print("Annotation data:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nTotal images:", len(df))


# ============================================================
# 3. CREATE IMAGE PATHS
# ============================================================

image_paths = [
    os.path.join(IMAGE_ROOT, filename)
    for filename in df["file_name"]
]

labels = df["category"].astype(int).values


# ============================================================
# 4. CHECK IMAGE PATH
# ============================================================

print("\nExample image path:")
print(image_paths[0])

print("\nDoes file exist?")
print(os.path.exists(image_paths[0]))


# ============================================================
# 5. CHECK CLASSES
# ============================================================

unique_classes = np.unique(labels)

print("\nNumber of classes:")
print(len(unique_classes))

print("\nClass IDs:")
print(unique_classes)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("\nAfter train/test split:")
print("Training images:", len(train_paths))
print("Testing images:", len(test_paths))


# ============================================================
# 7. TRAIN / VALIDATION SPLIT
# ============================================================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths,
    train_labels,
    test_size=0.20,
    random_state=42,
    stratify=train_labels
)

print("\nAfter validation split:")
print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))
print("Testing images:", len(test_paths))


# ============================================================
# 8. IMAGE SIZE
# ============================================================

IMG_SIZE = (128, 128)


# ============================================================
# 9. IMAGE LOADING FUNCTION
# ============================================================

def load_image(path, label):

    # Convert TensorFlow tensor to Python string
    path = path.numpy().decode("utf-8")

    # Open image
    image = Image.open(path).convert("RGB")

    # Resize
    image = image.resize(IMG_SIZE)

    # Convert to NumPy array
    image = np.array(
        image,
        dtype=np.float32
    )

    # Normalize pixels
    image = image / 255.0

    return image, np.int32(label)


def tf_load_image(path, label):

    image, label = tf.py_function(
        func=load_image,
        inp=[path, label],
        Tout=[tf.float32, tf.int32]
    )

    # Tell TensorFlow the output shape
    image.set_shape(
        [IMG_SIZE[0], IMG_SIZE[1], 3]
    )

    label.set_shape([])

    return image, label


# ============================================================
# 10. CREATE TENSORFLOW DATASETS
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)


# ============================================================
# 11. LOAD AND PREPROCESS IMAGES
# ============================================================

train_dataset = train_dataset.map(
    tf_load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.map(
    tf_load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.map(
    tf_load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# 12. BATCHING
# ============================================================

BATCH_SIZE = 32

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)


# ============================================================
# 13. PREFETCH
# ============================================================

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 14. VERIFY TRAINING DATA
# ============================================================

for images, labels in train_dataset.take(1):

    print("\nTrain batch:")
    print("Images:", images.shape)
    print("Labels:", labels.shape)

    print(
        "First image shape:",
        images[0].shape
    )

    print(
        "First label:",
        labels[0].numpy()
    )

    print(
        "Pixel range:",
        tf.reduce_min(images).numpy(),
        "to",
        tf.reduce_max(images).numpy()
    )


# ============================================================
# 15. VERIFY VALIDATION DATA
# ============================================================

for images, labels in val_dataset.take(1):

    print("\nValidation batch:")
    print("Images:", images.shape)
    print("Labels:", labels.shape)


# ============================================================
# 16. VERIFY TEST DATA
# ============================================================

for images, labels in test_dataset.take(1):

    print("\nTest batch:")
    print("Images:", images.shape)
    print("Labels:", labels.shape)