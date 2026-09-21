import os
import pandas as pd
import numpy as np
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PATHS
# ============================================================

DATASET_ROOT = r"D:\python\Me\Images\belgium-ts\versions\10"

TRAIN_ROOT = os.path.join(
    DATASET_ROOT,
    "BelgiumTSC_Training",
    "Training"
)

TEST_ROOT = os.path.join(
    DATASET_ROOT,
    "BelgiumTSC_Testing",
    "Testing"
)


# ============================================================
# 2. READ BTSD ANNOTATIONS
# ============================================================

def read_btsd_annotations(root_folder):

    image_paths = []
    labels = []

    # Each class has its own folder
    class_folders = sorted(
        [
            folder
            for folder in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, folder))
        ]
    )

    print("Number of class folders:", len(class_folders))

    for class_folder in class_folders:

        class_path = os.path.join(
            root_folder,
            class_folder
        )

        # Find GT-xxxxx.csv
        csv_files = [
            file
            for file in os.listdir(class_path)
            if file.lower().endswith(".csv")
        ]

        if not csv_files:
            print("No CSV found in:", class_path)
            continue

        csv_path = os.path.join(
            class_path,
            csv_files[0]
        )

        # BTSD CSV uses ; as separator
        df = pd.read_csv(
            csv_path,
            sep=";"
        )

        for _, row in df.iterrows():

            filename = str(row["Filename"])
            label = int(row["ClassId"])

            image_path = os.path.join(
                class_path,
                filename
            )

            image_paths.append(image_path)
            labels.append(label)

    return image_paths, labels


# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

train_paths, train_labels = read_btsd_annotations(
    TRAIN_ROOT
)

print("\nTotal training images:", len(train_paths))
print("Total training labels:", len(train_labels))
print("\nTraining images by folder:")
for folder in sorted(os.listdir(TRAIN_ROOT)):
    folder_path = os.path.join(TRAIN_ROOT, folder)

    if os.path.isdir(folder_path):
        count = len([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".ppm")
        ])

        print(folder, ":", count)


# ============================================================
# 4. LOAD TESTING DATA
# ============================================================

test_paths, test_labels = read_btsd_annotations(
    TEST_ROOT
)

print("Total testing images:", len(test_paths))
print("Total testing labels:", len(test_labels))
print("\nTesting images by folder:")
for folder in sorted(os.listdir(TEST_ROOT)):
    folder_path = os.path.join(TEST_ROOT, folder)

    if os.path.isdir(folder_path):
        count = len([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".ppm")
        ])

        print(folder, ":", count)

from collections import Counter

print("\nTraining class distribution:")
print(sorted(Counter(train_labels).items()))

print("\nTesting class distribution:")
print(sorted(Counter(test_labels).items()))
# ============================================================
# 5. CHECK CLASS COUNT
# ============================================================

print("\nNumber of unique training classes:")
print(len(np.unique(train_labels)))

print("Number of unique testing classes:")
print(len(np.unique(test_labels)))


# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths,
    train_labels,
    test_size=0.20,
    random_state=42,
    stratify=train_labels
)

print("\nAfter split:")
print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))
print("Testing images:", len(test_paths))


# ============================================================
# 7. IMAGE SIZE
# ============================================================

IMG_SIZE = (128, 128)


# ============================================================
# 8. IMAGE LOADING FUNCTION
# ============================================================

def load_image(path, label):

    # Convert TensorFlow string tensor to Python string
    path = path.numpy().decode("utf-8")

    # Open PPM using PIL
    image = Image.open(path).convert("RGB")

    # Resize
    image = image.resize(IMG_SIZE)

    # Convert to NumPy
    image = np.array(image, dtype=np.float32)

    # Normalize
    image = image / 255.0

    return image, np.int32(label)


def tf_load_image(path, label):

    image, label = tf.py_function(
        func=load_image,
        inp=[path, label],
        Tout=[tf.float32, tf.int32]
    )

    # Tell TensorFlow the shape
    image.set_shape(
        [IMG_SIZE[0], IMG_SIZE[1], 3]
    )

    label.set_shape([])

    return image, label


# ============================================================
# 9. CREATE TENSORFLOW DATASETS
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
# 10. APPLY IMAGE LOADING
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
# 11. BATCHING
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
# 12. PREFETCH
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
# 13. VERIFY TRAINING DATA
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
# 14. VERIFY VALIDATION DATA
# ============================================================

for images, labels in val_dataset.take(1):

    print("\nValidation batch:")
    print("Images:", images.shape)
    print("Labels:", labels.shape)


# ============================================================
# 15. VERIFY TEST DATA
# ============================================================

for images, labels in test_dataset.take(1):

    print("\nTest batch:")
    print("Images:", images.shape)
    print("Labels:", labels.shape)