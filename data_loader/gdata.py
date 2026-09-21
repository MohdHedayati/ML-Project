import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ============================================================
# 1. PATHS
# ============================================================

DATASET_ROOT = r"D:\python\Me\Images\gtsrb-german-traffic-sign\versions\1"

TRAIN_CSV = os.path.join(DATASET_ROOT, "Train.csv")
TEST_CSV = os.path.join(DATASET_ROOT, "Test.csv")


# ============================================================
# 2. LOAD CSV
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Original training images:", len(train_df))
print("Testing images:", len(test_df))


# ============================================================
# 3. TRAIN / VALIDATION SPLIT
# ============================================================

train_df, val_df = train_test_split(
    train_df,
    test_size=0.20,
    random_state=42,
    stratify=train_df["ClassId"]
)

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))


# ============================================================
# 4. CREATE IMAGE PATHS AND LABELS
# ============================================================

train_paths = [
    os.path.join(DATASET_ROOT, path)
    for path in train_df["Path"]
]

val_paths = [
    os.path.join(DATASET_ROOT, path)
    for path in val_df["Path"]
]

test_paths = [
    os.path.join(DATASET_ROOT, path)
    for path in test_df["Path"]
]

train_labels = train_df["ClassId"].values
val_labels = val_df["ClassId"].values
test_labels = test_df["ClassId"].values


# ============================================================
# 5. IMAGE LOADING FUNCTION
# ============================================================

IMG_SIZE = (128, 128)


def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_png(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# ============================================================
# 6. CREATE TENSORFLOW DATASETS
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
# 7. LOAD AND PREPROCESS IMAGES
# ============================================================

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# 8. BATCHING
# ============================================================

BATCH_SIZE = 32

train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


# ============================================================
# 9. PREFETCH
# ============================================================

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# ============================================================
# 10. VERIFY
# ============================================================

for images, labels in train_dataset.take(1):

    print("\nTrain batch:")
    print("Images:", images.shape)
    print("Labels:", labels.shape)

    print("First image:", images[0].shape)
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