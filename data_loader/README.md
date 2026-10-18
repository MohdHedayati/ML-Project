# Data Loader Documentation

This folder contains TensorFlow data loaders for three traffic sign datasets.
This document explains the pipeline used in `bdata.py` (Belgium) and `gdata.py` (German),
which share the same overall structure.

---

## Files Overview

| File | Dataset | Classes | Image Format |
|---|---|---|---|
| `gdata.py` | GTSRB — German Traffic Sign Recognition Benchmark | 43 | PNG |
| `bdata.py` | BTSD — Belgian Traffic Sign Dataset | 62 | PPM |
| `cdata.py` | Chinese Traffic Signs Dataset | 58 | PNG |

---

## Pipeline (Step by Step)

### Step 1 — Define Paths

**`gdata.py`** reads from two CSV files that come with the dataset:

```
gtsrb-german-traffic-sign/versions/1/
├── Train.csv       ← paths + labels for training images
├── Test.csv        ← paths + labels for test images
├── Train/          ← actual training images (PNG)
└── Test/           ← actual test images (PNG)
```

**`bdata.py`** reads from a folder structure where each class has its own subfolder:

```
belgium-ts/versions/10/
├── BelgiumTSC_Training/Training/
│   ├── 00000/
│   │   ├── GT-00000.csv    ← annotations for this class
│   │   ├── 00001.ppm
│   │   └── ...
│   ├── 00001/
│   └── ...
└── BelgiumTSC_Testing/Testing/
    └── (same structure)
```

---

### Step 2 — Read Annotations

**`gdata.py`** — simple CSV read. The CSV already has `Path` and `ClassId` columns:

```python
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)
```

**`bdata.py`** — walks each class folder, finds the `GT-xxxxx.csv` inside it (semicolon-separated),
and collects image paths + labels:

```python
for class_folder in class_folders:
    csv_path = ...                        # finds GT-xxxxx.csv inside the folder
    df = pd.read_csv(csv_path, sep=";")   # BTSD uses semicolons, not commas
    for row in df.iterrows():
        image_paths.append(...)
        labels.append(int(row["ClassId"]))
```

---

### Step 3 — Train / Validation Split

Both datasets come with an **official test set** (separate folder or CSV), so `train_test_split`
is only called **once** — to carve out a validation set from the training data.

```python
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths,
    train_labels,
    test_size=0.20,       # 20% of training data becomes validation
    random_state=42,      # fixed seed for reproducibility
    stratify=train_labels # preserves class distribution in both halves
)
```

After this split, the data is divided as:

```
All training data  ->  80% train  +  20% validation
Official test data ->  test (untouched)
```

---

### Step 4 — Image Loading Function

This is the main difference between the two loaders.

**`gdata.py`** — uses native TensorFlow ops (faster, no Python overhead):

```python
IMG_SIZE = (49, 49)

def load_image(path, label):
    image = tf.io.read_file(path)                   # read raw bytes
    image = tf.image.decode_png(image, channels=3)  # decode PNG -> tensor
    image = tf.image.resize(image, IMG_SIZE)         # resize to 49x49
    image = tf.cast(image, tf.float32) / 255.0       # normalize to [0.0, 1.0]
    return image, label
```

**`bdata.py`** — uses PIL because TensorFlow cannot natively decode `.ppm` files.
PIL runs in Python, so it must be wrapped with `tf.py_function` to work inside `tf.data`:

```python
def load_image(path, label):
    path = path.numpy().decode("utf-8")     # convert TF tensor -> Python string
    image = Image.open(path).convert("RGB") # open PPM with PIL
    image = image.resize(IMG_SIZE)          # resize to 49x49
    image = np.array(image, dtype=np.float32)
    image = image / 255.0                   # normalize to [0.0, 1.0]
    return image, np.int32(label)

def tf_load_image(path, label):
    image, label = tf.py_function(
        func=load_image,
        inp=[path, label],
        Tout=[tf.float32, tf.int32]
    )
    # tf.py_function loses shape info, so we restore it manually
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    label.set_shape([])
    return image, label
```

> **Why tf.py_function?**
> TensorFlow's graph execution cannot run arbitrary Python code directly.
> tf.py_function wraps a Python function so it can be called inside a tf.data pipeline,
> but it runs eagerly and bypasses graph optimizations — making it slightly slower than
> native TF ops. It is necessary here because PIL (used to open .ppm files) is pure Python.

---

### Step 5 — Build tf.data.Dataset

Convert the lists of paths and labels into TensorFlow datasets:

```python
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_dataset   = tf.data.Dataset.from_tensor_slices((val_paths,   val_labels))
test_dataset  = tf.data.Dataset.from_tensor_slices((test_paths,  test_labels))
```

This creates a dataset where each element is a (path, label) pair — images are not
loaded into memory yet. They are loaded lazily when the pipeline runs.

---

### Step 6 — Apply Image Loading (.map)

`.map()` applies the loading function to every element in the dataset.
`AUTOTUNE` lets TensorFlow decide how many parallel threads to use:

```python
train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset   = val_dataset.map(load_image,   num_parallel_calls=tf.data.AUTOTUNE)
test_dataset  = test_dataset.map(load_image,  num_parallel_calls=tf.data.AUTOTUNE)
```

After this step, each element is a (image_tensor, label) pair,
where image_tensor has shape (128, 128, 3) and values in [0.0, 1.0].

---

### Step 7 — Batch

Groups individual samples into batches of 32:

```python
BATCH_SIZE = 32

train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset   = val_dataset.batch(BATCH_SIZE)
test_dataset  = test_dataset.batch(BATCH_SIZE)
```

After batching, each element has shape (32, 128, 128, 3) for images
and (32,) for labels.

---

### Step 8 — Prefetch

Prepares the next batch in the background while the model is training on the current one.
This hides I/O latency and keeps the GPU busy:

```python
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset   = val_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset  = test_dataset.prefetch(tf.data.AUTOTUNE)
```

---

### Step 9 — Verification

Each loader ends with a quick sanity check to confirm shapes and pixel range:

```python
for images, labels in train_dataset.take(1):
    print("Images:", images.shape)   # expected: (32, 128, 128, 3)
    print("Labels:", labels.shape)   # expected: (32,)
    print("Pixel range:", tf.reduce_min(images).numpy(), "to", tf.reduce_max(images).numpy())
    # expected: 0.0 to 1.0
```

---

## Full Pipeline Summary

```
Raw files on disk
      |
      v
read_btsd_annotations() / pd.read_csv()
  -> list of (path, label) pairs
      |
      v
train_test_split(stratify=True)
  -> train / val / test path lists
      |
      v
tf.data.Dataset.from_tensor_slices()
  -> lazy (path, label) dataset
      |
      v
.map(load_image, AUTOTUNE)
  -> (image_tensor, label) -- shape (49, 49, 3), normalized [0, 1]
      |
      v
.batch(32)
  -> batches of 32 images
      |
      v
.prefetch(AUTOTUNE)
  -> ready for model.fit() / model.evaluate()
```

---

## Key Constants

| Constant | Value | Purpose |
|---|---|---|
| `IMG_SIZE` | `(49, 49)` | All images resized to this |
| `BATCH_SIZE` | `32` | Number of images per batch |
| `test_size` | `0.20` | 20% of training data used for validation |
| `random_state` | `42` | Fixed seed for reproducible splits |
