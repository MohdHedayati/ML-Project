import os
from pathlib import Path
from typing import List, Tuple, Generator, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from imagecorruptions import corrupt, get_corruption_names


IMAGE_SIZE = (49, 49)


CORRUPTIONS = [
    "gaussian_noise",
    "impulse_noise",
    "shot_noise",
    "snow",
    "fog",
    "frost",
    "motion_blur",
    "zoom_blur",
    "defocus_blur",
    "brightness",
    "contrast",
    "pixelate",
    "elastic_transform",
    "jpeg_compression",
]


SUPPORTED_CORRUPTIONS = set(get_corruption_names())
MISSING_CORRUPTIONS = [c for c in CORRUPTIONS if c not in SUPPORTED_CORRUPTIONS]
if MISSING_CORRUPTIONS:
    print(f"Warning: The following corruptions are not natively in imagecorruptions: {MISSING_CORRUPTIONS}")


def preprocess_corrupted_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    corruption: Optional[str] = None,
    severity: int = 1,
    handle_small_images: str = "upscale",
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """
    Standardized Image Processing Pipeline:
    Original Image -> (Optional Minimum Size Adaptation) -> Corruption -> Resize to target_size -> Normalize to [0, 1].

    Parameters:
    - image_input: File path (str/Path), PIL Image, or uint8 RGB numpy array.
    - corruption: Name of corruption from CORRUPTIONS (or None for clean processing).
    - severity: Integer severity from 1 to 5 (or 0 for clean processing).
    - handle_small_images: Strategy if original image dimensions < 32 pixels.
      'upscale' proportionally scales up to at least 32x32 to meet imagecorruptions requirements.
      'error' raises ValueError.
    - target_size: Output spatial resolution (H, W). Defaults to IMAGE_SIZE (49, 49).

    Returns:
    - float32 numpy array with shape (H, W, 3) and values in [0.0, 1.0].
    """
    # 1. Load image
    if isinstance(image_input, (str, Path)):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        with Image.open(image_input) as img:
            original = img.convert("RGB")
    elif isinstance(image_input, Image.Image):
        original = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        original = Image.fromarray(image_input).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Clean baseline passthrough if corruption is None or severity is 0
    if corruption is None or severity == 0:
        resized = original.resize(target_size, Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0

    if corruption not in CORRUPTIONS:
        raise ValueError(f"Unsupported corruption: {corruption}. Must be one of {CORRUPTIONS}")

    if severity not in range(1, 6):
        raise ValueError(f"Severity must be an integer between 1 and 5. Got: {severity}")

    # 2. Minimum dimension handling (imagecorruptions requires width, height >= 32)
    width, height = original.size
    if width < 32 or height < 32:
        if handle_small_images == "upscale":
            # Scale proportionally so min dimension is at least 32
            scale = max(32 / width, 32 / height)
            new_w = max(32, int(round(width * scale)))
            new_h = max(32, int(round(height * scale)))
            original = original.resize((new_w, new_h), Image.Resampling.BILINEAR)
        elif handle_small_images == "error":
            raise ValueError(
                f"Image is too small for imagecorruptions ({width}x{height} < 32x32): {image_input}"
            )
        else:
            raise ValueError(f"Unknown handle_small_images strategy: {handle_small_images}")

    # 3. Convert to uint8 numpy array for imagecorruptions
    original_array = np.asarray(original, dtype=np.uint8)

    # 4. Apply corruption at original resolution
    corrupted_array = corrupt(
        original_array,
        corruption_name=corruption,
        severity=severity,
    )

    # Ensure output is uint8 RGB array
    if corrupted_array.dtype != np.uint8:
        corrupted_array = np.clip(corrupted_array, 0, 255).astype(np.uint8)

    # 5. Resize to CNN input resolution
    corrupted_image = Image.fromarray(corrupted_array)
    resized = corrupted_image.resize(target_size, Image.Resampling.BILINEAR)

    # 6. Normalize to [0.0, 1.0] float32
    return np.asarray(resized, dtype=np.float32) / 255.0


def corrupted_batch_generator(
    image_paths: List[str],
    labels: List[int],
    corruption: Optional[str],
    severity: int,
    batch_size: int = 32,
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Streams corrupted batches into memory without loading the entire dataset into RAM.
    Yields (batch_images, batch_labels) where:
    - batch_images is a float32 ndarray of shape (B, H, W, 3) in [0.0, 1.0]
    - batch_labels is an int32 ndarray of shape (B,)
    """
    total = len(image_paths)
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_paths = image_paths[start_idx:end_idx]
        batch_lbls = labels[start_idx:end_idx]

        batch_imgs = []
        valid_lbls = []
        for path, lbl in zip(batch_paths, batch_lbls):
            try:
                img_arr = preprocess_corrupted_image(
                    path, corruption=corruption, severity=severity, target_size=target_size
                )
                batch_imgs.append(img_arr)
                valid_lbls.append(lbl)
            except Exception as e:
                print(f"Error processing image {path}: {e}")

        if batch_imgs:
            yield np.array(batch_imgs, dtype=np.float32), np.array(valid_lbls, dtype=np.int32)


def get_dataset_manifest(
    dataset_name: str,
    project_root: Optional[str] = None
) -> Tuple[List[str], List[int], int]:
    """
    Loads exact test image paths and integer labels for the specified dataset.
    Preserves official test splits and exact training class mappings.

    Returns:
    - test_paths: List of absolute image paths
    - test_labels: List of integer class labels
    - num_classes: Integer count of distinct target classes
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    name = dataset_name.lower().strip()
    images_dir = os.path.join(project_root, "Images")

    if name in ["gtsrb", "german"]:
        dataset_root = os.environ.get(
            "GTSRB_DATASET_ROOT",
            os.path.join(images_dir, "gtsrb-german-traffic-sign", "versions", "1")
        )
        test_csv_path = os.path.join(dataset_root, "Test.csv")
        if not os.path.exists(test_csv_path):
            raise FileNotFoundError(f"GTSRB Test.csv not found at: {test_csv_path}")

        df = pd.read_csv(test_csv_path)
        test_paths = [os.path.join(dataset_root, p) for p in df["Path"]]
        test_labels = df["ClassId"].astype(int).tolist()
        return test_paths, test_labels, 43

    elif name in ["btsd", "belgium", "belgiumtsc"]:
        dataset_root = os.environ.get(
            "BELGIUM_DATASET_ROOT",
            os.path.join(images_dir, "belgium-ts", "versions", "10")
        )
        test_root = os.path.join(dataset_root, "BelgiumTSC_Testing", "Testing")
        if not os.path.exists(test_root):
            raise FileNotFoundError(f"BelgiumTSC test directory not found at: {test_root}")

        test_paths = []
        test_labels = []
        class_folders = sorted(
            [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
        )
        for cf in class_folders:
            class_path = os.path.join(test_root, cf)
            csv_files = [f for f in os.listdir(class_path) if f.lower().endswith(".csv")]
            if not csv_files:
                continue
            csv_file = os.path.join(class_path, csv_files[0])
            df = pd.read_csv(csv_file, sep=";")
            for _, row in df.iterrows():
                fname = str(row["Filename"])
                lbl = int(row["ClassId"])
                img_path = os.path.join(class_path, fname)
                test_paths.append(img_path)
                test_labels.append(lbl)

        return test_paths, test_labels, 62

    elif name in ["ctsd", "chinese"]:
        dataset_root = os.environ.get(
            "CTSD_DATASET_ROOT",
            os.path.join(images_dir, "chinese-traffic-signs", "versions", "2")
        )
        annotation_file = os.path.join(dataset_root, "annotations.csv")
        image_root = os.path.join(dataset_root, "images")
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"CTSD annotations.csv not found at: {annotation_file}")

        df = pd.read_csv(annotation_file)
        all_paths = [os.path.join(image_root, fn) for fn in df["file_name"]]
        all_labels = df["category"].astype(int).values

        # Exact deterministic 80/20 train/test split matching model training (random_state=42)
        _, test_paths, _, test_labels = train_test_split(
            all_paths,
            all_labels,
            test_size=0.20,
            random_state=42,
            stratify=all_labels
        )
        return list(test_paths), [int(l) for l in test_labels], 58

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Must be 'gtsrb', 'btsd', or 'ctsd'.")