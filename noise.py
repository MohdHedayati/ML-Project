"""
Noise Generation and Image Corruption Module.

Provides functions to apply 14 standard corruptions to images at 5 severity levels,
following the noise-first then downscaling pipeline.
"""

import numpy as np
# NumPy 2.0+ compatibility shim for imagecorruptions
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128

import os
import copy
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import tempfile
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from imagecorruptions import corrupt

# The 14 benchmark noise corruptions for GTSRB
NOISE_TYPES = [
    'gaussian_noise', 'impulse_noise', 'shot_noise', 'snow', 'fog',
    'frost', 'motion_blur', 'zoom_blur', 'defocus_blur', 'brightness',
    'contrast', 'pixelate', 'elastic_transform', 'jpeg_compression'
]


def add_noise(image, noise_type: str, severity: int = 1):
    """
    Applies corruption noise to an image.
    
    Args:
        image: PIL.Image.Image or np.ndarray or torch.Tensor.
        noise_type: One of the 14 corruption types in NOISE_TYPES.
        severity: Severity level (1 to 5).
        
    Returns:
        Corrupted image as PIL.Image.Image.
    """
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"Unknown noise type '{noise_type}'. Supported types: {NOISE_TYPES}")
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity must be an integer between 1 and 5, got {severity}")

    # Convert torch.Tensor to PIL Image if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            # Assume (C, H, W)
            to_pil = transforms.ToPILImage()
            image = to_pil(image)
        elif image.dim() == 4 and image.size(0) == 1:
            to_pil = transforms.ToPILImage()
            image = to_pil(image.squeeze(0))
        else:
            raise ValueError(f"Unsupported tensor shape for add_noise: {image.shape}")

    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))

    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # imagecorruptions requires image dimensions >= 32px
    w, h = image.size
    if w < 32 or h < 32:
        image = image.resize((max(w, 32), max(h, 32)))

    image_np = np.array(image)
    corrupted_np = corrupt(image_np, corruption_name=noise_type, severity=severity)
    return Image.fromarray(corrupted_np)


def add_noise_and_scale(image, noise_type: str, severity: int = 1, target_size: tuple = (49, 49)):
    """
    Applies noise corruption FIRST on native/high-resolution image,
    then downscales the corrupted image to target_size.
    
    Args:
        image: PIL.Image.Image, np.ndarray, or torch.Tensor.
        noise_type: Corruption name from NOISE_TYPES.
        severity: Severity level (1 to 5).
        target_size: Desired output resolution (width, height), default (49, 49).
        
    Returns:
        PIL.Image.Image downscaled to target_size.
    """
    # 1. Apply noise first
    corrupted_image = add_noise(image, noise_type, severity)
    
    # 2. Downscale to target_size
    scaled_image = corrupted_image.resize(target_size, resample=Image.Resampling.BILINEAR)
    return scaled_image


class CorruptedGTSRBDataset(Dataset):
    """
    Dataset wrapper that dynamically injects noise corruption and downscales images.
    Works seamlessly with GTSRB, BTSD, and CTSD datasets.
    """
    def __init__(self, base_dataset_or_csv, root_dir='data', noise_type='fog', severity=1, target_size=(49, 49), transform=None):
        """
        Args:
            base_dataset_or_csv: A Dataset instance (GTSRB, BTSD, CTSD) OR path to a CSV annotations file.
            root_dir: Data directory if CSV path is provided.
            noise_type: Noise corruption name from NOISE_TYPES.
            severity: Severity level (1 to 5).
            target_size: Output resolution tuple (default: (49, 49)).
            transform: Optional transform to apply after noise and downscaling. Defaults to ToTensor().
        """
        self.noise_type = noise_type
        self.severity = severity
        self.target_size = target_size
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.base_dataset = base_dataset_or_csv if isinstance(base_dataset_or_csv, Dataset) else None

        if self.base_dataset is not None and hasattr(self.base_dataset, 'get_pil_image'):
            self.length = len(self.base_dataset)
        elif hasattr(base_dataset_or_csv, 'annotations'):
            self.annotations = base_dataset_or_csv.annotations
            self.root_dir = getattr(base_dataset_or_csv, 'root_dir', root_dir)
            self.paths = [os.path.join(self.root_dir, p) for p in self.annotations['Path']]
            self.labels = self.annotations['ClassId'].astype(int).tolist()
            self.length = len(self.paths)
        elif isinstance(base_dataset_or_csv, str):
            import pandas as pd
            df = pd.read_csv(base_dataset_or_csv)
            self.paths = [os.path.join(root_dir, p) for p in df['Path']]
            self.labels = df['ClassId'].astype(int).tolist()
            self.length = len(self.paths)
        else:
            raise TypeError("base_dataset_or_csv must be a Dataset instance or path to CSV file.")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.base_dataset is not None and hasattr(self.base_dataset, 'get_pil_image'):
            image, y_label = self.base_dataset.get_pil_image(index)
        else:
            image = Image.open(self.paths[index]).convert('RGB')
            y_label = self.labels[index]
        
        # Apply noise first, then downscale to target_size
        noisy_scaled_pil = add_noise_and_scale(
            image, 
            noise_type=self.noise_type, 
            severity=self.severity, 
            target_size=self.target_size
        )
        
        image_tensor = self.transform(noisy_scaled_pil)
        return image_tensor, y_label


    def get_metadata(self, index):
        """Returns full sample metadata including corruption type and severity."""
        if self.base_dataset is not None and hasattr(self.base_dataset, 'get_metadata'):
            meta = self.base_dataset.get_metadata(index).copy()
        elif hasattr(self, 'paths'):
            meta = {
                'path': self.paths[index],
                'unified_label': self.labels[index],
                'dataset': 'unknown',
                'original_label': self.labels[index],
                'country': 'unknown'
            }
        else:
            meta = {}
        meta['corruption_type'] = self.noise_type
        meta['severity'] = self.severity
        return meta

    def get_sample_info(self, index):
        """Returns (image_tensor, metadata_dict) for inspection."""
        tensor, _ = self[index]
        return tensor, self.get_metadata(index)


# Backward-compatible alias and universal name
CorruptedDataset = CorruptedGTSRBDataset


def corruption_cache_key(dataset, target_size):
    if not hasattr(dataset, 'paths') or not hasattr(dataset, 'labels'):
        return None
    if len(dataset.paths) != len(dataset) or len(dataset.labels) != len(dataset):
        raise ValueError('Cache sources and labels must match the dataset length.')
    sources = []
    for path, label in zip(dataset.paths, dataset.labels):
        resolved = Path(path).resolve()
        stat = resolved.stat()
        sources.append((str(resolved), stat.st_size, stat.st_mtime_ns, int(label)))
    signature = {
        'sources': sources,
        'size': tuple(target_size),
        'implementation': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'versions': {name: version(name) for name in ('imagecorruptions', 'numpy', 'Pillow', 'scipy', 'scikit-image', 'torchvision')},
    }
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


class CachedCorruptedDataset(Dataset):
    def __init__(self, dataset, cache_dir, batch_size=128, num_workers=0):
        if dataset.noise_type not in NOISE_TYPES or dataset.severity not in range(1, 6):
            raise ValueError('Invalid cached corruption type or severity.')
        self.labels = list(dataset.base_dataset.labels)
        self.transform = dataset.transform
        self.path = Path(cache_dir) / f'{dataset.noise_type}_{dataset.severity}.npy'
        self.images = None
        width, height = dataset.target_size
        shape = (len(dataset), height, width, 3)
        valid = False
        if self.path.exists():
            try:
                saved = np.load(self.path, mmap_mode='r', allow_pickle=False)
                valid = saved.shape == shape and saved.dtype == np.uint8
                del saved
            except (OSError, ValueError, EOFError):
                pass
        if valid:
            print(f'[Cache] Reusing {self.path.name}')
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        print(f'[Cache] Building {self.path.name} ({len(dataset)} images)')
        raw_dataset = copy.copy(dataset)
        raw_dataset.transform = transforms.PILToTensor()
        loader = DataLoader(raw_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        descriptor, temporary_path = tempfile.mkstemp(prefix=self.path.stem + '-', suffix='.tmp', dir=self.path.parent)
        os.close(descriptor)
        images = None
        try:
            images = np.lib.format.open_memmap(temporary_path, mode='w+', dtype=np.uint8, shape=shape)
            offset = 0
            for batch, labels in loader:
                count = len(batch)
                images[offset:offset + count] = batch.permute(0, 2, 3, 1).numpy()
                offset += count
            if offset != len(dataset):
                raise RuntimeError('Corruption cache did not receive every image.')
            images.flush()
            del images
            images = None
            os.replace(temporary_path, self.path)
        finally:
            if images is not None:
                del images
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.images is None:
            self.images = np.load(self.path, mmap_mode='r', allow_pickle=False)
        return self.transform(Image.fromarray(self.images[index])), self.labels[index]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['images'] = None
        return state

