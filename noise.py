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
from PIL import Image
import torch
from torch.utils.data import Dataset
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
    Works seamlessly with PyTorch DataLoader multiprocessing.
    """
    def __init__(self, base_dataset_or_csv, root_dir='data', noise_type='fog', severity=1, target_size=(49, 49), transform=None):
        """
        Args:
            base_dataset_or_csv: A GTSRBDataset instance OR path to a CSV annotations file.
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

        if hasattr(base_dataset_or_csv, 'annotations'):
            self.annotations = base_dataset_or_csv.annotations
            self.root_dir = getattr(base_dataset_or_csv, 'root_dir', root_dir)
        elif isinstance(base_dataset_or_csv, str):
            import pandas as pd
            self.annotations = pd.read_csv(base_dataset_or_csv)
            self.root_dir = root_dir
        else:
            raise TypeError("base_dataset_or_csv must be a GTSRBDataset or path to CSV file.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index]['Path'])
        image = Image.open(img_path).convert('RGB')
        
        # Apply noise first, then downscale to target_size
        noisy_scaled_pil = add_noise_and_scale(
            image, 
            noise_type=self.noise_type, 
            severity=self.severity, 
            target_size=self.target_size
        )
        
        image_tensor = self.transform(noisy_scaled_pil)
        y_label = int(self.annotations.iloc[index]['ClassId'])
        return image_tensor, y_label
