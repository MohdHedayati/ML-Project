"""
Dataset Loader Module for Traffic Sign Recognition Benchmarks:
- GTSRB (German Traffic Sign Recognition Benchmark) — 43 classes
- BTSD (Belgian Traffic Sign Dataset) — 62 classes
- CTSD (Chinese Traffic Sign Dataset) — 58 classes

Loads images and annotations and scales them down to the target
resolution (default: 49x49) as PyTorch tensors.
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


DATASET_CLASSES = {
    'gtsrb': 43,
    'btsd': 62,
    'ctsd': 58
}


class GTSRBDataset(Dataset):
    """
    Dataset class for loading GTSRB traffic sign images.
    """
    def __init__(self, csv_file='data/Train.csv', root_dir='data', transform=None, target_size=(49, 49)):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.target_size = target_size
        
        # Fallback to Images/ folder if data/ CSV does not exist
        if not os.path.exists(csv_file):
            alt_root = os.path.join('Images', 'gtsrb-german-traffic-sign', 'versions', '1')
            alt_csv = os.path.join(alt_root, os.path.basename(csv_file))
            if os.path.exists(alt_csv):
                self.csv_file = alt_csv
                self.root_dir = alt_root

        self.annotations = pd.read_csv(self.csv_file)
        self.paths = [os.path.join(self.root_dir, p) for p in self.annotations['Path']]
        self.labels = self.annotations['ClassId'].astype(int).tolist()
        self.cache = [None] * len(self.paths)
        self.pil_cache = [None] * len(self.paths)
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        cached = self.cache[index]
        if cached is not None:
            return cached, self.labels[index]

        image = Image.open(self.paths[index]).convert('RGB')
        y_label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        self.cache[index] = image
        return image, y_label

    def get_pil_image(self, index):
        """Returns the uncorrupted, original PIL Image and label (cached in RAM)."""
        cached = self.pil_cache[index]
        if cached is not None:
            return cached, self.labels[index]
        image = Image.open(self.paths[index]).convert('RGB')
        self.pil_cache[index] = image
        return image, self.labels[index]


class BTSDDataset(Dataset):
    """
    Dataset class for loading Belgian Traffic Sign Dataset (BTSD) images.
    Organized into class-named folders with semicolon-delimited GT-xxxxx.csv files.
    """
    def __init__(self, root_dir='Images/belgium-ts/versions/10', split='train', transform=None, target_size=(49, 49)):
        self.root_dir = root_dir
        self.split = split.lower()
        self.target_size = target_size

        if self.split in ['train', 'training']:
            self.data_dir = os.path.join(self.root_dir, 'BelgiumTSC_Training', 'Training')
        else:
            self.data_dir = os.path.join(self.root_dir, 'BelgiumTSC_Testing', 'Testing')

        self.samples = []
        if os.path.exists(self.data_dir):
            class_folders = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
            for cf in class_folders:
                cpath = os.path.join(self.data_dir, cf)
                csv_files = [f for f in os.listdir(cpath) if f.lower().endswith('.csv')]
                if not csv_files:
                    continue
                csv_file = os.path.join(cpath, csv_files[0])
                df = pd.read_csv(csv_file, sep=';')
                for _, row in df.iterrows():
                    img_path = os.path.join(cpath, str(row['Filename']))
                    lbl = int(row['ClassId'])
                    self.samples.append((img_path, lbl))

        self.cache = [None] * len(self.samples)
        self.pil_cache = [None] * len(self.samples)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cached = self.cache[index]
        if cached is not None:
            return cached, self.samples[index][1]

        img_path, label = self.samples[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        self.cache[index] = image
        return image, label

    def get_pil_image(self, index):
        """Returns the uncorrupted, original PIL Image and label (cached in RAM)."""
        cached = self.pil_cache[index]
        if cached is not None:
            return cached, self.samples[index][1]
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        self.pil_cache[index] = image
        return image, label


class CTSDDataset(Dataset):
    """
    Dataset class for loading Chinese Traffic Sign Dataset (CTSD) images.
    Uses annotations.csv with a deterministic 80/20 train/test split.
    """
    def __init__(self, root_dir='Images/chinese-traffic-signs/versions/2', split='train', transform=None, target_size=(49, 49)):
        self.root_dir = root_dir
        self.split = split.lower()
        self.target_size = target_size

        csv_file = os.path.join(self.root_dir, 'annotations.csv')
        image_dir = os.path.join(self.root_dir, 'images')

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CTSD annotations not found at: {csv_file}")

        df = pd.read_csv(csv_file)
        all_paths = [os.path.join(image_dir, fn) for fn in df['file_name']]
        all_labels = df['category'].astype(int).values

        # Deterministic 80/20 split matching model training
        train_p, test_p, train_l, test_l = train_test_split(
            all_paths, all_labels, test_size=0.20, random_state=42, stratify=all_labels
        )

        if self.split in ['train', 'training']:
            self.paths = list(train_p)
            self.labels = [int(l) for l in train_l]
        else:
            self.paths = list(test_p)
            self.labels = [int(l) for l in test_l]

        self.cache = [None] * len(self.paths)
        self.pil_cache = [None] * len(self.paths)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        cached = self.cache[index]
        if cached is not None:
            return cached, self.labels[index]

        img_path = self.paths[index]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        self.cache[index] = image
        return image, label

    def get_pil_image(self, index):
        """Returns the uncorrupted, original PIL Image and label (cached in RAM)."""
        cached = self.pil_cache[index]
        if cached is not None:
            return cached, self.labels[index]
        img_path = self.paths[index]
        image = Image.open(img_path).convert('RGB')
        self.pil_cache[index] = image
        return image, self.labels[index]


def get_train_dataset(dataset='gtsrb', target_size=(49, 49), transform=None, **kwargs):
    """Helper function to load the training dataset for gtsrb, btsd, or ctsd."""
    ds = dataset.lower().strip()
    if ds == 'gtsrb':
        csv_file = kwargs.get('csv_file', 'data/Train.csv')
        root_dir = kwargs.get('root_dir', 'data')
        return GTSRBDataset(csv_file=csv_file, root_dir=root_dir, transform=transform, target_size=target_size)
    elif ds in ['btsd', 'belgium']:
        root_dir = kwargs.get('root_dir', 'Images/belgium-ts/versions/10')
        return BTSDDataset(root_dir=root_dir, split='train', transform=transform, target_size=target_size)
    elif ds in ['ctsd', 'chinese']:
        root_dir = kwargs.get('root_dir', 'Images/chinese-traffic-signs/versions/2')
        return CTSDDataset(root_dir=root_dir, split='train', transform=transform, target_size=target_size)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Supported datasets: {list(DATASET_CLASSES.keys())}")


def get_test_dataset(dataset='gtsrb', target_size=(49, 49), transform=None, **kwargs):
    """Helper function to load the test dataset for gtsrb, btsd, or ctsd."""
    ds = dataset.lower().strip()
    if ds == 'gtsrb':
        csv_file = kwargs.get('csv_file', 'data/Test.csv')
        root_dir = kwargs.get('root_dir', 'data')
        return GTSRBDataset(csv_file=csv_file, root_dir=root_dir, transform=transform, target_size=target_size)
    elif ds in ['btsd', 'belgium']:
        root_dir = kwargs.get('root_dir', 'Images/belgium-ts/versions/10')
        return BTSDDataset(root_dir=root_dir, split='test', transform=transform, target_size=target_size)
    elif ds in ['ctsd', 'chinese']:
        root_dir = kwargs.get('root_dir', 'Images/chinese-traffic-signs/versions/2')
        return CTSDDataset(root_dir=root_dir, split='test', transform=transform, target_size=target_size)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Supported datasets: {list(DATASET_CLASSES.keys())}")


def get_train_loader(dataset=None, dataset_name='gtsrb', batch_size=64, shuffle=True, num_workers=0, target_size=(49, 49)):
    """Helper function to create a DataLoader for training."""
    if dataset is None:
        dataset = get_train_dataset(dataset=dataset_name, target_size=target_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_test_loader(dataset=None, dataset_name='gtsrb', batch_size=64, shuffle=False, num_workers=0, target_size=(49, 49)):
    """Helper function to create a DataLoader for testing."""
    if dataset is None:
        dataset = get_test_dataset(dataset=dataset_name, target_size=target_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    for dname, ncls in DATASET_CLASSES.items():
        try:
            train_ds = get_train_dataset(dataset=dname)
            test_ds = get_test_dataset(dataset=dname)
            img, label = train_ds[0]
            print(f"[{dname.upper()}] Classes: {ncls} | Train samples: {len(train_ds)} | Test samples: {len(test_ds)} | Image shape: {img.shape}")
        except Exception as e:
            print(f"[{dname.upper()}] Error loading: {e}")