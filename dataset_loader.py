"""
Dataset Loader Module for Traffic Sign Recognition.

Supports loading 3 independent traffic sign benchmarks:
- GTSRB (German Traffic Sign Recognition Benchmark) — 43 classes (DEFAULT)
- BTSD (Belgian Traffic Sign Dataset) — 62 classes
- CTSD (Chinese Traffic Sign Database) — 58 classes

By default, loads GTSRB. The source for loading images changes based on the selected dataset,
while keeping everything else in the pipeline identical.
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
    'ctsd': 58,
}

DATASET_COUNTRIES = {
    'gtsrb': 'Germany',
    'btsd': 'Belgium',
    'ctsd': 'China',
}

DEFAULT_DATASET = 'gtsrb'
TOTAL_CLASSES = 43  # Default GTSRB class count for pipeline compatibility
NUM_CLASSES = 43


class TrafficSignDataset(Dataset):
    """
    Dataset loader supporting GTSRB, BTSD, and CTSD datasets.
    By default, loads GTSRB (43 classes).
    
    Each sample preserves:
    - path: filesystem path to image
    - label: native dataset class ID (0 to num_classes - 1)
    - dataset: source benchmark name ('gtsrb', 'btsd', or 'ctsd')
    - country: country of origin ('Germany', 'Belgium', or 'China')
    """
    def __init__(
        self,
        dataset='gtsrb',
        split='train',
        root_dir='data',
        target_size=(49, 49),
        transform=None,
        **kwargs
    ):
        self.root_dir = root_dir
        self.split = 'train' if str(split).lower() in ['train', 'training'] else 'test'
        self.target_size = target_size

        # Transforms: Resize to target_size and convert to Tensor
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor()
            ])

        # Normalize dataset selector (default: gtsrb)
        ds_str = str(dataset).lower().strip() if dataset is not None else DEFAULT_DATASET
        if ds_str not in DATASET_CLASSES:
            ds_str = DEFAULT_DATASET
        self.dataset_name = ds_str
        self.num_classes = DATASET_CLASSES[self.dataset_name]
        self.country = DATASET_COUNTRIES[self.dataset_name]

        self.samples = []
        if self.dataset_name == 'gtsrb':
            self._load_gtsrb()
        elif self.dataset_name == 'btsd':
            self._load_btsd()
        elif self.dataset_name == 'ctsd':
            self._load_ctsd()

        self.paths = [s[0] for s in self.samples]
        self.labels = [int(s[1]) for s in self.samples]

        # Fast in-memory caches
        self.cache = [None] * len(self.samples)
        self.pil_cache = [None] * len(self.samples)

    def _load_gtsrb(self):
        """Loads GTSRB samples (43 classes)."""
        csv_name = 'Train.csv' if self.split == 'train' else 'Test.csv'
        csv_candidates = [
            os.path.join(self.root_dir, csv_name),
            os.path.join(self.root_dir, 'gtsrb', csv_name),
            os.path.join('data', csv_name)
        ]
        target_csv = next((c for c in csv_candidates if os.path.exists(c)), None)
        if not target_csv:
            return

        df = pd.read_csv(target_csv)
        csv_dir = os.path.dirname(target_csv) or '.'
        
        # Fast prefix resolution using first sample
        first_rel = str(df['Path'].iloc[0])
        cand_prefixes = [self.root_dir, csv_dir, 'data']
        chosen_prefix = next((p for p in cand_prefixes if os.path.exists(os.path.join(p, first_rel))), self.root_dir)

        paths = df['Path'].astype(str).tolist()
        class_ids = df['ClassId'].astype(int).tolist()
        self.samples.extend([
            (os.path.join(chosen_prefix, p).replace('\\', '/'), cid, 'gtsrb', 'Germany')
            for p, cid in zip(paths, class_ids)
        ])

    def _load_btsd(self):
        """Loads BTSD samples (62 classes)."""
        btsd_root_candidates = [
            os.path.join(self.root_dir, 'btsd'),
            os.path.join(self.root_dir, 'belgium-ts'),
            os.path.join('data', 'btsd'),
            self.root_dir
        ]
        btsd_root = next((r for r in btsd_root_candidates if os.path.exists(os.path.join(r, 'BelgiumTSC_Training')) or os.path.exists(os.path.join(r, 'Training'))), None)
        if not btsd_root:
            return

        split_sub = 'BelgiumTSC_Training/Training' if self.split == 'train' else 'BelgiumTSC_Testing/Testing'
        data_dir = os.path.join(btsd_root, split_sub)
        if not os.path.exists(data_dir):
            alt_sub = 'Training' if self.split == 'train' else 'Testing'
            data_dir = os.path.join(btsd_root, alt_sub)
            if not os.path.exists(data_dir):
                return

        class_folders = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

        for cf in class_folders:
            cpath = os.path.join(data_dir, cf)
            csv_files = [f for f in os.listdir(cpath) if f.lower().endswith('.csv')]
            if csv_files:
                csv_file = os.path.join(cpath, csv_files[0])
                df = pd.read_csv(csv_file, sep=';')
                for _, row in df.iterrows():
                    img_path = os.path.join(cpath, str(row['Filename']))
                    class_id = int(row['ClassId'])
                    self.samples.append((img_path, class_id, 'btsd', 'Belgium'))
            else:
                try:
                    class_id = int(cf)
                    for fn in os.listdir(cpath):
                        if fn.lower().endswith(('.png', '.ppm', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(cpath, fn), class_id, 'btsd', 'Belgium'))
                except ValueError:
                    pass

    def _load_ctsd(self):
        """Loads CTSD samples (58 classes) with 80/20 train/test split."""
        ctsd_root_candidates = [
            os.path.join(self.root_dir, 'ctsd'),
            os.path.join(self.root_dir, 'chinese-traffic-signs'),
            os.path.join('data', 'ctsd'),
            self.root_dir
        ]
        ctsd_root = next((r for r in ctsd_root_candidates if os.path.exists(os.path.join(r, 'annotations.csv'))), None)
        if not ctsd_root:
            return

        ann_csv = os.path.join(ctsd_root, 'annotations.csv')
        img_dir = os.path.join(ctsd_root, 'images') if os.path.exists(os.path.join(ctsd_root, 'images')) else ctsd_root

        df = pd.read_csv(ann_csv)
        all_paths = [os.path.join(img_dir, fn) for fn in df['file_name']]
        all_labels = df['category'].astype(int).values

        train_p, test_p, train_l, test_l = train_test_split(
            all_paths, all_labels, test_size=0.20, random_state=42, stratify=all_labels
        )

        target_p, target_l = (train_p, train_l) if self.split == 'train' else (test_p, test_l)

        for p, orig_l in zip(target_p, target_l):
            class_id = int(orig_l)
            self.samples.append((p, class_id, 'ctsd', 'China'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cached = self.cache[index]
        if cached is not None:
            return cached, self.labels[index]

        img_path, label = self.samples[index][0], self.samples[index][1]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        self.cache[index] = image
        return image, label

    def get_pil_image(self, index):
        """Returns uncorrupted, original PIL Image and label (cached in RAM)."""
        cached = self.pil_cache[index]
        if cached is not None:
            return cached, self.labels[index]

        img_path, label = self.samples[index][0], self.samples[index][1]
        image = Image.open(img_path).convert('RGB')
        self.pil_cache[index] = (image, label)
        return image, label

    def get_metadata(self, index):
        """Returns sample metadata dict."""
        s = self.samples[index]
        return {
            'path': s[0],
            'label': s[1],
            'dataset': s[2],
            'country': s[3]
        }


# Aliases
GTSRBDataset = TrafficSignDataset


def get_train_dataset(dataset='gtsrb', root_dir='data', target_size=(49, 49), **kwargs):
    """Factory function for training dataset (default: gtsrb)."""
    return TrafficSignDataset(dataset=dataset, split='train', root_dir=root_dir, target_size=target_size, **kwargs)


def get_test_dataset(dataset='gtsrb', root_dir='data', target_size=(49, 49), **kwargs):
    """Factory function for test dataset (default: gtsrb)."""
    return TrafficSignDataset(dataset=dataset, split='test', root_dir=root_dir, target_size=target_size, **kwargs)


def get_train_loader(dataset='gtsrb', batch_size=64, shuffle=True, num_workers=0, target_size=(49, 49), root_dir='data', **kwargs):
    """Returns DataLoader for training set."""
    ds = get_train_dataset(dataset=dataset, root_dir=root_dir, target_size=target_size, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_test_loader(dataset='gtsrb', batch_size=64, shuffle=False, num_workers=0, target_size=(49, 49), root_dir='data', **kwargs):
    """Returns DataLoader for test set."""
    ds = get_test_dataset(dataset=dataset, root_dir=root_dir, target_size=target_size, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
