"""
Unified Multi-Dataset Loader Module for Traffic Sign Recognition.

Combines all 3 international traffic sign benchmarks into a single, unified dataset
with authoritative semantic label mapping:
- GTSRB (German Traffic Sign Recognition Benchmark) — 43 classes (Germany)
- BTSD (Belgian Traffic Sign Dataset) — 62 classes (Belgium)
- CTSD (Chinese Traffic Sign Database) — 58 classes (China)

Total unified semantic classes: 109 (indexed from 0 to 108)
Replaces previous label offsets (0..42, 43..104, 105..162) with semantic equivalence
where equivalent signs across countries (e.g. Stop, Yield, No Entry, Roundabout, etc.)
share identical unified class IDs while country-specific signs retain unique IDs.

Every sample preserves:
- image path
- unified semantic label ID
- original dataset name ('gtsrb', 'btsd', 'ctsd')
- original class ID
- country of origin ('Germany', 'Belgium', 'China')
"""

import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Load Unified Semantic Label Mapping
# ---------------------------------------------------------------------------
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MAPPING_CANDIDATES = [
    os.path.join(_CURRENT_DIR, 'unified_label_mapping.json'),
    os.path.join('.', 'unified_label_mapping.json'),
    'unified_label_mapping.json',
]
_MAPPING_FILE = next((p for p in _MAPPING_CANDIDATES if os.path.exists(p)), _MAPPING_CANDIDATES[0])

if os.path.exists(_MAPPING_FILE):
    with open(_MAPPING_FILE, 'r', encoding='utf-8') as _f:
        UNIFIED_MAPPING_DATA = json.load(_f)
else:
    raise FileNotFoundError(f"Unified label mapping file not found at candidates: {_MAPPING_CANDIDATES}")

TOTAL_CLASSES = UNIFIED_MAPPING_DATA['metadata']['total_unified_classes']  # 109
UNIFIED_CLASSES = UNIFIED_MAPPING_DATA['unified_classes']
UNIFIED_CLASS_NAMES = [c['name'] for c in UNIFIED_CLASSES]

# Fast O(1) integer lookup mappings: {dataset: {orig_id: unified_id}}
UNIFIED_MAPPINGS = {
    ds: {int(k): int(v) for k, v in m.items()}
    for ds, m in UNIFIED_MAPPING_DATA['mappings'].items()
}

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


class TrafficSignDataset(Dataset):
    """
    Unified Dataset class that loads and combines images from GTSRB, BTSD, and CTSD
    into one single dataset using unified semantic class IDs (109 classes).
    
    Each sample preserves comprehensive metadata:
    - path: filesystem path to image
    - unified_label: semantic class ID (0 to 108)
    - dataset: source dataset name ('gtsrb', 'btsd', or 'ctsd')
    - original_label: raw class ID in the source benchmark
    - country: country of origin ('Germany', 'Belgium', or 'China')
    """
    def __init__(
        self,
        dataset='all',
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

        # Normalize dataset selector
        ds_str = str(dataset).lower().strip() if dataset is not None else 'all'
        if ds_str in ['all', 'combined', 'universal', 'merged']:
            self.active_datasets = ['gtsrb', 'btsd', 'ctsd']
            self.num_classes = TOTAL_CLASSES
        else:
            self.active_datasets = [ds_str]
            self.num_classes = TOTAL_CLASSES

        self.samples = []
        self._load_all_samples()

        self.paths = [s[0] for s in self.samples]
        self.labels = [int(s[1]) for s in self.samples]

        # Fast in-memory caches
        self.cache = [None] * len(self.samples)
        self.pil_cache = [None] * len(self.samples)

    def _load_all_samples(self):
        """Loads samples from each active dataset mapped to unified semantic class IDs."""
        for ds_name in self.active_datasets:
            if ds_name == 'gtsrb':
                self._load_gtsrb()
            elif ds_name == 'btsd':
                self._load_btsd()
            elif ds_name == 'ctsd':
                self._load_ctsd()

    def _load_gtsrb(self):
        """Loads GTSRB samples mapped to unified semantic labels."""
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
        mapping = UNIFIED_MAPPINGS['gtsrb']

        for _, row in df.iterrows():
            rel_path = str(row['Path'])
            cand_paths = [
                os.path.join(self.root_dir, rel_path),
                os.path.join(csv_dir, rel_path),
                os.path.join('data', rel_path)
            ]
            img_path = next((p for p in cand_paths if os.path.exists(p)), cand_paths[0])
            orig_id = int(row['ClassId'])
            unified_id = mapping.get(orig_id, orig_id)
            self.samples.append((img_path, unified_id, 'gtsrb', orig_id, 'Germany'))

    def _load_btsd(self):
        """Loads BTSD samples mapped to unified semantic labels."""
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

        mapping = UNIFIED_MAPPINGS['btsd']
        class_folders = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

        for cf in class_folders:
            cpath = os.path.join(data_dir, cf)
            csv_files = [f for f in os.listdir(cpath) if f.lower().endswith('.csv')]
            if csv_files:
                csv_file = os.path.join(cpath, csv_files[0])
                df = pd.read_csv(csv_file, sep=';')
                for _, row in df.iterrows():
                    img_path = os.path.join(cpath, str(row['Filename']))
                    orig_id = int(row['ClassId'])
                    unified_id = mapping.get(orig_id, orig_id)
                    self.samples.append((img_path, unified_id, 'btsd', orig_id, 'Belgium'))
            else:
                try:
                    orig_id = int(cf)
                    unified_id = mapping.get(orig_id, orig_id)
                    for fn in os.listdir(cpath):
                        if fn.lower().endswith(('.png', '.ppm', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(cpath, fn), unified_id, 'btsd', orig_id, 'Belgium'))
                except ValueError:
                    pass

    def _load_ctsd(self):
        """Loads CTSD samples mapped to unified semantic labels."""
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

        mapping = UNIFIED_MAPPINGS['ctsd']
        target_p, target_l = (train_p, train_l) if self.split == 'train' else (test_p, test_l)

        for p, orig_l in zip(target_p, target_l):
            orig_id = int(orig_l)
            unified_id = mapping.get(orig_id, orig_id)
            self.samples.append((p, unified_id, 'ctsd', orig_id, 'China'))

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
        """Returns the uncorrupted, original PIL Image and unified label (cached in RAM)."""
        cached = self.pil_cache[index]
        if cached is not None:
            return cached, self.labels[index]

        img_path, label = self.samples[index][0], self.samples[index][1]
        image = Image.open(img_path).convert('RGB')
        self.pil_cache[index] = image
        return image, label

    def get_metadata(self, index):
        """
        Returns full provenance and semantic metadata for sample at index:
        path, unified_label, dataset, original_label, country, class_name.
        """
        s = self.samples[index]
        unified_id = int(s[1])
        class_name = UNIFIED_CLASS_NAMES[unified_id] if unified_id < len(UNIFIED_CLASS_NAMES) else f"Class_{unified_id}"
        return {
            'path': s[0],
            'unified_label': unified_id,
            'dataset': s[2],
            'original_label': int(s[3]),
            'country': s[4],
            'class_name': class_name
        }


# Universal aliases
CombinedTrafficSignDataset = TrafficSignDataset
GTSRBDataset = TrafficSignDataset
BTSDDataset = TrafficSignDataset
CTSDDataset = TrafficSignDataset


def get_train_dataset(dataset='all', root_dir='data', target_size=(49, 49), transform=None, **kwargs):
    """Loads all 3 datasets combined into a single training set (109 unified classes)."""
    return TrafficSignDataset(dataset=dataset, split='train', root_dir=root_dir, target_size=target_size, transform=transform, **kwargs)


def get_test_dataset(dataset='all', root_dir='data', target_size=(49, 49), transform=None, **kwargs):
    """Loads all 3 datasets combined into a single testing set (109 unified classes)."""
    return TrafficSignDataset(dataset=dataset, split='test', root_dir=root_dir, target_size=target_size, transform=transform, **kwargs)


def get_train_loader(dataset=None, batch_size=64, shuffle=True, num_workers=0, target_size=(49, 49), root_dir='data', **kwargs):
    """Returns DataLoader for the combined training set."""
    if dataset is None:
        dataset = get_train_dataset(root_dir=root_dir, target_size=target_size, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_test_loader(dataset=None, batch_size=64, shuffle=False, num_workers=0, target_size=(49, 49), root_dir='data', **kwargs):
    """Returns DataLoader for the combined test set."""
    if dataset is None:
        dataset = get_test_dataset(root_dir=root_dir, target_size=target_size, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    train_ds = get_train_dataset()
    test_ds = get_test_dataset()
    img, lbl = train_ds[0]
    meta = train_ds.get_metadata(0)
    print(f"Combined Dataset: Train={len(train_ds):,}, Test={len(test_ds):,}, Classes={train_ds.num_classes}, Shape={tuple(img.shape)}")
    print(f"Sample 0 Metadata: {meta}")