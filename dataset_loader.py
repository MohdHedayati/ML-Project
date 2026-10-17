"""
Dataset Loader Module for the German Traffic Sign Recognition Benchmark (GTSRB).

Loads images and annotations from CSV files and scales them down to the target
resolution (default: 49x49) as PyTorch tensors.
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class GTSRBDataset(Dataset):
    """
    Dataset class for loading GTSRB traffic sign images.
    """
    def __init__(self, csv_file, root_dir='data', transform=None, target_size=(49, 49)):
        """
        Args:
            csv_file (str): Path to the CSV annotations file (e.g., 'data/Train.csv').
            root_dir (str): Root directory for images (default: 'data').
            transform (callable, optional): Transform to be applied on a sample.
            target_size (tuple, optional): Target size (width, height) to resize if transform is None.
        """
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.target_size = target_size
        self.annotations = pd.read_csv(csv_file)
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index]['Path'])
        image = Image.open(img_path).convert('RGB')
        y_label = int(self.annotations.iloc[index]['ClassId'])

        if self.transform:
            image = self.transform(image)

        return image, y_label

    def get_pil_image(self, index):
        """Returns the uncorrupted, original PIL Image and label for visualization or custom pipeline."""
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index]['Path'])
        image = Image.open(img_path).convert('RGB')
        y_label = int(self.annotations.iloc[index]['ClassId'])
        return image, y_label


def get_train_dataset(csv_file='data/Train.csv', root_dir='data', target_size=(49, 49), transform=None):
    """Helper function to load the training dataset."""
    return GTSRBDataset(csv_file=csv_file, root_dir=root_dir, transform=transform, target_size=target_size)


def get_test_dataset(csv_file='data/Test.csv', root_dir='data', target_size=(49, 49), transform=None):
    """Helper function to load the test dataset."""
    return GTSRBDataset(csv_file=csv_file, root_dir=root_dir, transform=transform, target_size=target_size)


def get_train_loader(dataset=None, batch_size=64, shuffle=True, num_workers=0, target_size=(49, 49)):
    """Helper function to create a DataLoader for training."""
    if dataset is None:
        dataset = get_train_dataset(target_size=target_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_test_loader(dataset=None, batch_size=64, shuffle=False, num_workers=0, target_size=(49, 49)):
    """Helper function to create a DataLoader for testing."""
    if dataset is None:
        dataset = get_test_dataset(target_size=target_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    train_dataset = get_train_dataset()
    img, label = train_dataset[0]
    print(f"DatasetLoader test: Image shape: {img.shape}, Label: {label}, Total samples: {len(train_dataset)}")