import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class GTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # The GTSRB CSV has a 'Path' column with the relative image path
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index]['Path'])
        image = Image.open(img_path)
        y_label = int(self.annotations.iloc[index]['ClassId'])

        if self.transform:
            image = self.transform(image)

        return image, y_label

# Let's test if it successfully loads and resizes the first image
if __name__ == "__main__":
    # Resize to 49x49 and convert to a PyTorch tensor
    my_transform = transforms.Compose([
        transforms.Resize((49, 49)), 
        transforms.ToTensor()
    ])
    
    train_dataset = GTSRBDataset(csv_file='data/Train.csv', root_dir='data', transform=my_transform)
    
    # Grab the first image and label
    img, label = train_dataset[0]
    
    print(f"Success! Image shape: {img.shape}, Label: {label}")