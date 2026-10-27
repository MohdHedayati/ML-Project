"""
Baseline Convolutional Neural Network for GTSRB Traffic Sign Recognition.

Implements the standard model interface expected by eval.py:
- forward(x): returns classification logits for 49x49 inputs
- train(train_dataset, ...): trains model or toggles PyTorch train mode
- test(test_dataset_or_loader, ...): runs inference and returns (all_preds, all_labels)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Baseline(nn.Module):
    """
    Baseline CNN architecture for 49x49 GTSRB images.
    Consists of 3 Conv blocks with GELU activations and Dropout,
    followed by fully-connected classifier layers.
    """
    def __init__(self, num_classes=43):
        super(Baseline, self).__init__()
        self.num_classes = num_classes

        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        self.drop1 = nn.Dropout2d(p=0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.act3 = nn.GELU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier: after three 2x2 poolings, 49x49 becomes 6x6. 128 * 6 * 6 = 4608 features.
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4608, 512)
        self.act4 = nn.GELU()
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.drop1(self.act2(self.conv2(x))))
        x = self.pool3(self.act3(self.conv3(x)))
        x = self.flatten(x)
        x = self.drop2(self.act4(self.fc1(x)))
        x = self.fc2(x)
        return x

    def train(self, dataset_or_mode=True, epochs=5, batch_size=64, lr=0.001, device=None, save_dir='output/baseline', **kwargs):
        """
        Dual-purpose train method:
        1. If passed a boolean (or called with no args), acts as standard PyTorch nn.Module.train(mode).
        2. If passed a Dataset or DataLoader, executes the model training loop.
        """
        # Case 1: Standard PyTorch train mode toggle
        if isinstance(dataset_or_mode, bool) and not kwargs and epochs == 5 and batch_size == 64:
            return super().train(mode=dataset_or_mode)

        # Case 2: Executing training on dataset
        train_data = dataset_or_mode
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        super().train(True)

        if isinstance(train_data, DataLoader):
            train_loader = train_data
        elif isinstance(train_data, Dataset):
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        else:
            raise TypeError("Expected Dataset or DataLoader for training.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        print(f"\n[Baseline] Starting training for {epochs} epochs on device: {device}")
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
            for images, labels in loop:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loop.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    acc=f"{100.0 * correct / total:.2f}%"
                )

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100.0 * correct / total
            print(f"Epoch [{epoch+1}/{epochs}] Finished -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

        # Save model checkpoint
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'baseline_model.pth')
            torch.save(self.state_dict(), save_path)
            print(f"[Baseline] Trained weights saved to '{save_path}'")

        return {"epochs": epochs, "loss": epoch_loss, "accuracy": epoch_acc}

    def test(self, test_dataset_or_loader, batch_size=64, device=None):
        """
        Evaluates the model on clean or corrupted data.
        
        Args:
            test_dataset_or_loader: Dataset or DataLoader with test samples.
            batch_size: Batch size if a Dataset was passed.
            device: torch.device (defaults to GPU if available, else CPU).
            
        Returns:
            tuple: (all_preds, all_labels) as python lists/numpy arrays.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()

        if isinstance(test_dataset_or_loader, DataLoader):
            loader = test_dataset_or_loader
        elif isinstance(test_dataset_or_loader, Dataset):
            loader = DataLoader(test_dataset_or_loader, batch_size=batch_size, shuffle=False)
        else:
            raise TypeError("Expected Dataset or DataLoader for testing.")

        all_preds = []
        all_labels = []

        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                outputs = self(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())

        return all_preds, all_labels

    def load_weights(self, weights_path=None):
        """Loads trained weights from specified path or default locations."""
        if weights_path is None:
            # Check default candidate locations
            candidates = [
                os.path.join('output', 'baseline', 'baseline_model.pth'),
                os.path.join('output', 'baseline_model.pth')
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    weights_path = candidate
                    break

        if weights_path and os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path, weights_only=True, map_location='cpu'))
            print(f"[Baseline] Successfully loaded weights from '{weights_path}'")
            return True
        else:
            print(f"[Baseline] Warning: No checkpoint found at '{weights_path}'")
            return False


# Aliases for backward compatibility
TrafficSignNet = Baseline
BaselineModel = Baseline


if __name__ == "__main__":
    model = Baseline(num_classes=43)
    dummy_input = torch.randn(4, 3, 49, 49)
    out = model(dummy_input)
    print(f"Baseline initialized. Input: {dummy_input.shape} -> Output: {out.shape}")
