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
    Simplified Baseline CNN architecture for 49x49 GTSRB images with Batch Normalization.
    Consists of 3 Conv blocks with BatchNorm2d, ReLU activations, and Dropout,
    followed by fully-connected classifier layers with BatchNorm1d.
    Supports Adaptive Batch Normalization (AdaBN) for test-time adaptation.
    """
    def __init__(self, num_classes=43):
        super(Baseline, self).__init__()
        self.num_classes = num_classes

        # Feature extractor: 3 Conv blocks with BatchNorm, ReLU, Pooling, and Dropout
        self.features = nn.Sequential(
            # Block 1: (3, 49, 49) -> (32, 24, 24)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: (32, 24, 24) -> (64, 12, 12)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: (64, 12, 12) -> (128, 6, 6)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier head: 128 * 6 * 6 = 4608 -> 512 -> num_classes (raw logits)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)  # Raw unnormalized logits output
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def train(self, dataset_or_mode=True, epochs=5, batch_size=64, lr=0.001, device=None, save_dir='output/baseline', **kwargs):
      
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
            pin_mem = (device.type == "cuda")
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
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
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

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

    def adapt_bn(self, target_dataset_or_loader, batch_size=64, device=None, verbose=True):
        """
        Adapts Batch Normalization layers to a target domain using AdaBN (Adaptive Batch Normalization).
        
        Freezes all weights and updates running_mean and running_var for all BatchNorm layers
        using unlabeled target domain samples (e.g., corrupted images).
        
        Args:
            target_dataset_or_loader: Dataset or DataLoader with target samples.
            batch_size: Batch size to use if Dataset was provided.
            device: Device to run adaptation on.
            verbose: Whether to log completion message.
        """
        if device is None:
            device = next(self.parameters()).device
        self.to(device)

        # 1. Reset running stats and set BatchNorm layers to update running average
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.reset_running_stats()
                m.momentum = None  # Computes simple cumulative average over the target data
                m.train()          # Activates stats accumulation
            else:
                m.eval()           # Freezes Dropout and all other non-BN components

        if isinstance(target_dataset_or_loader, DataLoader):
            loader = target_dataset_or_loader
        elif isinstance(target_dataset_or_loader, Dataset):
            loader = DataLoader(target_dataset_or_loader, batch_size=batch_size, shuffle=False)
        else:
            raise TypeError("Expected Dataset or DataLoader for AdaBN adaptation.")

        # 2. Forward pass over target data without computing gradients
        with torch.no_grad():
            for batch in loader:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                images = images.to(device, non_blocking=True)
                _ = self(images)

        # 3. Restore eval mode and standard momentum
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.momentum = 0.1

        if verbose:
            print(f"[Baseline] AdaBN adaptation completed across {len(loader)} target batches.")

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
            try:
                self.load_state_dict(torch.load(weights_path, weights_only=True, map_location='cpu'))
                print(f"[Baseline] Successfully loaded weights from '{weights_path}'")
                return True
            except RuntimeError as e:
                print(f"[Baseline] Checkpoint at '{weights_path}' is from an older architecture without BatchNorm.")
                print(f"[Baseline] Retraining is needed to save new weights with BatchNorm.")
                return False
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
