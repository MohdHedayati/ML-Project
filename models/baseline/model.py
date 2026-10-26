"""
Baseline Convolutional Neural Network for Traffic Sign Recognition.

Implements the standard model interface expected by eval.py:
- forward(x): returns raw classification logits for 49x49 inputs
- train(train_dataset, ...): trains model or toggles PyTorch train mode
- test(test_dataset_or_loader, ...): runs inference and returns (all_preds, all_labels)
- load_weights(path): loads checkpoint weights
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from dataset_loader import TOTAL_CLASSES
except ImportError:
    TOTAL_CLASSES = 109


class Baseline(nn.Module):
    """
    Standard Baseline CNN architecture for 49x49 Traffic Sign images.
    Consists of 3 Conv blocks with BatchNorm2d, ReLU activations, and Dropout,
    followed by fully-connected classifier layers with BatchNorm1d.
    """
    def __init__(self, num_classes=TOTAL_CLASSES):
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
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def train(self, dataset_or_mode=True, epochs=5, batch_size=64, lr=0.001, device=None, save_dir='output/baseline', **kwargs):
        """
        Dual-purpose train method:
        - train(True/False): PyTorch train mode toggle
        - train(dataset, epochs=..., batch_size=...): Model training loop
        """
        if isinstance(dataset_or_mode, bool) and not kwargs and epochs == 5 and batch_size == 64:
            return super().train(mode=dataset_or_mode)

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
        epoch_loss = 0.0
        epoch_acc = 0.0
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

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'baseline_model.pth')
            torch.save(self.state_dict(), save_path)
            print(f"[Baseline] Trained weights saved to '{save_path}'")

            # Save model metadata
            meta_path = os.path.join(save_dir, 'model_metadata.json')
            meta_content = {
                "model_name": "Baseline",
                "num_classes": self.num_classes,
                "total_unified_classes": TOTAL_CLASSES,
                "input_size": [3, 49, 49],
                "dataset": "GTSRB + BTSD + CTSD (Unified)",
                "mapping_file": "unified_label_mapping.json",
                "epochs": epochs,
                "loss": epoch_loss,
                "accuracy": epoch_acc
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_content, f, indent=2)
            print(f"[Baseline] Model metadata saved to '{meta_path}'")

        return {"epochs": epochs, "loss": epoch_loss, "accuracy": epoch_acc}

    def test(self, test_dataset_or_loader, batch_size=64, device=None):
        """Runs test inference returning (all_preds, all_labels)."""
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
            candidates = [
                os.path.join('output', 'baseline', 'baseline_model.pth'),
                os.path.join('models', 'baseline', 'baseline_model.pth'),
                os.path.join('output', 'adabn', 'adabn_model.pth'),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    weights_path = candidate
                    break

        if weights_path and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, weights_only=True, map_location='cpu')
                if isinstance(state_dict, dict) and 'classifier.5.weight' in state_dict:
                    ckpt_classes = state_dict['classifier.5.weight'].shape[0]
                    if ckpt_classes != self.num_classes:
                        print(f"[Baseline] Checkpoint classes ({ckpt_classes}) != model classes ({self.num_classes}). Retraining needed.")
                        return False
                self.load_state_dict(state_dict)
                print(f"[Baseline] Successfully loaded weights from '{weights_path}'")
                return True
            except Exception as e:
                print(f"[Baseline] Note: Could not load '{weights_path}': {e}")
                return False
        return False


# Aliases
BaselineModel = Baseline
TrafficSignNet = Baseline


if __name__ == "__main__":
    model = Baseline(num_classes=TOTAL_CLASSES)
    x = torch.randn(2, 3, 49, 49)
    print(f"[Baseline] Forward pass: {x.shape} -> {model(x).shape} (Classes={TOTAL_CLASSES})")
