---
name: model-interface
description: >-
  Defines the architecture and interface contract for GTSRB traffic sign recognition models evaluated by eval.py.
  Use this skill whenever implementing, refactoring, or integrating a new neural network model
  into the project to ensure seamless compatibility with dataset loading, clean evaluation, and robustness benchmarks.
---

# Modular Model Interface Specification

This skill documents the standard architecture interface, lifecycle contract, and directory conventions required for models plugging into the modular evaluation pipeline (`eval.py`).

---

## 1. Directory & File Organization

Each model in the project must reside in its own dedicated directory under `models/`:

```text
models/
├── baseline/
│   └── model.py          # Baseline CNN implementation
└── <new_model_name>/
    └── model.py          # New model class adhering to this interface
```

Import model classes directly from their module:
```python
from models.baseline.model import Baseline
```

---

## 2. Model Class Interface Contract

Any model class evaluated by `Evaluator.eval_model(model_cls, model_name)` must conform to the following interface:

### 2.1 Class Signature & Inheritance
The model must inherit from `torch.nn.Module`:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes=43):
        super(MyModel, self).__init__()
        # Layers definition...
```

### 2.2 Input & Output Dimensions
* **Input tensor**: `(Batch_Size, 3, 49, 49)` (RGB images scaled down to $49 \times 49$ pixels).
* **Output logits**: `(Batch_Size, 43)` (unnormalized raw logits for the 43 GTSRB classes).

```python
    def forward(self, x):
        # x is of shape (B, 3, 49, 49)
        # Returns logits of shape (B, num_classes)
        return logits
```

### 2.3 Dual-Purpose `train(...)` Method
Because PyTorch's `nn.Module` has a built-in `train(mode: bool = True)` method, the model's `train` method must gracefully support both mode toggling and dataset training:

```python
    def train(self, dataset_or_mode=True, epochs=5, batch_size=64, lr=0.001, device=None, save_dir=None, **kwargs):
        # 1. PyTorch train mode toggle
        if isinstance(dataset_or_mode, bool) and not kwargs and epochs == 5 and batch_size == 64:
            return super().train(mode=dataset_or_mode)

        # 2. Dataset training loop
        train_data = dataset_or_mode
        # Train model for `epochs` using Adam and CrossEntropyLoss...
        # Save trained weights to os.path.join(save_dir, f"{model_name}_model.pth")
```

### 2.4 Testing Method `test(...)`
The evaluator uses `test` for both clean evaluation (`test_without_noise`) and corrupted robustness runs (`test_with_noise`):

```python
    def test(self, test_dataset_or_loader, batch_size=64, device=None):
        """
        Runs inference on the provided test dataset or DataLoader.
        
        Returns:
            tuple: (all_preds, all_labels)
                all_preds: 1D list or numpy array of predicted class integers
                all_labels: 1D list or numpy array of true ground-truth class integers
        """
```

### 2.5 Weight Loading Helper `load_weights(...)` (Recommended)
Allows `eval.py` to optionally skip retraining if saved weights already exist in `output/<model_name>/`:

```python
    def load_weights(self, weights_path=None):
        """Loads state_dict from disk if present. Returns True if loaded, False otherwise."""
```

---

## 3. How `eval.py` Interacts with the Model

When you call `Evaluator.eval_model(model_cls, model_name)`:

1. **Instantiation**: `model_instance = model_cls(num_classes=43)`
2. **Dataset Loading**: Loads GTSRB Train and Test sets scaled to $49 \times 49$.
3. **Training**: Calls `model_instance.train(train_dataset, ...)` (or loads existing checkpoint).
4. **Clean Evaluation**: Calls `Evaluator.test_without_noise(model_instance, test_dataset, model_name)`, which executes `model_instance.test(...)` and writes `output/<model_name>/test_without_noise.pdf`.
5. **Corruption Benchmark**: Calls `Evaluator.test_with_noise(model_instance, test_dataset, model_name)`, which applies all 14 corruptions at 5 severities via `noise.py` with parallelized workers and writes:
   - `output/<model_name>/noise/<noise_type>_severity_5.pdf` (14 confusion matrices)
   - `output/<model_name>/robustness_degradation.pdf`
   - `output/<model_name>/robustness_results.csv`

---

## 4. Boilerplate Template for Adding a New Model

To add a new model (e.g. `resnet18` or `mobilenet`):

```python
# models/<model_name>/model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class MyModel(nn.Module):
    def __init__(self, num_classes=43):
        super(MyModel, self).__init__()
        self.num_classes = num_classes
        # Define architecture for (B, 3, 49, 49) inputs
        ...

    def forward(self, x):
        return ...

    def train(self, dataset_or_mode=True, epochs=5, batch_size=64, lr=0.001, device=None, save_dir=None, **kwargs):
        if isinstance(dataset_or_mode, bool) and not kwargs and epochs == 5 and batch_size == 64:
            return super().train(mode=dataset_or_mode)
        
        # Training loop implementation
        ...

    def test(self, test_dataset_or_loader, batch_size=64, device=None):
        # Inference implementation returning (all_preds, all_labels)
        ...

    def load_weights(self, weights_path=None):
        # Checkpoint loading
        ...
```

Then in `eval.py`:
```python
if __name__ == "__main__":
    from models.my_model.model import MyModel
    evaluator = Evaluator()
    evaluator.eval_model(MyModel, model_name="my_model")
```
