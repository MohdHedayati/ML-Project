---
name: model-interface
description: >-
  Defines the architecture and interface contract for traffic sign recognition models evaluated by eval.py.
  Use this skill whenever implementing, refactoring, or integrating a new neural network model
  into the project to ensure seamless compatibility with dataset loading, clean evaluation, and robustness benchmarks.
---

# Modular Model Interface Specification

This skill documents the standard architecture interface, lifecycle contract, directory conventions, and output artifact preservation required for models plugging into the modular evaluation pipeline (`eval.py`).

---

## 1. Directory & File Organization

Each model in the project must reside in its own dedicated directory under `models/`:

```text
models/
├── baseline/
│   └── model.py          # Baseline CNN architecture
├── adabn/
│   └── model.py          # AdaBN CNN architecture
├── aco/
│   └── model.py          # Ant Colony Optimization feature gating CNN
└── <new_model_name>/
    └── model.py          # New model class adhering to this interface
```

Import model classes dynamically or directly:
```python
from models.baseline.model import Baseline
```

---

## 2. Model Class Interface Contract

Any model class evaluated by `Evaluator.eval_model(model_cls, model_name)` must conform to the following interface:

### 2.1 Class Signature & Inheritance
The model must inherit from `torch.nn.Module` and accept `num_classes` (defaulting to 43 for GTSRB):

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes=43):
        super(MyModel, self).__init__()
        self.num_classes = num_classes
        # Layers definition...
```

### 2.2 Input & Output Dimensions
* **Input tensor**: `(Batch_Size, 3, 49, 49)` (RGB images scaled down to $49 \times 49$ pixels).
* **Output logits**: `(Batch_Size, num_classes)` (unnormalized raw logits for the target benchmark).

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

### 2.5 Weight Loading Helper `load_weights(...)`
Allows `eval.py` to optionally skip retraining if saved weights already exist in `output/<model_name>/` or `output/<model_name>/<dataset>/`:

```python
    def load_weights(self, weights_path=None):
        """Loads state_dict from disk if present. Returns True if loaded, False otherwise."""
```

---

## 3. Dedicated Dataset Output Structure & Asset Preservation

To ensure that running benchmarks across different datasets (`gtsrb`, `btsd`, `ctsd`) **never overwrites or wipes out another dataset's assets**, all outputs are stored in dedicated dataset folders with tagged filenames:

```text
output/
└── <model_name>/
    ├── <model_name>_model.pth                    # Global model weights
    └── <dataset>/                                # Dedicated dataset folder (e.g. gtsrb, btsd, ctsd)
        ├── noise/                                # Dedicated noise confusion matrices
        │   ├── brightness_severity_5.pdf
        │   ├── contrast_severity_5.pdf
        │   ├── ... (14 corruptions)
        │   └── zoom_blur_severity_5.pdf
        ├── test_without_noise_<dataset>.pdf      # Dataset-tagged clean matrix
        ├── robustness_degradation_<dataset>.pdf  # Dataset-tagged degradation curve plot
        └── robustness_results_<dataset>.csv      # Dataset-tagged numerical benchmark metrics
```

### Supported Datasets:
* **`gtsrb`** (German — 43 classes): Default benchmark.
* **`btsd`** (Belgian — 62 classes): Regional European benchmark.
* **`ctsd`** (Chinese — 58 classes): Asian benchmark.

---

## 4. How `eval.py` and `cli.py` Interact with Models

When evaluating a model:
```bash
python cli.py --model baseline --dataset gtsrb
```

1. **Model Discovery**: `cli.py` dynamically scans `models/` for available models.
2. **Dataset Loading**: `eval.py` loads the requested dataset (`gtsrb` by default) and determines `num_classes`.
3. **Dedicated Output Setup**: Sets output directory to `output/<model_name>/<dataset>/` and creates `noise/`.
4. **Weights Loading**: Resolves weights from `output/<model>/<dataset>/` or `output/<model>/` without unnecessary retraining.
5. **Clean Evaluation**: Writes `test_without_noise_<dataset>.pdf`.
6. **Corruption Benchmark**: Runs 14 corruptions across 5 severities and writes:
   * 14 confusion matrices to `output/<model>/<dataset>/noise/<noise_type>_severity_5.pdf`
   * Degradation curve to `output/<model>/<dataset>/robustness_degradation_<dataset>.pdf`
   * Metrics table to `output/<model>/<dataset>/robustness_results_<dataset>.csv`
