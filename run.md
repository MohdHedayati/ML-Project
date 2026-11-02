# Running the traffic-sign models

Run commands from `D:\python\ML-Project` in **Windows PowerShell**. The three models are `baseline`, `adabn`, and `aco`; datasets are `gtsrb`, `btsd`, and `ctsd`.

The distinction between operations matters:

| Operation | What runs | Changes weights? |
|---|---|---|
| Train only | Learn from the clean training split and save a checkpoint | Yes |
| Test only | Load a checkpoint and score the clean test split | No |
| Corruption evaluation only | Load a checkpoint and score 14 corruptions at 5 severities | No learned-weight updates; AdaBN temporarily changes BN statistics |
| Complete pipeline | Load existing weights or train, then clean testing and corruption evaluation | Only if training is needed or explicitly requested |

`cli.py` has **no** `--train-only`, `--test-only`, or `--eval-only` flags. Use the standalone Python examples below for those operations. Do not run `python models/aco/model.py` to train: its main block only checks a dummy forward pass.

## 1. Python and dependencies

### Version snapshot

The following versions were inspected in the active Python installation when this guide was written. They are a local version snapshot, **not a fresh-environment reproduction guarantee or a fully transitive lock file**. Training dependencies in `requirements.txt` are mostly unpinned; installing that file alone will not reproduce this snapshot.

| Component / package | Version | Purpose |
|---|---|---|
| Python, 64-bit | 3.11; inspected interpreter: 3.11.0 | Runtime; use an up-to-date Python 3.11 patch release for a new environment |
| setuptools | 69.5.1 | Matches the repository's `setuptools<70` requirement |
| torch | 2.5.1; local build `2.5.1+cu121` | CNN training and inference |
| torchvision | 0.20.1; local build `0.20.1+cu121` | Image transforms; paired with torch 2.5.1 |
| numpy | 1.26.4 | Array operations and legacy corruption compatibility |
| pandas | 2.3.3 | Annotation/result CSVs |
| Pillow | 12.3.0 | Image reading and resizing |
| matplotlib | 3.11.2 | Confusion matrices and degradation plots |
| scikit-learn | 1.9.0 | Metrics and dataset splitting |
| imagecorruptions | 1.1.2 | Synthetic corruption benchmark |
| scipy | 1.17.1 | Corruption and scientific-image operations |
| scikit-image | 0.26.0 | Image-corruption operations |
| tqdm | 4.70.0 | Progress bars |
| opencv-python | 4.11.0.86, proposed clean-environment pin | Dependency required by imagecorruptions |

**Existing-environment warning:** the inspected installation has `opencv-python-headless==4.11.0.86` and `opencv-contrib-python==4.10.0.84`, but no distribution named `opencv-python`. Although `cv2` imports, `pip check` reports imagecorruptions' missing dependency. There is also an unrelated `datasets`/`fsspec` mismatch. Do not call this existing installation dependency-clean. Use a separate environment rather than changing packages used by other projects. Install only one OpenCV wheel family in the new environment.

### Create an isolated environment

```powershell
cd D:\python\ML-Project
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "setuptools==69.5.1"
```

Check `python --version` before creating the environment: these commands assume Python 3.11. If another Python version is selected, use `py -3.11 -m venv .venv` instead. Do not recreate `.venv` if you already use it for something else.

If PowerShell blocks activation, do not change the machine-wide execution policy. Invoke `.\.venv\Scripts\python.exe` in place of `python` in all subsequent commands. Check the selected interpreter with:

```powershell
python -c "import sys; print(sys.executable)"
```

### Install PyTorch: choose one option

For a supported NVIDIA GPU and compatible driver, use CUDA 12.1 wheels:

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only execution, use this **instead**, not in addition:

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

The pairing and wheel commands follow the [official PyTorch 2.5.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v251). `torchaudio` is not used by this project. Other GPUs or newer GPU architectures may need a different supported PyTorch build; do not assume this CUDA build supports every device.

### Install the remaining runtime packages

```powershell
python -m pip install `
    "numpy==1.26.4" "pandas==2.3.3" "Pillow==12.3.0" `
    "matplotlib==3.11.2" "scikit-learn==1.9.0" `
    "scipy==1.17.1" "scikit-image==0.26.0" `
    "opencv-python==4.11.0.86" "imagecorruptions==1.1.2" "tqdm==4.70.0"
python -m pip check
python -c "import cli, noise; import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Run `pip check` before a long experiment. A CUDA-enabled wheel does not guarantee a working GPU: the driver and hardware must also be compatible. Keep NumPy at the documented version for this environment; do not independently upgrade it while keeping the legacy corruption stack unchanged.

The existing Word/PDF reports are retained as static deliverables. Their document-generation helper scripts have been removed during cleanup. Training and evaluation do not require `python-docx`, `reportlab`, or `PyMuPDF`.

## 2. Dataset layout

The application does not download datasets. Extract them before running:

```text
data/
  Train.csv                          GTSRB annotations
  Test.csv
  Train/...                          Image paths referenced by GTSRB CSVs
  Test/...
  btsd/
    BelgiumTSC_Training/Training/...
    BelgiumTSC_Testing/Testing/...
  ctsd/
    annotations.csv
    images/...
```

GTSRB has 43 outputs, BTSD 62, and CTSD 58. CTSD uses a stratified 80/20 split with seed 42. Datasets are separate; their class IDs are not interchangeable. Keep the input size at **49 x 49** because the classifier has a fixed 4608-feature input.

## 3. Complete pipeline: individual model commands

### Train from scratch, then clean testing and all 70 corruption conditions

**Warning:** `--retrain` replaces the selected model/dataset checkpoint when training finishes. Result CSVs and PDFs for the same model/dataset are overwritten too. Back up existing results first if you need to retain them.

```powershell
python cli.py --model baseline --dataset gtsrb --retrain --epochs 5
python cli.py --model adabn --dataset gtsrb --retrain --epochs 5
python cli.py --model aco --dataset gtsrb --retrain --epochs 5

python cli.py --model baseline --dataset btsd --retrain --epochs 5
python cli.py --model adabn --dataset btsd --retrain --epochs 5
python cli.py --model aco --dataset btsd --retrain --epochs 5

python cli.py --model baseline --dataset ctsd --retrain --epochs 5
python cli.py --model adabn --dataset ctsd --retrain --epochs 5
python cli.py --model aco --dataset ctsd --retrain --epochs 5
```

Run only the line you need. ACO needs **at least two epochs** to execute its search, which starts before epoch 2. All three models use Adam and cross-entropy loss; the default learning rate is 0.001. No separate custom result folder is needed.

### Reuse a checkpoint, then run clean testing and corruption evaluation

```powershell
python cli.py --model baseline --dataset gtsrb
python cli.py --model adabn --dataset gtsrb
python cli.py --model aco --dataset gtsrb
```

Change `gtsrb` to `btsd` or `ctsd` as needed. `--epochs` does not retrain a successfully loaded checkpoint. However, **the CLI falls back to training when no compatible checkpoint loads**, even if you supplied `--weights`. For a strict no-training workflow, use sections 6 or 7 instead.

An explicit checkpoint and device can be selected like this:

```powershell
python cli.py --model aco --dataset gtsrb --weights output/aco/gtsrb/aco_model.pth --device cuda --batch-size 64 --workers 4
python cli.py --help
```

Use `--device cpu` if needed, and `--workers 0` for worker-process problems. The CLI doubles `--batch-size` for corruption testing, so `64` becomes `128`. Do not combine `--weights` with `--retrain`. `python eval.py` delegates to the same CLI; it is not evaluation-only.

## 4. Select a model and dataset for standalone Python workflows

Set these once in the current PowerShell session, then run **one** of sections 5-7. Each example is independent and imports everything it needs.

```powershell
$env:MODEL = "baseline"
$env:DATASET = "gtsrb"
```

To run AdaBN on BTSD, for example:

```powershell
$env:MODEL = "adabn"
$env:DATASET = "btsd"
```

The `@' ... '@ | python -` syntax sends the enclosed Python program to the selected interpreter. These snippets deliberately use zero worker processes so they work through standard input on Windows. When putting the code into a `.py` file and enabling worker processes, protect execution with `if __name__ == '__main__':`.

## 5. Train only: no testing or corruption benchmark

This writes `output/<model>/<dataset>/<model>_model.pth`. It overwrites that checkpoint if one already exists; it does not resume its optimizer or training epochs. Existing result CSVs and plots are not regenerated by training-only: until testing is rerun, they may describe the previous checkpoint.

```powershell
@'
import os
from pathlib import Path
import torch
from cli import load_model_class
from dataset_loader import DATASET_CLASSES, get_train_dataset

model_name = os.environ['MODEL']
dataset_name = os.environ['DATASET']
if model_name not in ('baseline', 'adabn', 'aco') or dataset_name not in DATASET_CLASSES:
    raise ValueError('Choose a supported MODEL and DATASET.')
training_data = get_train_dataset(dataset=dataset_name, root_dir='data', target_size=(49, 49))
if not len(training_data):
    raise ValueError('Training dataset is empty; check the dataset files.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model_class(model_name)(num_classes=training_data.num_classes)
output_dir = Path('output') / model_name / dataset_name
training_metrics = model.train(
    training_data, epochs=5, batch_size=64, lr=0.001,
    device=device, save_dir=str(output_dir)
)
print(training_metrics)
'@ | python -
```

`model.train(True)` only switches PyTorch into training mode. Passing the training dataset, as above, invokes this project's actual training loop. For AdaBN, the training batch size must avoid a final batch containing only one sample because its dense layer uses BatchNorm1d.

## 6. Test only: clean accuracy, precision, recall, F1 and confusion matrix

This loads the exact model/dataset checkpoint and **fails instead of training** if it is absent or incompatible. AdaBN is evaluated with its saved source statistics, without target adaptation, matching the standard clean-test protocol.

```powershell
@'
import json
import os
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from cli import load_model_class
from dataset_loader import DATASET_CLASSES, get_test_dataset
from eval import Evaluator

model_name = os.environ['MODEL']
dataset_name = os.environ['DATASET']
if model_name not in ('baseline', 'adabn', 'aco') or dataset_name not in DATASET_CLASSES:
    raise ValueError('Choose a supported MODEL and DATASET.')
test_data = get_test_dataset(dataset=dataset_name, root_dir='data', target_size=(49, 49))
if not len(test_data):
    raise ValueError('Test dataset is empty; check the dataset files.')
output_dir = Path('output') / model_name / dataset_name
checkpoint = output_dir / f'{model_name}_model.pth'
model = load_model_class(model_name)(num_classes=test_data.num_classes)
if not checkpoint.is_file() or not model.load_weights(str(checkpoint)):
    raise RuntimeError(f'Missing or incompatible checkpoint: {checkpoint}')
evaluator = Evaluator(data_dir='data', target_size=(49, 49), num_workers=0)
clean = evaluator.test_without_noise(
    model, test_data, str(output_dir), batch_size=64, dataset=dataset_name
)
precision, recall, macro_f1, support = precision_recall_fscore_support(
    clean['labels'], clean['preds'], average='macro', zero_division=0
)
metrics = {
    'dataset': dataset_name, 'model': model_name, 'condition': 'clean',
    'accuracy_percent': clean['accuracy'],
    'macro_precision_percent': 100 * precision,
    'macro_recall_percent': 100 * recall,
    'macro_f1': macro_f1
}
metrics_path = output_dir / f'clean_metrics_{dataset_name}.json'
metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print(json.dumps(metrics, indent=2))
print('Saved:', metrics_path)
'@ | python -
```

This example adds **clean-test** macro-precision/recall to a JSON file without changing `eval.py`. It also writes the clean confusion matrix PDF. Macro metrics use the classes present in truth or predictions, following sklearn's default class selection. These are not averages across corrupted conditions.

## 7. Corruption evaluation only: no training and no clean testing

This reuses the selected checkpoint, shared corruption cache, and the current evaluator. AdaBN adapts independently to each condition using unlabeled images; baseline and ACO do not adapt.

```powershell
@'
import os
from pathlib import Path
from cli import load_model_class
from dataset_loader import DATASET_CLASSES, get_test_dataset
from eval import Evaluator

model_name = os.environ['MODEL']
dataset_name = os.environ['DATASET']
if model_name not in ('baseline', 'adabn', 'aco') or dataset_name not in DATASET_CLASSES:
    raise ValueError('Choose a supported MODEL and DATASET.')
test_data = get_test_dataset(dataset=dataset_name, root_dir='data', target_size=(49, 49))
if not len(test_data):
    raise ValueError('Test dataset is empty; check the dataset files.')
output_dir = Path('output') / model_name / dataset_name
checkpoint = output_dir / f'{model_name}_model.pth'
model = load_model_class(model_name)(num_classes=test_data.num_classes)
if not checkpoint.is_file() or not model.load_weights(str(checkpoint)):
    raise RuntimeError(f'Missing or incompatible checkpoint: {checkpoint}')
evaluator = Evaluator(data_dir='data', target_size=(49, 49), num_workers=0)
model.to(evaluator.device)
adaptation_hook = None
if hasattr(model, 'adapt_bn'):
    def adaptation_hook(current_model, loader):
        current_model.adapt_bn(loader, device=evaluator.device, verbose=False)
results = evaluator.test_with_noise(
    model, test_data, str(output_dir), batch_size=128,
    num_workers=0, pre_test_hook=adaptation_hook, dataset=dataset_name
)
print(results[['Accuracy', 'Macro_F1']].mean())
'@ | python -
```

This keeps the current CSV format: accuracy (%) and macro-F1 (0-1). It does **not** add corruption macro-precision/recall, which the evaluator does not currently save. Do not mistake the clean metrics from section 6 for corrupted metrics, or infer macro-precision/recall from macro-F1. Micro-precision and micro-recall equal accuracy for this single-label task but do not describe class-balanced performance.

## 8. Compare completed corruption results without rerunning inference

```powershell
@'
from pathlib import Path
import pandas as pd

rows = []
for model_name in ('baseline', 'adabn', 'aco'):
    for dataset_name in ('gtsrb', 'btsd', 'ctsd'):
        path = Path('output') / model_name / dataset_name / f'robustness_results_{dataset_name}.csv'
        if not path.is_file():
            print('Missing:', path)
            continue
        results = pd.read_csv(path)
        if len(results) != 70 or results[['Noise_Type', 'Severity']].drop_duplicates().shape[0] != 70:
            print('Skipping incomplete or duplicated results:', path)
            continue
        rows.append({
            'Model': model_name, 'Dataset': dataset_name,
            'Accuracy (%)': results['Accuracy'].mean(),
            'Macro F1': results['Macro_F1'].mean()
        })
if rows:
    summary = pd.DataFrame(rows)
    print(summary.sort_values(['Dataset', 'Accuracy (%)'], ascending=[True, False]).round(4).to_string(index=False))
    complete = summary.groupby('Model').filter(lambda group: len(group) == 3)
    if not complete.empty:
        print('\nEqual-weight averages across all three datasets:')
        print(complete.groupby('Model')[['Accuracy (%)', 'Macro F1']].mean().round(4).to_string())
else:
    print('No complete benchmark results found.')
'@ | python -
```

Each dataset mean covers 70 conditions; the model average gives each of the three datasets equal weight. This is not a pooled, sample-count-weighted score. The comparison ignores archives and the shared cache.

## 9. Files, caching and interruption behavior

```text
output/<model>/<dataset>/
  <model>_model.pth
  model_metadata.json                    AdaBN training only
  test_without_noise_<dataset>.pdf
  clean_metrics_<dataset>.json            Only the section 6 example writes this
  robustness_results_<dataset>.csv
  robustness_degradation_<dataset>.pdf
  noise/<corruption>_severity_5.pdf        14 matrices

output/.corruption_cache/<key>/
  <corruption>_<severity>.npy
```

- The first benchmark builds corrupted-image caches; later models reuse them. Cached uint8 images for all three datasets require roughly 8.3 GB at 49 x 49. Cache generation and inference are different costs; caching does not eliminate model inference or AdaBN's adaptation pass.
- The cache is shared across models and remains under the repository's `output/.corruption_cache`, even with a custom output directory. Source paths, file metadata, target size, corruption code and selected library versions affect cache keys. Changing those can create another cache namespace; older namespaces are not automatically deleted.
- Avoid launching multiple initial cache builders for the same data simultaneously. The ordinary CLI supports workers for a single run; the standalone stdin examples intentionally use zero workers.
- Checkpoints are saved after the full training loop, not after each epoch. Stopping before training finishes does not save the partially trained model. Existing files are not automatically deleted, but a retraining run can replace the previous checkpoint once it finishes.
- Interrupting testing leaves existing checkpoints and completed cache conditions intact. The corruption CSV and degradation plot are written **after all 70 conditions**; partial in-memory metrics are not resumed. Rerunning starts evaluation from the beginning while reusing completed cache files.
- Repeating a run overwrites its result files. A failed or interrupted rerun may leave an older CSV in place; check timestamps before presenting it as a new result.
- Older GELU checkpoint shapes can match the current ReLU networks, so loading successfully is not proof that such checkpoints are valid current-architecture results. Retrain when intentionally switching architectures; do not retrain already completed current experiments just to view their metrics.

## 10. Lightweight checks

```powershell
python -m pip check
python cli.py --help
python scripts/check_training.py
python scripts/check_corruption_cache.py
```

The last two are synthetic regression checks, not full-dataset training. They cover the three model interfaces, PIL caching, ACO mode handling, AdaBN adaptation, checkpoint compatibility, and shared corruption-cache behavior. They do not establish full benchmark reproducibility or timing.

## Validation status of this guide

CLI signatures, dataset loaders, model interfaces, package versions and PyTorch's documented wheel pairing were inspected. All four standalone Python programs passed syntax checks. Train-only, clean-only, corruption-only, and missing-checkpoint refusal were checked for all nine model/dataset combinations with training and inference mocked inside a temporary directory. The comparison example ran against the real saved CSVs. Real checkpoints and experiment results were not changed. A fresh dependency installation and complete training/70-condition benchmark were not performed for this documentation task. The existing Python installation's `pip check` failures are disclosed above.
