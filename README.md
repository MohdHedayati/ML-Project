# Traffic Sign Classification and Corruption Robustness

This project trains convolutional neural networks (CNNs) to classify traffic-sign images, then measures how their predictions change under synthetic weather, blur, sensor noise, and digital distortion. It compares a standard CNN with a BatchNorm-based model that supports Adaptive Batch Normalization (AdaBN). A third model implements an experimental Ant Colony Optimization (ACO) channel-weighting procedure.

**Current implementation:** GTSRB is the default dataset. BTSD and CTSD are supported as separate benchmarks, selected one at a time. There is no active unified 109-class dataset or cross-dataset semantic mapping.

**Implementation update:** All three models now use ReLU. Repeated PIL-cache access and the ACO training-mode crash are fixed. Use `--retrain` for new experiments: old GELU checkpoints have compatible tensor shapes but are not equivalent ReLU models. See [Known issues and interpretation limits](#known-issues-and-interpretation-limits) for remaining limitations.

## Contents

- [Purpose and experiment design](#purpose-and-experiment-design)
- [Overall architecture](#overall-architecture)
- [Datasets and preprocessing](#datasets-and-preprocessing)
- [Model architectures](#model-architectures)
- [Training and optimization](#training-and-optimization)
- [AdaBN adaptation procedure](#adabn-adaptation-procedure)
- [ACO feature-weighting procedure](#aco-feature-weighting-procedure)
- [Corruption benchmark](#corruption-benchmark)
- [Evaluation metrics and output files](#evaluation-metrics-and-output-files)
- [Existing saved results](#existing-saved-results)
- [Installation and usage](#installation-and-usage)
- [Code organization and model interface](#code-organization-and-model-interface)
- [Known issues and interpretation limits](#known-issues-and-interpretation-limits)

## Purpose and experiment design

The task is **image classification**, not object detection. Each input is an image containing a traffic sign; the model predicts one native dataset class ID. There is no implemented road-scene sign detector, bounding-box prediction, video pipeline, or image-restoration network.

For example, a model might correctly classify a clear speed-limit sign but confuse it with another speed-limit class after strong blur. The benchmark measures this degradation and tests whether adjusting BatchNorm statistics helps.

The normal experiment is:

1. Train a model on clean training images, or reuse an existing checkpoint.
2. Evaluate the clean test split.
3. Apply each of 14 corruption types at severity levels 1 through 5.
4. For models that provide adaptation, adapt separately to each corrupted condition without using target labels.
5. Record accuracy, macro F1, confusion matrices, and degradation curves.

One run selects one model and one dataset. It produces **one clean evaluation plus 70 corrupted evaluations**, not 70 separately trained models.

## Overall architecture

```text
cli.py
  Select model, dataset, device, batch size, epochs, and output location
       |
       v
eval.py: Evaluator.eval_model()
       |
       +--> dataset_loader.py
       |      Load selected dataset's training and test splits
       |      Determine output class count: 43, 62, or 58
       |
       +--> models/<model>/model.py
       |      Instantiate model with the selected class count
       |      Load checkpoint if available; otherwise train on clean images
       |
       +--> Clean evaluation
       |      RGB -> resize -> tensor -> model -> predictions -> metrics
       |
       +--> For each corruption and severity:
       |      noise.py wraps the original test images
       |      Apply corruption -> resize -> tensor
       |           |
       |           +--> AdaBN: restore source state, adapt BN statistics
       |           |
       |           v
       |      Model inference -> predictions -> metrics
       |
       v
output/<model>/<dataset>/
  Checkpoint, confusion matrices, results CSV, degradation plot
```

The evaluator is model-agnostic: it calls the selected model's interface and automatically detects adaptation methods. Dataset loading and corruption generation remain outside the network architecture.

## Datasets and preprocessing

### Supported datasets

The following counts describe the local data inspected for this documentation. Counts can change if the underlying files or annotations change.

| Selector | Dataset | Country | Output classes | Training images | Test images |
|---|---|---|---:|---:|---:|
| `gtsrb` | German Traffic Sign Recognition Benchmark | Germany | 43 | 39,209 | 12,630 |
| `btsd` | Belgian Traffic Sign Dataset | Belgium | 62 | 4,591 | 2,534 |
| `ctsd` | Chinese Traffic Sign Database | China | 58 | 4,931 | 1,233 |

- **GTSRB:** Reads `data/Train.csv` and `data/Test.csv`, with image paths and native class IDs `0..42`. The loader also supports a `gtsrb/` subdirectory and fallback path resolution.
- **BTSD:** Reads training/testing class folders under `data/btsd/`, normally `BelgiumTSC_Training/Training/` and `BelgiumTSC_Testing/Testing/`. It uses semicolon-separated class CSVs when available, otherwise numeric folder names. Native labels are `0..61`.
- **CTSD:** Reads `data/ctsd/annotations.csv` and images under `data/ctsd/images/`. It creates a stratified 80/20 train/test split with `random_state=42`. Native labels are `0..57`; this is a code-generated split, not a supplied independent test directory.

The inspected BTSD test split represents 53 of its 62 classes. CTSD's test split represents 57 of its 58 classes. Output dimensions still match the full configured label space.

Labels are not translated between countries. Class ID 0 in one dataset must not be assumed to mean the same thing as class ID 0 in another.

### Clean-image preprocessing

The default transform in [dataset_loader.py](dataset_loader.py) is:

```text
Image file -> RGB -> resize to 49 x 49 -> ToTensor()
```

The network receives tensors with shape `(batch_size, 3, 49, 49)`, with floating-point pixel values in approximately `[0, 1]` for the normal RGB inputs.

There is no additional channel mean/std normalization, aspect-ratio-preserving padding, ROI cropping, random rotation, horizontal flipping, or colour augmentation. Annotation bounding boxes are not used to crop images by these loaders.

Each sample can expose its path, native label, dataset name, and country through `get_metadata()`. Transformed tensors and original PIL images are cached in memory. The original-image cache stores PIL images only, so repeated access consistently returns an image and its label.

## Model architectures

Let `K` be the selected dataset's number of classes. All models return raw class scores, called **logits**, with shape `(batch_size, K)`. The predicted class is the index of the largest score.

These are custom networks trained from scratch, not pretrained ResNet, YOLO, or transformer models.

### Baseline CNN

Implementation: [models/baseline/model.py](models/baseline/model.py).

| Stage | Operation | Output per image |
|---|---|---|
| Input | RGB tensor | `3 x 49 x 49` |
| Block 1 | Conv `3 -> 32`, ReLU, max pooling | `32 x 24 x 24` |
| Block 2 | Conv `32 -> 64`, ReLU, Dropout2d `p=0.2`, max pooling | `64 x 12 x 12` |
| Block 3 | Conv `64 -> 128`, ReLU, max pooling | `128 x 6 x 6` |
| Flatten | Collapse channels and spatial dimensions | `4608` |
| Hidden classifier | Linear `4608 -> 512`, ReLU, Dropout `p=0.5` | `512` |
| Output classifier | Linear `512 -> K` | `K` logits |

Every convolution uses a `3 x 3` kernel, stride 1, and padding 1. Every max-pooling layer uses a `2 x 2` window and stride 2. Spatial dimensions change as `49 -> 24 -> 12 -> 6`.

- **Convolution:** Learns spatial features used to distinguish classes.
- **ReLU:** Adds a nonlinear activation so the network can learn more than a linear transformation.
- **Max pooling:** Reduces feature-map resolution.
- **Dropout:** Randomly suppresses activations during training to reduce reliance on individual features; disabled during inference.
- **Linear layers:** Convert the extracted features into class scores.

For GTSRB (`K=43`), the baseline has **2,475,115 trainable parameters**.

### AdaBN CNN

Implementation: [models/adabn/model.py](models/adabn/model.py).

The spatial sizes and channel counts match the baseline, but the operations differ:

```text
Conv 3 -> 32   -> BatchNorm2d -> ReLU -> MaxPool
Conv 32 -> 64  -> BatchNorm2d -> ReLU -> Dropout2d(0.2) -> MaxPool
Conv 64 -> 128 -> BatchNorm2d -> ReLU -> MaxPool
Flatten
Linear 4608 -> 512 -> BatchNorm1d -> ReLU -> Dropout(0.5)
Linear 512 -> K
```

BatchNorm normalizes intermediate activations using mean/variance statistics and learned scale/offset parameters. The model has three spatial BatchNorm layers and one BatchNorm layer in its classifier.

For GTSRB, it has **2,476,587 trainable parameters**. Its additional adaptation procedure is explained separately below.

**Comparison caveat:** This is not simply the baseline with adaptation switched on. It introduces BatchNorm; both models now use ReLU. A baseline-versus-AdaBN comparison mixes architectural changes with the effect of adaptation.

### ACO-enhanced CNN

Implementation: [models/aco/model.py](models/aco/model.py).

The ACO model uses the baseline's convolution/classifier architecture and adds a channel gate:

```text
Input -> convolutional features: B x 128 x 6 x 6
      -> multiply by pheromone_gate: 1 x 128 x 1 x 1
      -> flatten -> classifier -> K logits
```

Each of the 128 channel weights is shared across that channel's spatial positions. A weight of 1 leaves a channel unchanged; 0.6 reduces it; 1.4 amplifies it.

The gate starts at ones and is registered as a buffer. It is saved in the checkpoint but is not a trainable parameter updated by Adam. For GTSRB, this model has **2,475,115 trainable parameters**, matching the baseline.

Its forward pass and two-epoch training path, including the ACO search, are covered by the focused self-check. Full-dataset ACO robustness results have not yet been produced.

## Training and optimization

All three models use the following defaults when training is required:

| Setting | Value |
|---|---|
| Optimizer for network parameters | Adam |
| Learning rate | `0.001` |
| Adam betas | `(0.9, 0.999)` |
| Adam epsilon | `1e-8` |
| Weight decay | `0` |
| Loss | `torch.nn.CrossEntropyLoss()` |
| Epochs | `5` |
| Training batch size | `64` |
| Training shuffle | Enabled |
| Hardware | CUDA GPU if available, otherwise CPU |

The Adam settings other than learning rate use the library defaults in the inspected environment. Dependencies are not fully version-pinned.

Each batch follows this sequence:

```text
optimizer.zero_grad()
  -> model(images)
  -> CrossEntropyLoss(logits, labels)
  -> loss.backward()
  -> optimizer.step()
```

Cross-entropy measures classification error; backpropagation computes gradients; Adam updates network parameters. No explicit softmax layer is needed before cross-entropy. Prediction uses the maximum logit.

Training reports loss and training accuracy. It saves the final `state_dict`, not a validation-selected best model. The checkpoint does not contain optimizer state for resuming an interrupted training run.

There is no validation loop, early stopping, learning-rate scheduler, class balancing, mixed precision, or corruption augmentation in the normal training path. An epoch is one pass through the training split; it is not a corruption severity level.

The CLI exposes epoch count and batch size but not learning rate or the ACO search settings. Those remain model-method defaults unless called programmatically.

### Checkpoint reuse

The evaluator searches the dataset-specific output folder first, then older model-level checkpoint locations, including `models/<model>/`. An explicit `--weights` path can also be supplied.

If weights load successfully, training is skipped. Consequently, `--epochs 20` alone does not force training. Add `--retrain` to bypass all checkpoint loading and train from scratch; it cannot be combined with `--weights`. A fresh output folder alone may still reuse a legacy checkpoint. All three models reject checkpoints with a different output class count before loading their weights. Activation functions are not recorded in legacy state dictionaries, so use `--retrain` rather than reusing old GELU weights.

## AdaBN adaptation procedure

AdaBN treats corruption as a change in the input distribution. For example, fog changes brightness/contrast patterns and therefore the distribution of internal network activations. Re-estimating BatchNorm statistics can help align the network with this target distribution.

For each corruption/severity pair, the evaluator:

1. Restores the original source model state, including its BatchNorm buffers.
2. Calls `adapt_bn()` with the corrupted test loader.
3. Resets BatchNorm running statistics and sets their momentum to `None` for cumulative averaging over batches.
4. Enables training behaviour for BatchNorm layers while keeping dropout and other layers in evaluation mode.
5. Performs a no-gradient forward pass over the target images, ignoring their labels.
6. Returns the model to evaluation mode and predicts over the target condition.

No optimizer step is performed during adaptation. Convolution weights, classifier weights, and BatchNorm scale/offset parameters are not updated; running-statistic buffers are updated.

Clean evaluation does not run this adaptation pass. Corrupted conditions are adapted independently, and the source state is restored after the benchmark.

This is **batch/transductive test-time adaptation**: the target test distribution is seen before its predictions are scored. It is not a demonstrated one-image-at-a-time deployment procedure, and it does not remove corruption from the input image. With the shared corruption cache, adaptation and scoring use the same corrupted images.

Background: [Revisiting Batch Normalization for Practical Domain Adaptation](https://arxiv.org/abs/1603.04779).

## ACO feature-weighting procedure

ACO supplements Adam; it does not replace the network's gradient-based optimizer.

The schedule is one warm-up epoch, one ACO search before the second epoch, then continued Adam training with the learned gate. The parent model enters evaluation mode during the search so dropout is disabled, and training mode is restored in a `finally` block, including on search failure. Requesting only one epoch never reaches the ACO phase.

| Search setting | Default |
|---|---:|
| Channels | 128 |
| Ants per iteration | 15 |
| Iterations | 5 |
| Evaporation rate `rho` | 0.15 |
| Pheromone exponent `alpha` | 1.0 |
| Heuristic exponent `beta` | 1.0 |
| Deposit scale `Q` | 1.0 |

The implemented search logic is:

1. Take one batch from the shuffled training loader and extract its features.
2. Compute per-channel variance across batch and spatial dimensions; normalize it to obtain desirability `eta`.
3. Calculate channel inclusion probabilities from pheromones `tau` and desirability:

   ```text
   score_i = tau_i^alpha * eta_i^beta
   probability_i = clip(score_i / (1 + score_i), 0.1, 0.95)
   ```

4. Each ant samples a binary channel mask. If it selects fewer than 25% of channels, the highest-probability half are activated as a fallback.
5. Multiply the features by that mask, run the classifier, and measure accuracy on the same labelled training batch.
6. Evaporate pheromones with `tau <- (1 - rho) * tau`, then deposit `Q * accuracy^2` on channels selected by each ant.
7. Normalize accumulated pheromones into a continuous gate approximately in `[0.5, 1.5]` and install it in the model.

Although the code tracks a best binary mask, the returned gate comes from accumulated pheromones, not directly from that mask. Final channels are reweighted, not removed. The fitness objective uses clean training-batch accuracy, not corruption robustness or held-out validation performance. Higher activation variance is a heuristic, not proof of a noise-invariant feature.

## Corruption benchmark

[noise.py](noise.py) uses the [imagecorruptions library](https://github.com/bethgelab/imagecorruptions) to generate the following conditions:

| Group | Corruptions |
|---|---|
| Sensor/noise-like | `gaussian_noise`, `impulse_noise`, `shot_noise` |
| Weather-like | `snow`, `fog`, `frost` |
| Blur | `motion_blur`, `zoom_blur`, `defocus_blur` |
| Appearance | `brightness`, `contrast` |
| Distortion/encoding | `pixelate`, `elastic_transform`, `jpeg_compression` |

Each type is evaluated at severities 1 through 5. This is a project-selected 14-corruption benchmark, not the full ImageNet-C evaluation protocol. ImageNet is not loaded as training data, and no ImageNet-C reference-normalized error metric is computed.

The corrupted-image path is:

```text
Original RGB image
  -> enlarge any dimension below 32 pixels to satisfy the corruption library
  -> apply selected corruption at selected severity
  -> resize to 49 x 49 with bilinear interpolation
  -> convert to tensor
  -> model
```

Corruption is applied before the final resize, rather than to an already downscaled tensor. Small source images may be enlarged rather than downscaled.

### Shared corruption cache

Evaluation automatically builds one memory-mapped uint8 RGB `.npy` file per corruption/severity under `output/.corruption_cache/`. The first evaluation creates each condition using the configured workers; subsequent models, processes and AdaBN adaptation/scoring passes reuse the saved pixels. No command changes or retraining are required. Training itself is unchanged.

At 49 x 49, all 70 conditions use approximately 6.37 GB for GTSRB, 1.28 GB for BTSD, and 0.62 GB for CTSD (decimal GB, excluding small headers). Conditions are read using memory maps rather than loading the complete benchmark into RAM. Cache generation adds disk writes on the first run; speedup depends on how much time corruption generation previously consumed.

The cache key includes ordered source paths, labels, file sizes/modification times, resolution, corruption implementation and relevant library versions. Changes create a separate cache namespace. Source images changed while preserving both size and modification time are not detected; delete the cache to force regeneration in that case. Custom datasets without `paths` and `labels` retain uncached behaviour. Cache files are excluded from Git.

Completed conditions survive interruption. A condition being generated is published only after all its images are written; interruption retries that condition. A forced process kill can leave an unused `.tmp` file. Missing or structurally invalid cache arrays are rebuilt. The results CSV still requires the complete benchmark; image caching is not metrics-resume support. Avoid concurrent first-time builders for the same cache, which can duplicate generation work.

The cache fixes one random corruption realization for reuse. For a fair comparison, evaluate every model against the same cache; old uncached scores are not an identical-image comparison. Stop evaluations before manually deleting cache files. Old cache namespaces are not automatically deleted.

Validate caching without a full benchmark:

```powershell
python scripts/check_corruption_cache.py
```

The wrapper is named `CorruptedGTSRBDataset` for compatibility but supports all three dataset loaders. `CorruptedDataset` is an alias.

## Evaluation metrics and output files

### Metrics

- **Accuracy:** Correct predictions divided by total images, reported as a percentage.
- **Macro F1:** F1 calculated per class and averaged, so frequent classes do not dominate the average merely through their sample counts. Values are on a `0..1` scale.
- **Confusion matrix:** Rows represent true classes; columns represent predicted classes. Off-diagonal entries show mistakes between classes.

The code supplies all configured classes when plotting confusion matrices, but does not supply an explicit full class list to macro F1. Its macro F1 therefore averages over labels present in the truth or predictions, which matters when test splits omit some classes.

### Artifacts

Default layout for a newly trained model:

```text
output/
  <model>/
    <dataset>/
      <model>_model.pth
      model_metadata.json                  (AdaBN training only)
      test_without_noise_<dataset>.pdf
      robustness_results_<dataset>.csv
      robustness_degradation_<dataset>.pdf
      noise/
        <corruption>_severity_5.pdf         (14 files)
```

The current CSV writer records `Dataset`, `Noise_Type`, `Severity`, `Accuracy`, and `Macro_F1`. Older committed CSVs omit the `Dataset` column. Clean metrics are printed and returned to the caller; they are not written as a separate clean-metrics CSV by the current implementation.

Corrupted evaluation uses twice the configured training batch size, so the default is 128. The default four worker processes apply to corrupted loading; clean inference and the default training loaders do not receive that worker setting. CUDA paths use pinned memory in selected loaders and nonblocking transfers where implemented.

Dataset subfolders separate normal outputs between datasets. They do not provide experiment versioning: repeating the same model/dataset run overwrites its metrics and figures, and retraining can overwrite its checkpoint. A custom `--output-dir` is used directly, without automatically appending model/dataset subfolders.

## Existing saved results

The inspected repository contains compatible 43-class checkpoints and GTSRB result CSVs for baseline and AdaBN. No saved ACO benchmark was found.

These numbers summarize **existing saved CSVs**, not a fresh rerun or verified reproduction with the current code. In particular, they predate the GELU-to-ReLU change and must not be presented as new ReLU results:

| Saved metric | Baseline | AdaBN |
|---|---:|---:|
| Mean accuracy over 70 corrupted conditions | 70.97% | 78.86% |
| Mean macro F1 over 70 conditions | 0.6710 | 0.7665 |
| Mean accuracy over the 14 severity-5 conditions | 54.78% | 63.47% |

AdaBN has higher accuracy in 60 of the 70 saved conditions and lower accuracy in 10. Selected severity-5 results:

| Corruption | Baseline accuracy | AdaBN accuracy |
|---|---:|---:|
| Fog | 30.49% | 46.22% |
| Contrast | 17.59% | 47.23% |
| Defocus blur | 26.47% | 22.75% |
| Elastic transform | 53.02% | 45.49% |

Sources: [baseline CSV](output/baseline/gtsrb/robustness_results_gtsrb.csv) and [AdaBN CSV](output/adabn/gtsrb/robustness_results_gtsrb.csv).

The mean accuracy difference is about 7.89 percentage points. It is not evidence that adaptation alone caused that improvement. A controlled adaptation experiment must evaluate the **same trained BatchNorm model with adaptation disabled and enabled**, then repeat across seeds. AdaBN does not improve every corruption condition.

## Installation and usage

Install dependencies in a Python environment:

```powershell
python -m pip install -r requirements.txt
```

Provide the dataset files under `data/` using the layouts described above. The application has no automatic dataset download step. GPU execution requires a compatible CUDA-enabled PyTorch installation; otherwise it selects CPU.

Run these commands from the project root. Each command trains from scratch, then runs clean evaluation and all 70 corruption conditions. Separate `output/relu/` paths preserve the older committed results; repeating these commands overwrites the corresponding new run:

```powershell
python cli.py --model baseline --dataset gtsrb --retrain --epochs 5 --output-dir output/relu/baseline/gtsrb
python cli.py --model adabn --dataset gtsrb --retrain --epochs 5 --output-dir output/relu/adabn/gtsrb
python cli.py --model aco --dataset gtsrb --retrain --epochs 5 --output-dir output/relu/aco/gtsrb

python cli.py --model baseline --dataset btsd --retrain --epochs 5 --output-dir output/relu/baseline/btsd
python cli.py --model adabn --dataset btsd --retrain --epochs 5 --output-dir output/relu/adabn/btsd
python cli.py --model aco --dataset btsd --retrain --epochs 5 --output-dir output/relu/aco/btsd

python cli.py --model baseline --dataset ctsd --retrain --epochs 5 --output-dir output/relu/baseline/ctsd
python cli.py --model adabn --dataset ctsd --retrain --epochs 5 --output-dir output/relu/adabn/ctsd
python cli.py --model aco --dataset ctsd --retrain --epochs 5 --output-dir output/relu/aco/ctsd
```

Run `python cli.py` for interactive model selection, or `python cli.py --help` for arguments. `python eval.py` delegates to the same CLI.

| Argument | Default | Purpose |
|---|---|---|
| `--model` | Interactive selection; baseline preferred | Select discovered model |
| `--dataset` | `gtsrb` | Choose `gtsrb`, `btsd`, or `ctsd` |
| `--data-dir` | `data` | Dataset root |
| `--target-size` | `49` | Square preprocessing size; see fixed-head limitation below |
| `--batch-size` | `64` | Training/clean batch size; doubled for corrupted evaluation |
| `--workers` | `4` | Corrupted-loader worker count |
| `--epochs` | `5` | Epochs only when training is required |
| `--device` | Automatic CUDA/CPU | Explicit computation device |
| `--output-dir` | `output/<model>/<dataset>` | Direct artifact destination |
| `--weights` | Automatic checkpoint search | Explicit checkpoint path |
| `--retrain` | Disabled | Skip checkpoint loading and train from scratch |

All three model names are supported: `baseline`, `adabn`, and `aco`. ACO requires at least two epochs to execute its search. `--workers 0` is supported for in-process corrupted loading after the PIL-cache fix.

Run the focused regression self-check without training on the full datasets:

```powershell
python scripts/check_training.py
```

It checks repeated image/corruption access, all three models with 43/62/58-class heads, two-epoch training, ACO evaluation/training modes and failure recovery, checkpoint round trips and class mismatch rejection, AdaBN adaptation/metadata, and CLI retraining behaviour.

## Code organization and model interface

| File | Responsibility |
|---|---|
| [cli.py](cli.py) | Parse arguments, discover model folders, dynamically import a model class, invoke evaluator |
| [dataset_loader.py](dataset_loader.py) | Load native labels and splits, preprocess/cache images, expose sample metadata |
| [noise.py](noise.py) | Validate corruption choices, transform images, wrap test datasets |
| [eval.py](eval.py) | Choose hardware, load datasets/checkpoints, invoke training/adaptation, score and export results |
| [models/baseline/model.py](models/baseline/model.py) | Standard CNN and its training/inference helpers |
| [models/adabn/model.py](models/adabn/model.py) | BatchNorm CNN and `adapt_bn()` |
| [models/aco/model.py](models/aco/model.py) | Channel-gated CNN and ant-colony search |
| [requirements.txt](requirements.txt) | Runtime dependencies |

The common model contract is:

- Constructor accepts `num_classes` and inherits from `torch.nn.Module`.
- `forward(images)` returns raw logits.
- `train(True/False)` toggles PyTorch mode; `train(dataset, ...)` runs the custom training loop.
- `test(dataset_or_loader, ...)` returns prediction and ground-truth lists.
- `load_weights(path)` loads a checkpoint and reports success.
- Optional `adapt_bn()` or `adapt()` is detected automatically for corrupted evaluation.

Inference uses `torch.inference_mode()`. AdaBN adaptation uses `torch.no_grad()` because it needs running-statistic updates without gradient-based learning.

Dependencies are PyTorch/torchvision for networks and transforms, Pillow for images, NumPy/pandas for data, scikit-learn for splitting/metrics, Matplotlib for figures, imagecorruptions for distortions, and tqdm for progress displays. The code includes NumPy alias compatibility shims for imagecorruptions; these do not constitute complete dependency version locking.

No Colab notebook is tracked in the inspected checkout. A previously uploaded Drive notebook is a separate artifact and is not automatically synchronized with this repository.

## Known issues and interpretation limits

### Runtime and data handling

The ACO method-versus-module crash and nested PIL-cache tuple bug are fixed and covered by `scripts/check_training.py`. Remaining limitations:

1. **Input failures are not always explicit.** Missing dataset files can result in empty datasets. Unknown selectors passed directly to the loader silently fall back to GTSRB; the CLI restricts selectors. Empty data, malformed labels, and invalid training settings are not comprehensively validated.
2. **Image resolution is constrained.** Although the CLI exposes `--target-size`, all classifier heads require 4,608 flattened features. Sizes that do not produce `128 x 6 x 6` features after pooling cause a matrix-shape error. Keep the intended `49 x 49` input unless the architecture is changed.
3. **BatchNorm needs adequate batch sizes.** A singleton training/adaptation batch can fail at the classifier's BatchNorm1d layer. The implementation does not explicitly handle that case for arbitrary custom datasets or batch settings.

### Research validity and reproducibility

1. **Potential CTSD split leakage:** 348 filename families occur in both splits after removing the `_j` suffix. This suggests related original/variant images can cross the split boundary; it is not a verified count of pixel-identical duplicates. Group related variants before splitting and verify their provenance before claiming independent generalization.
2. **No controlled adaptation ablation:** Baseline and AdaBN differ architecturally. Compare the same BatchNorm checkpoint with/without adaptation to isolate the technique's contribution.
3. **No full experiment seeding:** CTSD splitting is seeded, but model initialization, training shuffle, ACO sampling, and initial corruption draws are not centrally seeded. Models sharing a cache reuse identical corrupted images; rebuilding the cache can change those images.
4. **Historical artifact provenance is incomplete:** Existing CSVs/checkpoints do not establish the exact source revision, hyperparameters, random seeds, or adaptation settings that produced them. Their metrics must be identified as saved results, not newly reproduced findings.
5. **ACO robustness remains unproven:** Its implemented objective is clean single-batch accuracy, and there is no full-dataset ACO benchmark in this checkout. Passing a small training regression check does not establish robustness.
6. **Metadata provenance:** New AdaBN metadata uses the training dataset's name (or `unknown` for custom datasets without a name), rather than claiming a unified dataset. Previously saved metadata is not rewritten.
7. **No deployment validation:** Synthetic corruptions do not establish real-road safety, detection performance, latency, or reliability on unseen cameras. No full training or 70-condition benchmark was rerun for this documentation update.
