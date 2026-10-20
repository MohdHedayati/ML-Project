# Traffic Sign Recognition Robustness & AdaBN Benchmark

A deep learning benchmark evaluating the robustness of Convolutional Neural Networks (CNNs) for traffic sign classification under environmental, digital, and sensor image corruptions across three international benchmarks:
- **GTSRB** (German Traffic Sign Recognition Benchmark — 43 classes)
- **CTSD** (Chinese Traffic Sign Dataset — 58 classes)
- **BTSD** (Belgian Traffic Sign Dataset — 62 classes)

The project includes test-time domain adaptation via **AdaBN (Adaptive Batch Normalization)** to dynamically recover lost accuracy under heavy corruptions without requiring retraining or target labels.

---

## 📊 Benchmark Results

For full numerical results, severity degradation curves, and detailed per-corruption comparisons, see:
👉 **[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)**

### Highlights:
- **CTSD Clean Test Accuracy:** **99.27%**
- **CTSD Mean Corrupted Accuracy:** **86.00%** (Baseline) ➔ **94.48%** (with AdaBN, **+8.48% boost**)
- **GTSRB Mean Corrupted Accuracy:** **70.97%** (Baseline) ➔ **78.79%** (with AdaBN, **+7.82% boost**)
- **Dense Fog (CTSD Severity 5):** 32.52% ➔ **71.53%** (**+39.01%**)
- **Heavy Frost (CTSD Severity 5):** 56.45% ➔ **93.51%** (**+37.06%**)

---

## 🚀 Quickstart Guide

### 1. Training
Train baseline models on any dataset with native CUDA GPU acceleration:
```powershell
# GTSRB (43 classes)
python train.py --dataset gtsrb --epochs 5

# Belgian Traffic Sign Dataset (62 classes)
python train.py --dataset btsd --epochs 5

# Chinese Traffic Sign Dataset (58 classes)
python train.py --dataset ctsd --epochs 5
```

### 2. Evaluation & Robustness Benchmarking
Evaluate clean and corrupted performance across 14 noise types at 5 severities (70 experiments):
```powershell
# Standard baseline evaluation
python eval.py --dataset gtsrb
python eval.py --dataset btsd
python eval.py --dataset ctsd

# Evaluation with AdaBN test-time adaptation
python eval.py --dataset gtsrb --adabn
python eval.py --dataset btsd --adabn
python eval.py --dataset ctsd --adabn

# Generate comparison plots (Baseline vs. AdaBN)
python eval.py --dataset gtsrb --compare
python eval.py --dataset btsd --compare
python eval.py --dataset ctsd --compare
```

### 3. Checkpoint Inspection
Inspect trained `.pth` weight matrices, layers, parameter counts, and distributions:
```powershell
python inspect_weights.py output/baseline_ctsd/baseline_model.pth
```

---

## 📁 Project Structure

```text
ML-Project/
├── dataset_loader.py          # Unified data loader for GTSRB, BTSD, and CTSD with RAM caching
├── noise.py                   # 14 corruptions (weather, blurs, digital noise) via imagecorruptions
├── train.py                   # Unified GPU training script supporting all 3 datasets
├── eval.py                    # Complete evaluation pipeline (clean test, 70 corruptions, AdaBN, plots)
├── inspect_weights.py         # Checkpoint (.pth) layer inspection utility
├── BENCHMARK_RESULTS.md       # Comprehensive benchmark report with detailed accuracy tables
├── models/
│   └── baseline/
│       └── model.py           # CNN architecture with BatchNorm and AdaBN adaptation
└── output/                    # Model weights (.pth), confusion matrices (.pdf), & CSV tables
    ├── baseline/              # GTSRB baseline outputs
    ├── baseline_adabn/        # GTSRB AdaBN outputs
    ├── baseline_btsd/         # BTSD baseline outputs
    └── baseline_ctsd/         # CTSD baseline outputs
```