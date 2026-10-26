# Traffic Sign Recognition Robustness & AdaBN Benchmark

A deep learning benchmark evaluating the robustness of Convolutional Neural Networks (CNNs) for traffic sign classification under environmental, digital, and sensor image corruptions across three international benchmarks combined into a single unified dataset with authoritative semantic label mapping:
- **GTSRB** (German Traffic Sign Recognition Benchmark — 43 classes, Germany)
- **BTSD** (Belgian Traffic Sign Dataset — 62 classes, Belgium)
- **CTSD** (Chinese Traffic Sign Database — 58 classes, China)

Total unified semantic classes: **109 classes** (indexed from 0 to 108).

The project includes test-time domain adaptation via **AdaBN (Adaptive Batch Normalization)** to dynamically recover lost accuracy under heavy corruptions without requiring retraining or target labels.

---

## 🏷️ Unified Semantic Label Mapping

Instead of arbitrary label-offset concatenation (`0..42`, `43..104`, `105..162`) which artificially created duplicate classes (such as German Stop vs. Belgian Stop vs. Chinese Stop), this project implements a unified semantic label mapping defined in [unified_label_mapping.json](file:///d:/python/ML-Project/unified_label_mapping.json):

1. **Equivalent Signs Unified**: Signs with the same real-world meaning across datasets share identical unified IDs (e.g. Stop $\to$ ID `0`, Yield $\to$ ID `1`, No Entry $\to$ ID `2`, Roundabout $\to$ ID `4`, Road Work $\to$ ID `8`, Traffic Signals $\to$ ID `9`, etc.).
2. **Distinct & Country-Specific Signs Preserved**: Signs unique to a single region or standard (such as German speed limit cancellation signs, Belgian parking regulation variants, or Chinese inspection signs) maintain separate, unique unified IDs.
3. **Ambiguous Classes Flagged**: Classes where definitions are compound or ambiguous (e.g., Belgian compound speed limit code `C43`) are preserved as distinct classes and flagged with `"needs_manual_review": true`.
4. **Full Provenance & Metadata**: Every sample retains its original image path, unified semantic label, original dataset name, original class ID, and country of origin.

---

## 🚀 Quickstart Guide

### 1. Training & Evaluation Pipeline
Evaluate clean and corrupted performance across 14 noise types at 5 severities (70 benchmark experiments) on the unified 109-class dataset:

```powershell
# Run AdaBN with test-time domain adaptation (saved to output/adabn/)
python eval.py --model adabn

# Run Standard Baseline without test-time adaptation (saved to output/baseline/)
python eval.py --model baseline
```

---

## 📁 Project Structure

```text
ML-Project/
├── __pycache__/
├── .agents/
├── data/                      # Normalized datasets: GTSRB (root), BTSD (btsd/), CTSD (ctsd/)
├── env/                       # Python virtual environment
├── models/
│   ├── adabn/
│   │   └── model.py           # Adaptive CNN architecture (AdaBN test-time adaptation, 109 classes)
│   └── baseline/
│       └── model.py           # Standard Baseline CNN architecture (109 classes)
├── output/                    # Model weights (.pth), metadata, degradation curves, & confusion matrices
│   ├── adabn/                 # AdaBN evaluation & checkpoint outputs
│   └── baseline/              # Baseline evaluation & checkpoint outputs
├── .gitignore
├── dataset_loader.py          # Unified loader combining GTSRB + BTSD + CTSD (109 semantic classes)
├── eval.py                    # Complete evaluation & training pipeline for adabn and baseline
├── noise.py                   # 14 corruptions (weather, blurs, digital noise) via imagecorruptions
├── README.md                  # Project overview and instructions
├── requirements.txt           # Project dependencies
├── unified_label_mapping.json # Authoritative cross-dataset semantic mapping table
└── visualize_corruptions.py   # Corruption sample generator
```