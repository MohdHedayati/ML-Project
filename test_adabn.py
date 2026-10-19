"""
Interactive & Fast Test-Time Adaptation (AdaBN) Evaluation Script.

Demonstrates the immediate benefit of AdaBN by running side-by-side
comparisons between Standard Inference (no adaptation) and AdaBN Adaptation.

Usage:
    # Test a specific corruption (e.g. fog severity 5):
    python test_adabn.py --noise fog --severity 5

    # Test Gaussian noise:
    python test_adabn.py --noise gaussian_noise --severity 3

    # Test all 14 noises at severity 5:
    python test_adabn.py --all
"""

import argparse
import copy
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from dataset_loader import get_test_dataset
from models.baseline.model import Baseline
from noise import NOISE_TYPES, CorruptedGTSRBDataset


def evaluate_corruption(model, test_dataset, noise_type, severity, device, batch_size=128):
    """
    Evaluates a specific corruption condition:
    1. Standard test inference (without AdaBN)
    2. AdaBN test-time adaptation + inference
    """
    is_cuda = (device.type == "cuda")
    corrupted_ds = CorruptedGTSRBDataset(
        base_dataset_or_csv=test_dataset,
        root_dir='data',
        noise_type=noise_type,
        severity=severity,
        target_size=(49, 49)
    )
    loader = DataLoader(corrupted_ds, batch_size=batch_size, shuffle=False, pin_memory=is_cuda)

    # 1. Standard evaluation (without adaptation)
    source_state = copy.deepcopy(model.state_dict())
    preds_standard, labels = model.test(loader, device=device)
    acc_standard = accuracy_score(labels, preds_standard) * 100
    f1_standard = f1_score(labels, preds_standard, average='macro')

    # 2. AdaBN evaluation: adapt running mean and variance to target distribution
    model.load_state_dict(copy.deepcopy(source_state))
    model.adapt_bn(loader, device=device, verbose=False)
    preds_adabn, _ = model.test(loader, device=device)
    acc_adabn = accuracy_score(labels, preds_adabn) * 100
    f1_adabn = f1_score(labels, preds_adabn, average='macro')

    # Restore clean source model weights
    model.load_state_dict(source_state)

    gain = acc_adabn - acc_standard
    return {
        'noise': noise_type,
        'severity': severity,
        'acc_standard': acc_standard,
        'f1_standard': f1_standard,
        'acc_adabn': acc_adabn,
        'f1_adabn': f1_adabn,
        'gain': gain
    }


def main():
    parser = argparse.ArgumentParser(description="Test-Time Adaptation (AdaBN) Evaluation")
    parser.add_argument('--noise', type=str, default='fog', choices=NOISE_TYPES,
                        help="Corruption type (default: fog)")
    parser.add_argument('--severity', type=int, default=5, choices=[1, 2, 3, 4, 5],
                        help="Severity level 1-5 (default: 5)")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument('--all', action='store_true', help="Run comparison across all 14 noises at the specified severity")
    parser.add_argument('--weights', type=str, default='output/baseline/baseline_model.pth', help="Path to trained weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Test-Time Adaptation (AdaBN) Evaluation")
    print(f" Device: {device} | Weights: {args.weights}")
    print(f"=======================================================")

    # 1. Load trained model
    model = Baseline(num_classes=43)
    if not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found. Please run training first (`python train.py`).")
        return
    model.load_weights(args.weights)
    model.to(device)

    # 2. Load test set
    print("Loading clean GTSRB test set...")
    test_dataset = get_test_dataset(target_size=(49, 49))
    print(f"Total test samples: {len(test_dataset)}")

    # 3. Clean baseline accuracy
    clean_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    clean_preds, clean_labels = model.test(clean_loader, device=device)
    clean_acc = accuracy_score(clean_labels, clean_preds) * 100
    print(f"Clean Test Accuracy (No Noise): {clean_acc:.2f}%\n")

    # 4. Run target corruptions
    if args.all:
        noises = NOISE_TYPES
    else:
        noises = [args.noise]

    print(f"{'Corruption':<20} | {'Severity':<8} | {'Standard Acc':<14} | {'AdaBN Acc':<12} | {'Gain':<10}")
    print("-" * 75)

    total_std = 0
    total_ada = 0
    for n in noises:
        res = evaluate_corruption(model, test_dataset, n, args.severity, device, batch_size=args.batch_size)
        total_std += res['acc_standard']
        total_ada += res['acc_adabn']
        sign = "+" if res['gain'] >= 0 else ""
        print(f"{res['noise']:<20} | {res['severity']:<8} | {res['acc_standard']:>6.2f}%        | {res['acc_adabn']:>6.2f}%     | {sign}{res['gain']:>5.2f}%")

    if len(noises) > 1:
        avg_std = total_std / len(noises)
        avg_ada = total_ada / len(noises)
        avg_gain = avg_ada - avg_std
        sign = "+" if avg_gain >= 0 else ""
        print("-" * 75)
        print(f"{'AVERAGE':<20} | {args.severity:<8} | {avg_std:>6.2f}%        | {avg_ada:>6.2f}%     | {sign}{avg_gain:>5.2f}%")


if __name__ == "__main__":
    main()
