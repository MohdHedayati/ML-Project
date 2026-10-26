"""
Modular Evaluation Pipeline for Traffic Sign Recognition Models.

Evaluates Baseline and AdaBN models on the unified multi-dataset benchmark
(GTSRB + BTSD + CTSD combined into one set with 109 unified semantic classes).

Methods:
- test_without_noise: evaluates clean test data, saves confusion matrix to output/<model_name>/test_without_noise.pdf
- test_with_noise: evaluates across 14 noise types x 5 severities,
  saves 14 confusion matrices to output/<model_name>/noise/<noise>_severity_5.pdf,
  saves robustness_degradation.pdf and robustness_results.csv to output/<model_name>/
- eval_model: instantiates model_cls, trains on combined data, runs clean and robustness tests
"""

import os
import copy
import argparse
import numpy as np
# NumPy 2.0+ compatibility shim for imagecorruptions
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128

import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from dataset_loader import get_train_dataset, get_test_dataset, TOTAL_CLASSES, UNIFIED_CLASS_NAMES
from noise import NOISE_TYPES, CorruptedGTSRBDataset
from models.adabn.model import AdaBN
from models.baseline.model import Baseline


class Evaluator:
    """
    Evaluator class for benchmarking traffic sign recognition models
    under clean and corrupted conditions.
    """
    def __init__(self, data_dir='data', target_size=(49, 49), device=None, num_workers=0):
        self.data_dir = data_dir
        self.target_size = target_size
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.num_workers = num_workers if num_workers is not None else 0
        print(f"[Evaluator] Initialized on device: {self.device} (Target: {self.target_size}, Workers: {self.num_workers})")

    def test_without_noise(self, model_instance, test_dataset, output_dir="output/adabn", batch_size=64):
        """
        Evaluates the model on clean data (without noise) scaled to target_size.
        Generates and saves a confusion matrix to output_dir/test_without_noise.pdf.
        """
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, 'test_without_noise.pdf')

        print(f"\n[Evaluator] Running clean evaluation in '{output_dir}'...")
        all_preds, all_labels = model_instance.test(test_dataset, batch_size=batch_size, device=self.device)

        acc = accuracy_score(all_labels, all_preds) * 100
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"--- CLEAN EVALUATION RESULTS ---")
        print(f"Accuracy: {acc:.2f}% | Macro F1-Score: {macro_f1:.4f}")

        # Generate and save confusion matrix
        fig, ax = plt.subplots(figsize=(14, 14))
        labels_range = list(range(TOTAL_CLASSES))
        cm = confusion_matrix(all_labels, all_preds, labels=labels_range)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', colorbar=False, include_values=False)
        plt.title(f"Confusion Matrix (Clean Data - {TOTAL_CLASSES} Unified Classes - {self.target_size[0]}x{self.target_size[1]})", fontsize=14)
        plt.tight_layout()
        plt.savefig(pdf_path, format="pdf")
        plt.close(fig)
        print(f"[Evaluator] Clean confusion matrix saved to: '{pdf_path}'")

        return {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'preds': all_preds,
            'labels': all_labels,
            'pdf_path': pdf_path
        }

    def test_with_noise(self, model_instance, test_dataset, output_dir="output/adabn", batch_size=128, num_workers=0, use_adabn=True):
        """
        Evaluates the model across 14 noise types at 5 severity levels (70 experiments).
        If use_adabn is True, adapts BatchNorm layers before testing each corruption.
        """
        if num_workers is None:
            num_workers = self.num_workers

        noise_dir = os.path.join(output_dir, 'noise')
        os.makedirs(noise_dir, exist_ok=True)

        is_cuda = (self.device.type == "cuda")
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory': is_cuda
        }
        if num_workers > 0:
            loader_kwargs['prefetch_factor'] = 2

        adabn_str = " [WITH AdaBN TEST-TIME ADAPTATION]" if use_adabn else " [WITHOUT ADAPTATION]"
        print(f"\n[Evaluator] Running robustness benchmark{adabn_str} across {len(NOISE_TYPES)} corruptions x 5 severities...")
        print(f"[Evaluator] Output directory: '{output_dir}'")

        source_state = copy.deepcopy(model_instance.state_dict())

        results = []
        total_experiments = len(NOISE_TYPES) * 5
        progress_bar = tqdm(total=total_experiments, desc="Robustness Benchmark", leave=True)

        for noise_type in NOISE_TYPES:
            for severity in range(1, 6):
                corrupted_ds = CorruptedGTSRBDataset(
                    base_dataset_or_csv=test_dataset,
                    noise_type=noise_type,
                    severity=severity,
                    target_size=self.target_size
                )

                corrupted_loader = DataLoader(corrupted_ds, **loader_kwargs)

                if use_adabn:
                    model_instance.load_state_dict(copy.deepcopy(source_state))
                    if hasattr(model_instance, 'adapt_bn'):
                        model_instance.adapt_bn(corrupted_loader, device=self.device, verbose=False)

                preds, labels = model_instance.test(corrupted_loader, device=self.device)
                acc = accuracy_score(labels, preds) * 100
                macro_f1 = f1_score(labels, preds, average='macro')

                if severity == 5:
                    self._save_severity_confusion_matrix(
                        labels=labels,
                        preds=preds,
                        noise_type=noise_type,
                        severity=5,
                        save_path=os.path.join(noise_dir, f"{noise_type}_severity_5.pdf")
                    )

                results.append({
                    'Noise_Type': noise_type,
                    'Severity': severity,
                    'Accuracy': round(acc, 2),
                    'Macro_F1': round(macro_f1, 4)
                })

                progress_bar.set_postfix(noise=noise_type[:8], sev=severity, acc=f"{acc:.1f}%")
                progress_bar.update(1)

        progress_bar.close()

        # Restore clean state
        model_instance.load_state_dict(source_state)

        results.sort(key=lambda x: (NOISE_TYPES.index(x['Noise_Type']), x['Severity']))
        results_df = pd.DataFrame(results)

        csv_path = os.path.join(output_dir, 'robustness_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"[Evaluator] Robustness results saved to: '{csv_path}'")

        plot_path = os.path.join(output_dir, 'robustness_degradation.pdf')
        self.plot_robustness(results_df, plot_path)

        return results_df

    def _save_severity_confusion_matrix(self, labels, preds, noise_type, severity, save_path):
        """Helper to render and save severity confusion matrix."""
        fig, ax = plt.subplots(figsize=(12, 12))
        labels_range = list(range(TOTAL_CLASSES))
        cm = confusion_matrix(labels, preds, labels=labels_range)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', colorbar=False, include_values=False)
        plt.title(f"Confusion Matrix: {noise_type} (Severity {severity} - {TOTAL_CLASSES} Unified Classes - {self.target_size[0]}x{self.target_size[1]})")
        plt.tight_layout()
        plt.savefig(save_path, format="pdf")
        plt.close(fig)

    def plot_robustness(self, results_data, output_path):
        """Generates and saves degradation curve plot across 14 corruptions."""
        if isinstance(results_data, str):
            df = pd.read_csv(results_data)
        else:
            df = results_data

        plt.figure(figsize=(12, 7))
        for noise_name, group in df.groupby('Noise_Type'):
            plt.plot(
                group['Severity'], 
                group['Accuracy'], 
                marker='o', 
                label=noise_name.replace('_', ' ').title()
            )

        plt.title(f'Model Accuracy vs. Corruption Severity Across 14 Noise Types ({self.target_size[0]}x{self.target_size[1]})', fontsize=14)
        plt.xlabel('Severity Level (1 to 5)', fontsize=12)
        plt.ylabel('Test Accuracy (%)', fontsize=12)
        plt.xticks([1, 2, 3, 4, 5])
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format='pdf')
        plt.close()
        print(f"[Evaluator] Robustness degradation plot saved to: '{output_path}'")

    def eval_model(self, model_cls=AdaBN, output_dir='output/adabn', epochs=5, batch_size=64, num_workers=0, load_if_exists=True, use_adabn=True):
        """
        Main orchestration method:
        1. Loads combined train and test datasets (all 3 datasets in one set, 163 classes).
        2. Instantiates model class (AdaBN or Baseline).
        3. Trains model or loads checkpoint.
        4. Runs test_without_noise.
        5. Runs test_with_noise.
        """
        print(f"\n==========================================")
        print(f" Starting Evaluation Pipeline -> {output_dir}")
        print(f" Datasets: GTSRB + BTSD + CTSD combined ({TOTAL_CLASSES} unified semantic classes)")
        print(f" Mode: {'AdaBN Test-Time Adaptation' if use_adabn else 'Standard Baseline'}")
        print(f"==========================================")

        train_dataset = get_train_dataset(root_dir=self.data_dir, target_size=self.target_size)
        test_dataset = get_test_dataset(root_dir=self.data_dir, target_size=self.target_size)
        print(f"[Evaluator] Loaded Combined Train ({len(train_dataset):,} samples) and Test ({len(test_dataset):,} samples)")

        model_instance = model_cls(num_classes=TOTAL_CLASSES)
        checkpoint_candidates = [
            os.path.join(output_dir, 'adabn_model.pth'),
            os.path.join(output_dir, 'baseline_model.pth'),
            os.path.join(output_dir, 'model.pth'),
            os.path.join('models', 'adabn', 'adabn_model.pth'),
        ]
        checkpoint_path = next((c for c in checkpoint_candidates if os.path.exists(c)), checkpoint_candidates[0])

        weights_loaded = False
        if load_if_exists and hasattr(model_instance, 'load_weights'):
            weights_loaded = model_instance.load_weights(checkpoint_path)

        if not weights_loaded:
            print(f"[Evaluator] Training model instance on combined dataset...")
            model_instance.train(
                dataset_or_mode=train_dataset,
                epochs=epochs,
                batch_size=batch_size,
                device=self.device,
                save_dir=output_dir
            )

        clean_results = self.test_without_noise(
            model_instance=model_instance,
            test_dataset=test_dataset,
            output_dir=output_dir
        )

        robustness_results = self.test_with_noise(
            model_instance=model_instance,
            test_dataset=test_dataset,
            output_dir=output_dir,
            num_workers=num_workers,
            use_adabn=use_adabn
        )

        print(f"\n==========================================")
        print(f" Finished Pipeline. Artifacts stored in: {output_dir}/")
        print(f"==========================================\n")

        return {
            'output_dir': output_dir,
            'clean_metrics': clean_results,
            'robustness_df': robustness_results
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Traffic Sign Recognition Model on Combined Benchmark")
    parser.add_argument('--model', type=str, default='adabn', choices=['adabn', 'baseline'], help="Model architecture: 'adabn' or 'baseline' (default: adabn)")
    parser.add_argument('--output-dir', type=str, default=None, help="Output directory (default: output/<model>)")
    parser.add_argument('--no-adapt', action='store_true', help="Disable test-time adaptation")
    parser.add_argument('--batch-size', type=int, default=128, help="Inference batch size")
    parser.add_argument('--num-workers', type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    model_cls = AdaBN if args.model == 'adabn' else Baseline
    output_dir = args.output_dir if args.output_dir else f"output/{args.model}"

    evaluator = Evaluator()
    evaluator.eval_model(
        model_cls=model_cls,
        output_dir=output_dir,
        use_adabn=(not args.no_adapt) and (args.model == 'adabn'),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        load_if_exists=True
    )
