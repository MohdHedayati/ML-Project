"""
Modular Evaluation Pipeline for GTSRB Traffic Sign Recognition Models.

Provides the Evaluator class with methods:
- test_without_noise: evaluates clean test data, saves confusion matrix to output/<model_name>/test_without_noise.pdf
- test_with_noise: evaluates across 14 noise types x 5 severities,
  saves 14 confusion matrices to output/<model_name>/noise/<noise>_severity_5.pdf,
  saves robustness_degradation.pdf and robustness_results.csv to output/<model_name>/
- plot_robustness: helper to generate and save degradation curves
- eval_model: instantiates model_cls, trains it on 49x49 data, runs test_without_noise and test_with_noise
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

from dataset_loader import GTSRBDataset, get_train_dataset, get_test_dataset, DATASET_CLASSES
from noise import NOISE_TYPES, CorruptedGTSRBDataset


class Evaluator:
    """
    Evaluator class for benchmarking GTSRB traffic sign recognition models
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

    def test_without_noise(self, model_instance, test_dataset, model_name="baseline", batch_size=64):
        """
        Evaluates the model on clean data (without noise) scaled to target_size.
        Generates and saves a confusion matrix to output/<model_name>/test_without_noise.pdf.
        
        Args:
            model_instance: An initialized model instance implementing test(loader/dataset).
            test_dataset: Test GTSRBDataset.
            model_name: Name of the model directory (default: 'baseline').
            batch_size: Evaluation batch size.
            
        Returns:
            dict: Evaluation metrics including accuracy, macro F1, predictions, and labels.
        """
        output_dir = os.path.join('output', model_name)
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, 'test_without_noise.pdf')

        print(f"\n[Evaluator] Running clean evaluation for model '{model_name}'...")
        all_preds, all_labels = model_instance.test(test_dataset, batch_size=batch_size, device=self.device)

        acc = accuracy_score(all_labels, all_preds) * 100
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"--- {model_name.upper()} CLEAN EVALUATION RESULTS ---")
        print(f"Accuracy: {acc:.2f}% | Macro F1-Score: {macro_f1:.4f}")

        # Generate and save confusion matrix
        fig, ax = plt.subplots(figsize=(12, 12))
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        plt.title(f"{model_name.title()} Confusion Matrix (Clean Data - {self.target_size[0]}x{self.target_size[1]})", fontsize=14)
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

    def test_with_noise(self, model_instance, test_dataset, model_name="baseline", batch_size=128, num_workers=0, use_adabn=False):
        """
        Evaluates the model across 14 noise types at 5 severity levels (70 experiments).
        If use_adabn is True, adapts BatchNorm layers (AdaBN) before testing each corruption.
        
        Saves:
        - 14 confusion matrices for severity 5 to output/<model_name>/noise/<noise_type>_severity_5.pdf
        - Robustness degradation plot to output/<model_name>/robustness_degradation.pdf
        - Benchmark metrics to output/<model_name>/robustness_results.csv
        
        Args:
            model_instance: Model instance implementing test(loader/dataset).
            test_dataset: Base test dataset or CSV path.
            model_name: Name of the model directory (default: 'baseline').
            batch_size: Batch size for corrupted DataLoader.
            num_workers: Number of DataLoader worker processes (default: 0).
            use_adabn: If True, perform test-time adaptation via AdaBN for each test condition.
            
        Returns:
            pd.DataFrame: Table of results across all corruptions and severities.
        """
        if num_workers is None:
            num_workers = self.num_workers

        output_subdir = f"{model_name}_adabn" if use_adabn else model_name
        output_dir = os.path.join('output', output_subdir)
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

        adabn_str = " [WITH AdaBN TEST-TIME ADAPTATION]" if use_adabn else ""
        print(f"\n[Evaluator] Running robustness benchmark{adabn_str} across {len(NOISE_TYPES)} corruptions x 5 severities...")
        print(f"[Evaluator] Output directory: '{output_dir}'")

        # Snapshot clean source state dict so each corruption adapts independently
        if use_adabn:
            source_state = copy.deepcopy(model_instance.state_dict())

        results = []
        total_experiments = len(NOISE_TYPES) * 5
        progress_bar = tqdm(total=total_experiments, desc=f"Robustness Benchmark ({output_subdir})", leave=True)

        for noise_type in NOISE_TYPES:
            for severity in range(1, 6):
                # Dynamic corrupted dataset with noise-first then downscaling
                corrupted_ds = CorruptedGTSRBDataset(
                    base_dataset_or_csv=test_dataset,
                    root_dir=self.data_dir,
                    noise_type=noise_type,
                    severity=severity,
                    target_size=self.target_size
                )

                corrupted_loader = DataLoader(corrupted_ds, **loader_kwargs)

                # AdaBN: reset to source weights and adapt BatchNorm statistics on target samples
                if use_adabn:
                    model_instance.load_state_dict(copy.deepcopy(source_state))
                    if hasattr(model_instance, 'adapt_bn'):
                        model_instance.adapt_bn(corrupted_loader, device=self.device, verbose=False)

                preds, labels = model_instance.test(corrupted_loader, device=self.device)
                acc = accuracy_score(labels, preds) * 100
                macro_f1 = f1_score(labels, preds, average='macro')

                # Save confusion matrix for Severity 5
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

        # Restore original source state if adapted
        if use_adabn:
            model_instance.load_state_dict(source_state)

        # Sort results consistently
        results.sort(key=lambda x: (NOISE_TYPES.index(x['Noise_Type']), x['Severity']))
        results_df = pd.DataFrame(results)

        # Save robustness results CSV
        csv_path = os.path.join(output_dir, 'robustness_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"[Evaluator] Robustness results saved to: '{csv_path}'")

        # Save degradation plot PDF
        plot_path = os.path.join(output_dir, 'robustness_degradation.pdf')
        self.plot_robustness(results_df, plot_path)

        return results_df

    def _save_severity_confusion_matrix(self, labels, preds, noise_type, severity, save_path):
        """Helper to render and save severity confusion matrix."""
        fig, ax = plt.subplots(figsize=(10, 10))
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        plt.title(f"Confusion Matrix: {noise_type} (Severity {severity} - {self.target_size[0]}x{self.target_size[1]})")
        plt.tight_layout()
        plt.savefig(save_path, format="pdf")
        plt.close(fig)

    def plot_robustness(self, results_data, output_path):
        """
        Generates and saves the degradation curve plot across 14 corruptions.
        
        Args:
            results_data: pd.DataFrame or path to CSV file containing robustness results.
            output_path: Path where the degradation PDF will be saved.
        """
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

    def plot_comparison(self, baseline_csv, adabn_csv, output_path):
        """
        Plots a side-by-side comparison curve showing accuracy before and after AdaBN.
        """
        if not (os.path.exists(baseline_csv) and os.path.exists(adabn_csv)):
            print(f"[Evaluator] Cannot plot comparison: '{baseline_csv}' or '{adabn_csv}' missing.")
            return

        base_df = pd.read_csv(baseline_csv)
        adabn_df = pd.read_csv(adabn_csv)

        # Compute average accuracy across all 14 noises per severity
        base_avg = base_df.groupby('Severity')['Accuracy'].mean()
        adabn_avg = adabn_df.groupby('Severity')['Accuracy'].mean()

        plt.figure(figsize=(9, 6))
        plt.plot(base_avg.index, base_avg.values, marker='o', color='#d9534f', linewidth=2.5, label='Standard Baseline (No Adaptation)')
        plt.plot(adabn_avg.index, adabn_avg.values, marker='s', color='#0275d8', linewidth=2.5, label='AdaBN (Test-Time Adaptation)')

        plt.title('Average Robustness: Standard Baseline vs. AdaBN', fontsize=14, fontweight='bold')
        plt.xlabel('Corruption Severity Level (1 to 5)', fontsize=12)
        plt.ylabel('Mean Accuracy Across 14 Noise Types (%)', fontsize=12)
        plt.xticks([1, 2, 3, 4, 5])
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower left', fontsize=11)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format='pdf')
        plt.close()
        print(f"[Evaluator] AdaBN comparison plot saved to: '{output_path}'")

    def eval_model(self, model_cls, model_name=None, dataset='gtsrb', epochs=5, batch_size=64, num_workers=0, load_if_exists=True, use_adabn=False):
        """
        Main orchestration method:
        1. Loads train and test datasets scaled to target_size (49x49) for the specified dataset.
        2. Creates an instance of model_cls with appropriate num_classes.
        3. Trains the model instance (or loads existing checkpoint if available and requested).
        4. Calls test_without_noise.
        5. Calls test_with_noise (with optional AdaBN adaptation).
        
        Args:
            model_cls: Class reference of the model (e.g., Baseline).
            model_name: Directory name under output/ (e.g., 'baseline' or 'baseline_<dataset>').
            dataset: Target dataset ('gtsrb', 'btsd', or 'ctsd').
            epochs: Training epochs if training from scratch.
            batch_size: Training batch size.
            num_workers: Number of DataLoader worker processes (default: 0).
            load_if_exists: If True, attempts to load existing checkpoint before training.
            use_adabn: If True, benchmarks using AdaBN test-time adaptation.
            
        Returns:
            dict: Summary containing clean results and robustness results dataframe.
        """
        if model_name is None:
            model_name = f"baseline_{dataset}" if dataset != 'gtsrb' else "baseline"

        output_subdir = f"{model_name}_adabn" if use_adabn else model_name
        num_classes = DATASET_CLASSES.get(dataset, 43)

        print(f"\n==========================================")
        print(f" Starting Evaluation Pipeline for: {output_subdir}")
        print(f" Dataset: {dataset.upper()} ({num_classes} classes)")
        print(f" Mode: {'AdaBN Test-Time Adaptation' if use_adabn else 'Standard Clean/Noisy Evaluation'}")
        print(f"==========================================")

        # 1. Load train and test datasets scaled down to target_size (49x49)
        train_dataset = get_train_dataset(dataset=dataset, target_size=self.target_size)
        test_dataset = get_test_dataset(dataset=dataset, target_size=self.target_size)
        print(f"[Evaluator] Loaded Train set ({len(train_dataset)} samples) and Test set ({len(test_dataset)} samples)")

        # 2. Create model instance
        model_instance = model_cls(num_classes=num_classes)
        checkpoint_path = os.path.join('output', model_name, f"{model_name}_model.pth")
        if not os.path.exists(checkpoint_path):
            alt_path = os.path.join('output', model_name, 'baseline_model.pth')
            if os.path.exists(alt_path):
                checkpoint_path = alt_path

        # 3. Train or load existing weights
        weights_loaded = False
        if load_if_exists and hasattr(model_instance, 'load_weights'):
            weights_loaded = model_instance.load_weights(checkpoint_path)

        if not weights_loaded:
            print(f"[Evaluator] Training model instance on loaded dataset...")
            model_instance.train(
                dataset_or_mode=train_dataset,
                epochs=epochs,
                batch_size=batch_size,
                device=self.device,
                save_dir=os.path.join('output', model_name)
            )

        # 4. Clean evaluation (test_without_noise)
        clean_results = self.test_without_noise(
            model_instance=model_instance,
            test_dataset=test_dataset,
            model_name=output_subdir
        )

        # 5. Robustness evaluation with noise (test_with_noise)
        robustness_results = self.test_with_noise(
            model_instance=model_instance,
            test_dataset=test_dataset,
            model_name=model_name,
            num_workers=num_workers,
            use_adabn=use_adabn
        )

        # 6. If both baseline and AdaBN results exist, plot comparison
        base_csv = os.path.join('output', model_name, 'robustness_results.csv')
        adabn_csv = os.path.join('output', f"{model_name}_adabn", 'robustness_results.csv')
        if os.path.exists(base_csv) and os.path.exists(adabn_csv):
            comp_path = os.path.join('output', f"{model_name}_vs_adabn_comparison.pdf")
            self.plot_comparison(base_csv, adabn_csv, comp_path)

        print(f"\n==========================================")
        print(f" Finished Evaluation Pipeline for: {output_subdir}")
        print(f" Output artifacts stored in: output/{output_subdir}/")
        print(f"==========================================\n")

        return {
            'model_name': output_subdir,
            'clean_metrics': clean_results,
            'robustness_df': robustness_results
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Traffic Sign Recognition Models")
    parser.add_argument('--dataset', type=str, default='gtsrb', choices=['gtsrb', 'btsd', 'ctsd'], help="Target dataset (default: 'gtsrb')")
    parser.add_argument('--model', type=str, default=None, help="Model name (default: baseline or baseline_<dataset>)")
    parser.add_argument('--adabn', action='store_true', help="Enable AdaBN (Adaptive Batch Normalization) at test time")
    parser.add_argument('--batch-size', type=int, default=128, help="Inference batch size")
    parser.add_argument('--num-workers', type=int, default=0, help="Number of workers for data loading (default: 0)")
    parser.add_argument('--compare', action='store_true', help="Generate comparison plot from existing baseline & adabn CSVs")
    args = parser.parse_args()

    from models.baseline.model import Baseline
    evaluator = Evaluator()

    model_name = args.model if args.model else (f"baseline_{args.dataset}" if args.dataset != 'gtsrb' else "baseline")

    if args.compare:
        base_csv = os.path.join('output', model_name, 'robustness_results.csv')
        adabn_csv = os.path.join('output', f"{model_name}_adabn", 'robustness_results.csv')
        comp_path = os.path.join('output', f"{model_name}_vs_adabn_comparison.pdf")
        evaluator.plot_comparison(base_csv, adabn_csv, comp_path)
    else:
        evaluator.eval_model(
            Baseline,
            model_name=model_name,
            dataset=args.dataset,
            use_adabn=args.adabn,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            load_if_exists=True
        )
