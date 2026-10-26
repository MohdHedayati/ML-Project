"""
Model-Agnostic Evaluation Pipeline for Traffic Sign Recognition.

Provides a completely generic evaluation engine for benchmarking neural network models
under clean and corrupted conditions. Contains NO model-specific code or default assumptions;
all configuration parameters, datasets, directories, devices, and hooks must be explicitly specified.

Methods:
- test_without_noise: evaluates clean test data, saves confusion matrix to <output_dir>/test_without_noise.pdf
- test_with_noise: evaluates across 14 noise types x 5 severities,
  saves 14 confusion matrices to <output_dir>/noise/<noise>_severity_5.pdf,
  saves robustness_degradation.pdf and robustness_results.csv to <output_dir>/
- eval_model: orchestrates full pipeline (checkpoint loading/training, clean eval, robustness benchmark)
"""

import os
import copy
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

from dataset_loader import TOTAL_CLASSES
from noise import NOISE_TYPES, CorruptedGTSRBDataset


class Evaluator:
    """
    Model-agnostic evaluator class for benchmarking traffic sign recognition models
    under clean and corrupted conditions.
    
    Contains NO default values or model-specific logic; all settings must be specified.
    """
    def __init__(self, data_dir, target_size, device, num_workers):
        self.data_dir = data_dir
        self.target_size = tuple(target_size)
        self.device = torch.device(device)
        self.num_workers = int(num_workers)
        print(f"[Evaluator] Initialized on device: {self.device} (Target Size: {self.target_size}, Workers: {self.num_workers})")

    def test_without_noise(self, model_instance, test_dataset, output_dir, batch_size):
        """
        Evaluates the model on clean data (without noise) scaled to target_size.
        Generates and saves a confusion matrix to <output_dir>/test_without_noise.pdf.
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
        plt.title(
            f"Confusion Matrix (Clean Data - {TOTAL_CLASSES} Unified Classes - {self.target_size[0]}x{self.target_size[1]})",
            fontsize=14
        )
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

    def test_with_noise(self, model_instance, test_dataset, output_dir, batch_size, num_workers, pre_test_hook):
        """
        Evaluates the model across 14 noise types at 5 severity levels (70 experiments).
        
        Args:
            model_instance: Initialized model instance complying with the model interface contract.
            test_dataset: PyTorch Dataset of test samples.
            output_dir: Destination directory for artifacts and matrices.
            batch_size: Mini-batch size for DataLoader.
            num_workers: Multiprocessing worker count for parallel noise generation.
            pre_test_hook: Optional callable hook `pre_test_hook(model_instance, corrupted_loader)`
                           invoked before evaluating each corruption (e.g. for test-time adaptation).
        """
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

        hook_status = f" [Pre-Test Hook: {pre_test_hook.__name__ if hasattr(pre_test_hook, '__name__') else 'Active'}]" if pre_test_hook else ""
        print(f"\n[Evaluator] Running robustness benchmark{hook_status} across {len(NOISE_TYPES)} corruptions x 5 severities...")
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

                if pre_test_hook is not None:
                    model_instance.load_state_dict(copy.deepcopy(source_state))
                    pre_test_hook(model_instance, corrupted_loader)

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

        # Restore original uncorrupted weights
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

    def eval_model(
        self,
        model_instance,
        model_name,
        output_dir,
        train_dataset,
        test_dataset,
        epochs,
        batch_size,
        num_workers,
        weights_path,
        load_if_exists,
        pre_test_hook
    ):
        """
        Main orchestration method:
        1. Prepares model instance.
        2. Loads existing checkpoint from weights_path or trains on train_dataset.
        3. Runs clean evaluation (test_without_noise).
        4. Runs corrupted robustness benchmark (test_with_noise).
        """
        print(f"\n==========================================")
        print(f" Starting Evaluation Pipeline -> {output_dir}")
        print(f" Model: {model_name}")
        print(f" Datasets: Combined Multi-Domain ({TOTAL_CLASSES} Unified Semantic Classes)")
        print(f" Target Resolution: {self.target_size[0]}x{self.target_size[1]}")
        print(f"==========================================")

        # Support passing either a class type or an instantiated model
        if isinstance(model_instance, type):
            model = model_instance(num_classes=TOTAL_CLASSES)
        else:
            model = model_instance

        weights_loaded = False
        if load_if_exists and weights_path and os.path.exists(weights_path) and hasattr(model, 'load_weights'):
            weights_loaded = model.load_weights(weights_path)

        if not weights_loaded:
            print(f"[Evaluator] Training {model_name} on training dataset ({len(train_dataset):,} samples)...")
            model.train(
                dataset_or_mode=train_dataset,
                epochs=epochs,
                batch_size=batch_size,
                device=self.device,
                save_dir=output_dir
            )

        clean_results = self.test_without_noise(
            model_instance=model,
            test_dataset=test_dataset,
            output_dir=output_dir,
            batch_size=batch_size
        )

        robustness_results = self.test_with_noise(
            model_instance=model,
            test_dataset=test_dataset,
            output_dir=output_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pre_test_hook=pre_test_hook
        )

        print(f"\n==========================================")
        print(f" Finished Pipeline for {model_name}. Artifacts stored in: '{output_dir}/'")
        print(f"==========================================\n")

        return {
            'output_dir': output_dir,
            'clean_metrics': clean_results,
            'robustness_df': robustness_results
        }


if __name__ == '__main__':
    from cli import main
    main()
