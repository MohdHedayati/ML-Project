"""
Model-Agnostic Evaluation Pipeline for Traffic Sign Recognition.

Handles dataset loading, hardware setup (CUDA/CPU), clean evaluation, and 
corruption robustness benchmarks. Completely model-agnostic: does NOT import 
any specific neural network model. Any model adhering to the project interface 
can be evaluated by passing its class or instance.

Methods:
- test_without_noise: evaluates clean test data, saves confusion matrix to <output_dir>/test_without_noise.pdf
- test_with_noise: evaluates across 14 noise types x 5 severities,
  saves 14 confusion matrices to <output_dir>/noise/<noise>_severity_5.pdf,
  saves robustness_degradation.pdf and robustness_results.csv to <output_dir>/
- eval_model: orchestrates dataset loading, hardware detection, training/checkpoint loading,
  clean evaluation, and robustness benchmarks for any passed model.
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

from dataset_loader import get_train_dataset, get_test_dataset, TOTAL_CLASSES
from noise import NOISE_TYPES, CorruptedGTSRBDataset


class Evaluator:
    """
    Evaluation engine for traffic sign recognition models.
    Manages dataset loading, hardware resource detection (GPU/CPU), and evaluation benchmarks.
    Does not import or couple to any specific model architecture.
    """
    def __init__(self, data_dir='data', target_size=(49, 49), device=None, num_workers=4):
        self.data_dir = data_dir
        self.target_size = tuple(target_size)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.num_workers = num_workers if num_workers is not None else 0
        print(f"[Evaluator] Initialized on device: {self.device} (Target Size: {self.target_size}, Workers: {self.num_workers})")

    def test_without_noise(self, model_instance, test_dataset, output_dir, batch_size=64):
        """
        Evaluates the model on clean test data scaled to target_size.
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

    def test_with_noise(self, model_instance, test_dataset, output_dir, batch_size=128, num_workers=None, pre_test_hook=None):
        """
        Evaluates the model across 14 noise types at 5 severity levels (70 experiments).
        
        Args:
            model_instance: Model instance conforming to model interface contract.
            test_dataset: PyTorch Dataset of test samples.
            output_dir: Destination directory for artifacts and matrices.
            batch_size: Mini-batch size for DataLoader.
            num_workers: Multiprocessing worker count for parallel noise generation.
            pre_test_hook: Optional callable `pre_test_hook(model_instance, corrupted_loader)`
                           invoked before evaluating each corruption (e.g. test-time adaptation).
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

        hook_status = f" [Pre-Test Hook Active]" if pre_test_hook else ""
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
        model_cls_or_instance,
        model_name,
        output_dir=None,
        train_dataset=None,
        test_dataset=None,
        epochs=5,
        batch_size=64,
        num_workers=None,
        weights_path=None,
        load_if_exists=True,
        pre_test_hook=None
    ):
        """
        Main orchestration method:
        1. Loads datasets if not already supplied.
        2. Instantiates model if class type was passed.
        3. Loads checkpoint or trains model.
        4. Configures adaptation hook if model provides one (e.g. adapt_bn).
        5. Runs clean evaluation and corrupted benchmark.
        """
        output_dir = output_dir or os.path.join('output', model_name)
        os.makedirs(output_dir, exist_ok=True)
        workers = num_workers if num_workers is not None else self.num_workers

        print(f"\n==========================================")
        print(f" Starting Evaluation Pipeline -> {output_dir}")
        print(f" Model: {model_name}")
        print(f" Datasets: Combined Multi-Domain ({TOTAL_CLASSES} Unified Semantic Classes)")
        print(f" Hardware: {self.device} | Target: {self.target_size[0]}x{self.target_size[1]} | Workers: {workers}")
        print(f"==========================================")

        # 1. Dataset loading handled directly by eval
        if train_dataset is None:
            print(f"[Evaluator] Loading training dataset from '{self.data_dir}' (Target: {self.target_size})...")
            train_dataset = get_train_dataset(root_dir=self.data_dir, target_size=self.target_size)
        if test_dataset is None:
            print(f"[Evaluator] Loading test dataset from '{self.data_dir}' (Target: {self.target_size})...")
            test_dataset = get_test_dataset(root_dir=self.data_dir, target_size=self.target_size)
        print(f"[Evaluator] Datasets ready: {len(train_dataset):,} train, {len(test_dataset):,} test samples.")

        # 2. Model preparation (agnostic: accepts class or instance)
        if isinstance(model_cls_or_instance, type):
            model = model_cls_or_instance(num_classes=TOTAL_CLASSES)
        else:
            model = model_cls_or_instance

        # 3. Checkpoint resolution & loading
        if weights_path is None:
            candidates = [
                os.path.join(output_dir, f"{model_name}_model.pth"),
                os.path.join(output_dir, 'model.pth'),
                os.path.join('models', model_name, f"{model_name}_model.pth"),
            ]
            weights_path = next((c for c in candidates if os.path.exists(c)), candidates[0])

        weights_loaded = False
        if load_if_exists and weights_path and os.path.exists(weights_path) and hasattr(model, 'load_weights'):
            weights_loaded = model.load_weights(weights_path)

        # 4. Training if no weights loaded
        if not weights_loaded:
            print(f"[Evaluator] Training {model_name} on training dataset...")
            model.train(
                dataset_or_mode=train_dataset,
                epochs=epochs,
                batch_size=batch_size,
                device=self.device,
                save_dir=output_dir
            )

        # 5. Generic pre-test adaptation hook (auto-detected if model defines adapt_bn or adapt)
        if pre_test_hook is None:
            if hasattr(model, 'adapt_bn'):
                pre_test_hook = lambda m, loader: m.adapt_bn(loader, device=self.device, verbose=False)
            elif hasattr(model, 'adapt'):
                pre_test_hook = lambda m, loader: m.adapt(loader, device=self.device)

        # 6. Clean evaluation
        clean_results = self.test_without_noise(
            model_instance=model,
            test_dataset=test_dataset,
            output_dir=output_dir,
            batch_size=batch_size
        )

        # 7. Robustness corrupted benchmark
        robustness_results = self.test_with_noise(
            model_instance=model,
            test_dataset=test_dataset,
            output_dir=output_dir,
            batch_size=batch_size * 2,
            num_workers=workers,
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
