import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128

import os
import argparse
from multiprocessing import Pool
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from imagecorruptions import corrupt
from tqdm import tqdm

from model import TrafficSignNet

# The 14 noise types required by your project guidelines
NOISE_TYPES = [
    'gaussian_noise', 'impulse_noise', 'shot_noise', 'snow', 'fog',
    'frost', 'motion_blur', 'zoom_blur', 'defocus_blur', 'brightness',
    'contrast', 'pixelate', 'elastic_transform', 'jpeg_compression'
]

# Custom Dataset class that applies noise corruptions on the fly (Noise FIRST, then downscale)
class CorruptedGTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, corruption_name, severity, target_size=(49, 49)):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.corruption_name = corruption_name
        self.severity = severity
        self.target_size = target_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index]['Path'])
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        # Ensure image meets target_size and imagecorruptions minimum dimension (>=32)
        if w < self.target_size[0] or h < self.target_size[1]:
            image = image.resize((max(w, self.target_size[0]), max(h, self.target_size[1])))
            
        image_np = np.array(image)

        # 1. Apply noise corruption FIRST on native / high-resolution image
        corrupted_np = corrupt(image_np, corruption_name=self.corruption_name, severity=self.severity)
        
        # 2. Down-scale corrupted image to target resolution (49x49)
        corrupted_pil = Image.fromarray(corrupted_np).resize(self.target_size)
        image_tensor = self.transform(corrupted_pil)
        
        y_label = int(self.annotations.iloc[index]['ClassId'])
        return image_tensor, y_label

def save_severity_5_confusion_matrix(all_labels, all_preds, noise_name, output_dir):
    """Generates and saves the severity 5 confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(10, 10))
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(f"Confusion Matrix: {noise_name} (Severity 5 - 49x49)")
    plt.tight_layout()
    cm_pdf = os.path.join(output_dir, f"{noise_name}_severity_5.pdf")
    plt.savefig(cm_pdf, format="pdf")
    plt.close(fig)

def evaluate_single_experiment(task_args):
    """
    Evaluates a single (noise_type, severity) experiment.
    Used for CPU multiprocessing worker processes.
    """
    noise, severity, model_path, test_csv, root_dir, confusion_dir = task_args
    
    device = torch.device("cpu")
    model = TrafficSignNet(num_classes=43).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    test_dataset = CorruptedGTSRBDataset(
        csv_file=test_csv,
        root_dir=root_dir,
        corruption_name=noise,
        severity=severity,
        target_size=(49, 49)
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    if severity == 5:
        save_severity_5_confusion_matrix(all_labels, all_preds, noise, confusion_dir)

    return {
        'Noise_Type': noise,
        'Severity': severity,
        'Accuracy': round(acc, 2),
        'Macro_F1': round(macro_f1, 4)
    }

def run_evaluation_gpu(tasks, model_path, batch_size=256):
    """
    GPU evaluation mode:
    Runs sequential experiments on the GPU while streaming batches with pinned memory.
    """
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"CUDA detected! Running on GPU: {gpu_name}")
    print("Evaluating 70 corruption experiments on GPU...\n")

    model = TrafficSignNet(num_classes=43).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    results = []
    for noise, severity, _, test_csv, root_dir, confusion_dir in tqdm(tasks, desc="GPU Robustness Benchmark"):
        test_dataset = CorruptedGTSRBDataset(
            csv_file=test_csv,
            root_dir=root_dir,
            corruption_name=noise,
            severity=severity,
            target_size=(49, 49)
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True
        )

        all_preds = []
        all_labels = []

        with torch.inference_mode():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds) * 100
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        if severity == 5:
            save_severity_5_confusion_matrix(all_labels, all_preds, noise, confusion_dir)

        results.append({
            'Noise_Type': noise,
            'Severity': severity,
            'Accuracy': round(acc, 2),
            'Macro_F1': round(macro_f1, 4)
        })

    return results

def run_evaluation_cpu(tasks, num_workers=4):
    """
    CPU evaluation mode:
    Uses task-level multiprocessing pool across CPU cores.
    """
    print(f"Running on CPU: Parallelizing across {num_workers} worker processes for 70 experiments...\n")
    results = []
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            for res in tqdm(pool.imap_unordered(evaluate_single_experiment, tasks), total=len(tasks), desc="CPU Robustness Benchmark"):
                results.append(res)
    else:
        for task in tqdm(tasks, desc="CPU Robustness Benchmark"):
            results.append(evaluate_single_experiment(task))
    return results

def run_robustness_evaluation(device_choice='auto', cpu_workers=4):
    # Determine target hardware
    has_cuda = torch.cuda.is_available()
    if device_choice == 'auto':
        use_gpu = has_cuda
    elif device_choice == 'cuda':
        if not has_cuda:
            raise RuntimeError("CUDA was explicitly requested, but torch.cuda.is_available() is False.")
        use_gpu = True
    else:
        use_gpu = False

    # Load baseline model weights check
    model_path = os.path.join('output', 'baseline_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please run train.py first.")

    confusion_dir = os.path.join('output', 'noise', 'confusion_matrices')
    os.makedirs(confusion_dir, exist_ok=True)

    # Build the 70 evaluation tasks (14 noise types x 5 severities)
    tasks = [
        (noise, severity, model_path, 'data/Test.csv', 'data', confusion_dir)
        for noise in NOISE_TYPES
        for severity in range(1, 6)
    ]

    # Run either GPU or CPU pipeline
    if use_gpu:
        results = run_evaluation_gpu(tasks, model_path)
    else:
        results = run_evaluation_cpu(tasks, num_workers=cpu_workers)

    # Sort results to keep a consistent table order (same as NOISE_TYPES, sorted by severity)
    results.sort(key=lambda x: (NOISE_TYPES.index(x['Noise_Type']), x['Severity']))

    # Save results to output/noise/robustness_results.csv
    csv_path = os.path.join('output', 'noise', 'robustness_results.csv')
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nRobustness evaluation complete! Results saved to '{csv_path}'.")

def main():
    parser = argparse.ArgumentParser(description="Robustness Evaluation: Defaults to GPU if available; uses CPU multiprocessing otherwise.")
    default_workers = min(6, max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], default='auto',
                        help="Device to run on: 'auto' (default: uses GPU if available, else CPU), 'cuda', or 'cpu'.")
    parser.add_argument('--workers', type=int, default=default_workers, 
                        help=f"Number of parallel worker processes when running on CPU (default: {default_workers}).")
    args = parser.parse_args()
    
    run_robustness_evaluation(device_choice=args.device, cpu_workers=args.workers)

if __name__ == "__main__":
    main()