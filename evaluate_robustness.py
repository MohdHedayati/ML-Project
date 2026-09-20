import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from imagecorruptions import corrupt

from model import TrafficSignNet

# The 14 noise types required by your project guidelines
NOISE_TYPES = [
    'gaussian_noise', 'impulse_noise', 'shot_noise', 'snow', 'fog',
    'frost', 'motion_blur', 'zoom_blur', 'defocus_blur', 'brightness',
    'contrast', 'pixelate', 'elastic_transform', 'jpeg_compression'
]

# Custom Dataset class that applies noise corruptions on the fly
class CorruptedGTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, corruption_name, severity):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.corruption_name = corruption_name
        self.severity = severity
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index]['Path'])
        image = Image.open(img_path).resize((32, 32))
        image_np = np.array(image)

        # Apply noise corruption
        corrupted_np = corrupt(image_np, corruption_name=self.corruption_name, severity=self.severity)
        
        # Convert to PIL Image then PyTorch Tensor
        corrupted_pil = Image.fromarray(corrupted_np)
        image_tensor = self.transform(corrupted_pil)
        
        y_label = int(self.annotations.iloc[index]['ClassId'])
        return image_tensor, y_label

def run_robustness_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running robustness evaluation on: {device}")

    # Load baseline model weights
    model = TrafficSignNet(num_classes=43).to(device)
    model.load_state_dict(torch.load('baseline_model.pth', weights_only=True))
    model.eval()

    results = []
    os.makedirs('confusion_matrices', exist_ok=True)

    total_experiments = len(NOISE_TYPES) * 5
    current_exp = 0

    for noise in NOISE_TYPES:
        for severity in range(1, 6):
            current_exp += 1
            print(f"[{current_exp}/{total_experiments}] Testing Noise: {noise} | Severity: {severity}...")

            test_dataset = CorruptedGTSRBDataset(
                csv_file='data/Test.csv',
                root_dir='data',
                corruption_name=noise,
                severity=severity
            )
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            acc = accuracy_score(all_labels, all_preds) * 100
            macro_f1 = f1_score(all_labels, all_preds, average='macro')

            results.append({
                'Noise_Type': noise,
                'Severity': severity,
                'Accuracy': round(acc, 2),
                'Macro_F1': round(macro_f1, 4)
            })

            # Save confusion matrix PDF for severe level 5 corruptions
            if severity == 5:
                fig, ax = plt.subplots(figsize=(10, 10))
                cm = confusion_matrix(all_labels, all_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(ax=ax, cmap='Blues', colorbar=False)
                plt.title(f"Confusion Matrix: {noise} (Severity 5)")
                plt.tight_layout()
                plt.savefig(f"confusion_matrices/{noise}_severity_5.pdf", format="pdf")
                plt.close()

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('robustness_results.csv', index=False)
    print("\nRobustness evaluation complete! Results saved to 'robustness_results.csv'.")

if __name__ == "__main__":
    run_robustness_evaluation()