import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from tqdm import tqdm
from dataset_loader import GTSRBDataset
from model import TrafficSignNet

def evaluate_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Load test dataset with 49x49 resolution
    transform = transforms.Compose([
        transforms.Resize((49, 49)),
        transforms.ToTensor()
    ])
    
    test_dataset = GTSRBDataset(csv_file='data/Test.csv', root_dir='data', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load trained weights from output/
    model_path = os.path.join('output', 'baseline_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please run train.py first.")

    model = TrafficSignNet(num_classes=43).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation on clean test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating Baseline", leave=True):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print("\n--- BASELINE RESULTS (CLEAN DATA) ---")
    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Macro F1-Score: {macro_f1:.4f}")

    # Save baseline confusion matrix to output/baseline/
    os.makedirs(os.path.join('output', 'baseline'), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 12))
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Baseline Confusion Matrix (Clean Data - 49x49)")
    plt.tight_layout()
    pdf_path = os.path.join('output', 'baseline', 'baseline_confusion_matrix.pdf')
    plt.savefig(pdf_path, format="pdf")
    print(f"Confusion matrix saved to '{pdf_path}'.")

if __name__ == "__main__":
    evaluate_baseline()