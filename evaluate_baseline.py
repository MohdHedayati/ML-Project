import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from dataset_loader import GTSRBDataset
from model import TrafficSignNet

def evaluate_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Load test dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    test_dataset = GTSRBDataset(csv_file='data/Test.csv', root_dir='data', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load trained weights
    model = TrafficSignNet(num_classes=43).to(device)
    model.load_state_dict(torch.load('baseline_model.pth', weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation on clean test set...")
    with torch.no_grad():
        for images, labels in test_loader:
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

    # Plot and save confusion matrix as PDF (per guideline requirement)
    fig, ax = plt.subplots(figsize=(12, 12))
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Baseline Confusion Matrix (Clean Data)")
    plt.tight_layout()
    plt.savefig("baseline_confusion_matrix.pdf", format="pdf")
    print("Confusion matrix saved to 'baseline_confusion_matrix.pdf'.")

if __name__ == "__main__":
    evaluate_baseline()