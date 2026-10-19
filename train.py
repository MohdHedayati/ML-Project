
import argparse
import os
import torch

from dataset_loader import get_train_dataset
from models.baseline.model import Baseline


def main():
    parser = argparse.ArgumentParser(description="Train GTSRB Traffic Sign Recognition Model")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument('--save-dir', type=str, default='output/baseline', help="Directory to save trained weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Running on device: {device}")


    print("[Train] Loading GTSRB training dataset (resized to 49x49)...")
    train_dataset = get_train_dataset()
    print(f"[Train] Total training images: {len(train_dataset)}")


    model = Baseline(num_classes=43)

    model.train(
        dataset_or_mode=train_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
