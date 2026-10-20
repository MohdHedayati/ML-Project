import argparse
import os
import torch

from dataset_loader import get_train_dataset, DATASET_CLASSES
from models.baseline.model import Baseline


def main():
    parser = argparse.ArgumentParser(description="Train Traffic Sign Recognition Model")
    parser.add_argument('--dataset', type=str, default='gtsrb', choices=['gtsrb', 'btsd', 'ctsd'], help="Dataset to train on: 'gtsrb', 'btsd', or 'ctsd' (default: gtsrb)")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument('--save-dir', type=str, default=None, help="Directory to save trained weights (default: output/baseline or output/baseline_<dataset>)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to train on: 'cuda' or 'cpu' (default: cuda if available)")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available():
        print(f"[Train] Accelerating with GPU: {torch.cuda.get_device_name(device)} (CUDA)")
    else:
        print(f"[Train] Running on device: {device}")

    num_classes = DATASET_CLASSES[args.dataset]
    save_dir = args.save_dir if args.save_dir else (f"output/baseline_{args.dataset}" if args.dataset != 'gtsrb' else "output/baseline")

    print(f"[Train] Loading {args.dataset.upper()} training dataset (resized to 49x49)...")
    train_dataset = get_train_dataset(dataset=args.dataset)
    print(f"[Train] Total training images: {len(train_dataset)}, Number of classes: {num_classes}")

    model = Baseline(num_classes=num_classes)

    model.train(
        dataset_or_mode=train_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        save_dir=save_dir
    )


if __name__ == "__main__":
    main()
