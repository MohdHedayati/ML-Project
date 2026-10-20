"""
Utility to inspect PyTorch checkpoint (.pth) files:
Displays layer names, shapes, parameter counts, and weight statistics.
"""

import sys
import os
import torch


def inspect_pth(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"\n{'='*70}")
    print(f" Inspecting Checkpoint: {file_path}")
    print(f" File Size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    print(f"{'='*70}\n")

    try:
        checkpoint = torch.load(file_path, weights_only=True, map_location='cpu')
    except Exception as e:
        # Fallback if checkpoint contains custom pickled objects
        checkpoint = torch.load(file_path, weights_only=False, map_location='cpu')

    # If wrapped in a dictionary (e.g., {'state_dict': ..., 'epoch': ...})
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"Extra Metadata:")
        for k in checkpoint.keys():
            if k != 'state_dict':
                print(f"  - {k}: {checkpoint[k]}")
        print()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        print(f"Loaded object type: {type(checkpoint)}")
        return

    print(f"{'Layer / Parameter Name':<40} {'Shape':<20} {'Params':<12} {'Mean / Std'}")
    print("-" * 85)

    total_params = 0
    trainable_params = 0

    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            numel = tensor.numel()
            total_params += numel
            is_floating = tensor.is_floating_point()
            stats = f"{tensor.mean().item():+.4f} / {tensor.std().item():.4f}" if is_floating else "N/A"
            shape_str = str(tuple(tensor.shape))
            print(f"{name:<40} {shape_str:<20} {numel:<12,d} {stats}")
        else:
            print(f"{name:<40} {type(tensor)}")

    print("-" * 85)
    print(f"Total Parameters: {total_params:,} ({total_params * 4 / (1024*1024):.2f} MB in float32)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "output/baseline/baseline_model.pth"
    inspect_pth(target_path)
