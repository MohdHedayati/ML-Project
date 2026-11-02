"""
Noise Feature Map Visualizer for Traffic Sign Recognition.

Applies all 14 benchmark noise corruptions (at configurable severity 1 to 5)
to a single input image and visualizes how each corruption alters the internal
convolutional feature representations (Conv1, Conv2, Conv3) and final classification.

Corruptions visualized:
1. gaussian_noise       2. impulse_noise      3. shot_noise
4. snow                 5. fog                6. frost
7. motion_blur          8. zoom_blur          9. defocus_blur
10. brightness          11. contrast          12. pixelate
13. elastic_transform   14. jpeg_compression

Usage:
    python visualize_noise_feature_maps.py --image data/Test/00001.png --severity 3
    python visualize_noise_feature_maps.py --checkpoint output/adabn/adabn_model.pth --severity 4
"""

import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import noise
from models.baseline.model import Baseline
from models.adabn.model import AdaBN


GTSRB_CLASSES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing veh > 3.5t",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles > 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing veh > 3.5t",
}


def load_model(checkpoint_path: str) -> nn.Module:
    """Instantiates and loads weights into the matching architecture."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # Determine model type from path or keys
    path_lower = checkpoint_path.lower()
    if "adabn" in path_lower or "features.0.weight" in ckpt:
        model = AdaBN(num_classes=43)
        model.load_state_dict(ckpt)
    else:
        model = Baseline(num_classes=43)
        model.load_state_dict(ckpt)

    model.eval()
    return model


def extract_activations_and_pred(
    model: nn.Module, tensor_img: torch.Tensor
) -> Tuple[Dict[str, np.ndarray], int, float]:
    """
    Passes a normalized (1, 3, 49, 49) tensor through the model.
    Returns:
        dict: Layer activations for 'conv1', 'conv2', 'conv3' as numpy arrays
        int: Predicted class index
        float: Softmax confidence percentage
    """
    activations = {}

    with torch.no_grad():
        if isinstance(model, Baseline):
            # Conv 1
            x1 = model.act1(model.conv1(tensor_img))
            activations["conv1"] = x1.squeeze(0).cpu().numpy()
            p1 = model.pool1(x1)

            # Conv 2
            x2 = model.act2(model.conv2(p1))
            activations["conv2"] = x2.squeeze(0).cpu().numpy()
            p2 = model.pool2(model.drop1(x2))

            # Conv 3
            x3 = model.act3(model.conv3(p2))
            activations["conv3"] = x3.squeeze(0).cpu().numpy()
            p3 = model.pool3(x3)

            # Head
            flat = model.flatten(p3)
            logits = model.fc2(model.drop2(model.act4(model.fc1(flat))))

        elif isinstance(model, AdaBN):
            # Feature extractor sequential
            # features[0]: Conv2d(3, 32)
            # features[1]: BatchNorm2d(32)
            # features[2]: ReLU
            # features[3]: MaxPool2d(2)
            x1 = model.features[2](model.features[1](model.features[0](tensor_img)))
            activations["conv1"] = x1.squeeze(0).cpu().numpy()
            p1 = model.features[3](x1)

            # features[4]: Conv2d(32, 64)
            # features[5]: BatchNorm2d(64)
            # features[6]: ReLU
            # features[7]: Dropout2d
            # features[8]: MaxPool2d(2)
            x2 = model.features[6](model.features[5](model.features[4](p1)))
            activations["conv2"] = x2.squeeze(0).cpu().numpy()
            p2 = model.features[8](model.features[7](x2))

            # features[9]: Conv2d(64, 128)
            # features[10]: BatchNorm2d(128)
            # features[11]: ReLU
            # features[12]: MaxPool2d(2)
            x3 = model.features[11](model.features[10](model.features[9](p2)))
            activations["conv3"] = x3.squeeze(0).cpu().numpy()
            p3 = model.features[12](x3)

            logits = model.classifier(p3)

        else:
            raise TypeError("Unsupported model type.")

        probs = F.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item() * 100.0

    return activations, pred_idx, confidence


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two feature arrays."""
    v1 = vec1.flatten()
    v2 = vec2.flatten()
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def select_informative_channels(clean_conv1: np.ndarray, num_channels: int = 5) -> List[int]:
    """Selects the most spatially informative channels based on activation variance."""
    # clean_conv1 shape: (32, 49, 49)
    variances = np.var(clean_conv1, axis=(1, 2))
    sorted_indices = np.argsort(-variances)
    return sorted_indices[:num_channels].tolist()


def generate_noise_feature_maps_matrix(
    image_path: str,
    checkpoint_path: str = "output/baseline/baseline_model.pth",
    severity: int = 3,
    output_dir: str = "output/noise_feature_maps",
    dpi: int = 300,
    show: bool = False,
) -> List[str]:
    """
    Generates a full comparative matrix of Clean + 14 Noises across convolutional feature maps.
    Saves:
        1. 01_all_14_noises_feature_matrix.png
        2. 02_noise_layer_degradation_metrics.png
        3. 03_feature_diff_heatmaps.png
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    print(f"\n[NoiseVisualizer] Loading checkpoint: '{checkpoint_path}'...")
    model = load_model(checkpoint_path)
    model_name = "AdaBN" if isinstance(model, AdaBN) else "Baseline"

    # Load high-resolution native image
    pil_img = Image.open(image_path).convert("RGB")
    img_basename = os.path.basename(image_path)
    img_name_clean = os.path.splitext(img_basename)[0]

    # Create subfolder for this image
    image_out_dir = os.path.join(output_dir, f"{img_name_clean}_sev{severity}_{model_name.lower()}")
    os.makedirs(image_out_dir, exist_ok=True)

    print(f"[NoiseVisualizer] Processing image: '{image_path}' across Clean + 14 Noises (Severity {severity})...")

    # 1. Clean Baseline Condition
    clean_scaled = pil_img.resize((49, 49), Image.Resampling.BILINEAR)
    t_clean = transforms.ToTensor()(clean_scaled).unsqueeze(0)
    clean_acts, clean_pred, clean_conf = extract_activations_and_pred(model, t_clean)
    clean_label_str = GTSRB_CLASSES.get(clean_pred, f"Class {clean_pred}")

    # Pick top informative channels for Conv1, Conv2, Conv3
    top_c1 = select_informative_channels(clean_acts["conv1"], num_channels=5)
    top_c2 = select_informative_channels(clean_acts["conv2"], num_channels=2)
    top_c3 = select_informative_channels(clean_acts["conv3"], num_channels=2)

    # 2. Corruptions
    conditions = [("Clean", clean_scaled, clean_acts, clean_pred, clean_conf, 1.0, 0.0)]
    metrics = {
        "noise_name": [],
        "conv1_cos": [],
        "conv2_cos": [],
        "conv3_cos": [],
        "conv1_energy_ratio": [],
        "pred": [],
        "conf": [],
        "correct": [],
    }

    clean_c1_energy = np.linalg.norm(clean_acts["conv1"])

    for noise_name in noise.NOISE_TYPES:
        corrupted_pil = noise.add_noise_and_scale(
            pil_img, noise_name, severity=severity, target_size=(49, 49)
        )
        t_noisy = transforms.ToTensor()(corrupted_pil).unsqueeze(0)
        noisy_acts, pred, conf = extract_activations_and_pred(model, t_noisy)

        # Compute cosine similarities vs Clean
        c1_cos = compute_cosine_similarity(noisy_acts["conv1"], clean_acts["conv1"])
        c2_cos = compute_cosine_similarity(noisy_acts["conv2"], clean_acts["conv2"])
        c3_cos = compute_cosine_similarity(noisy_acts["conv3"], clean_acts["conv3"])
        energy_ratio = float(np.linalg.norm(noisy_acts["conv1"]) / (clean_c1_energy + 1e-8))
        mse_diff = float(np.mean((noisy_acts["conv1"] - clean_acts["conv1"]) ** 2))

        conditions.append((noise_name, corrupted_pil, noisy_acts, pred, conf, c1_cos, mse_diff))

        metrics["noise_name"].append(noise_name.replace("_", " ").title())
        metrics["conv1_cos"].append(c1_cos)
        metrics["conv2_cos"].append(c2_cos)
        metrics["conv3_cos"].append(c3_cos)
        metrics["conv1_energy_ratio"].append(energy_ratio)
        metrics["pred"].append(pred)
        metrics["conf"].append(conf)
        metrics["correct"].append(pred == clean_pred)

    print(f"[NoiseVisualizer] Generating 15-condition feature map matrix...")

    # =========================================================================
    # FIGURE 1: Full 15-Row Feature Map Matrix
    # =========================================================================
    # Rows: 15 (Clean + 14 noises)
    # Cols:
    # 0: Image (RGB) + metadata box
    # 1-5: Top 5 Conv1 feature channels
    # 6-7: Top 2 Conv2 feature channels
    # 8-9: Top 2 Conv3 feature channels
    total_rows = len(conditions)  # 15
    fig1 = plt.figure(figsize=(19, 1.45 * total_rows + 1.2), constrained_layout=True)
    gs = GridSpec(
        total_rows,
        11,
        width_ratios=[2.2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5],
        figure=fig1,
    )

    col_headers = [
        "Input (49×49 RGB)",
        f"Conv1\nCh #{top_c1[0]}",
        f"Conv1\nCh #{top_c1[1]}",
        f"Conv1\nCh #{top_c1[2]}",
        f"Conv1\nCh #{top_c1[3]}",
        f"Conv1\nCh #{top_c1[4]}",
        f"Conv2\nCh #{top_c2[0]}",
        f"Conv2\nCh #{top_c2[1]}",
        f"Conv3\nCh #{top_c3[0]}",
        f"Conv3\nCh #{top_c3[1]}",
        "Robustness Metrics",
    ]

    for row_idx, (name, img_patch, acts, pred, conf, cos_sim, mse_val) in enumerate(conditions):
        is_correct = pred == clean_pred
        status_color = "#1b9e77" if is_correct else "#d95f02"
        pred_label_str = GTSRB_CLASSES.get(pred, f"Class {pred}")
        if len(pred_label_str) > 22:
            pred_label_str = pred_label_str[:20] + "..."

        # Col 0: Corrupted Image Patch
        ax_img = fig1.add_subplot(gs[row_idx, 0])
        ax_img.imshow(img_patch)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # Condition title
        clean_name = name.replace("_", " ").title() if name != "Clean" else "Clean (Ground Truth)"
        badge = " [CLEAN]" if name == "Clean" else f" [Sev {severity}]"
        ax_img.set_ylabel(
            f"{clean_name}{badge}",
            fontsize=8,
            fontweight="bold",
            color="#222" if name != "Clean" else "#0055aa",
        )

        # Set headers on top row
        if row_idx == 0:
            ax_img.set_title(col_headers[0], fontsize=9, fontweight="bold")

        # Cols 1-5: Conv 1 feature maps
        for i, ch_idx in enumerate(top_c1):
            ax_c1 = fig1.add_subplot(gs[row_idx, 1 + i])
            fmap = acts["conv1"][ch_idx]
            ax_c1.imshow(fmap, cmap="viridis", interpolation="nearest")
            ax_c1.set_xticks([])
            ax_c1.set_yticks([])
            if row_idx == 0:
                ax_c1.set_title(col_headers[1 + i], fontsize=8, fontweight="bold")

        # Cols 6-7: Conv 2 feature maps
        for i, ch_idx in enumerate(top_c2):
            ax_c2 = fig1.add_subplot(gs[row_idx, 6 + i])
            fmap = acts["conv2"][ch_idx]
            ax_c2.imshow(fmap, cmap="inferno", interpolation="nearest")
            ax_c2.set_xticks([])
            ax_c2.set_yticks([])
            if row_idx == 0:
                ax_c2.set_title(col_headers[6 + i], fontsize=8, fontweight="bold")

        # Cols 8-9: Conv 3 feature maps
        for i, ch_idx in enumerate(top_c3):
            ax_c3 = fig1.add_subplot(gs[row_idx, 8 + i])
            fmap = acts["conv3"][ch_idx]
            ax_c3.imshow(fmap, cmap="plasma", interpolation="nearest")
            ax_c3.set_xticks([])
            ax_c3.set_yticks([])
            if row_idx == 0:
                ax_c3.set_title(col_headers[8 + i], fontsize=8, fontweight="bold")

        # Col 10: Quantitative metrics text block
        ax_metric = fig1.add_subplot(gs[row_idx, 10])
        ax_metric.axis("off")
        if row_idx == 0:
            ax_metric.set_title(col_headers[10], fontsize=8, fontweight="bold")

        metric_text = (
            f"Pred: {pred} ({pred_label_str})\n"
            f"Conf: {conf:.1f}% {'✓' if is_correct else '✗'}\n"
            f"CosSim: {cos_sim:.3f} | MSE: {mse_val:.4f}"
        )
        ax_metric.text(
            0.05,
            0.5,
            metric_text,
            transform=ax_metric.transAxes,
            fontsize=7.5,
            verticalalignment="center",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#f9f9f9" if is_correct else "#ffebee",
                edgecolor=status_color,
                linewidth=1.2,
                alpha=0.9,
            ),
        )

    fig1.suptitle(
        f"Feature Map Degradation Across 14 Corruptions (Severity {severity}): {model_name} Model\n"
        f"Input: {img_basename} [Ground Truth: Class {clean_pred} - {clean_label_str}]",
        fontsize=13,
        fontweight="bold",
    )

    path_matrix = os.path.join(image_out_dir, f"01_all_14_noises_feature_matrix.png")
    fig1.savefig(path_matrix, dpi=dpi, bbox_inches="tight")
    saved_files.append(path_matrix)
    print(f"  [Saved] Full Matrix -> {path_matrix}")
    if show:
        plt.show()
    else:
        plt.close(fig1)

    # =========================================================================
    # FIGURE 2: Quantitative Layer Degradation Charts
    # =========================================================================
    fig2, (ax_bar, ax_energy) = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    x = np.arange(len(metrics["noise_name"]))
    width = 0.26

    # Panel A: Cosine Similarity across layers
    rects1 = ax_bar.bar(x - width, metrics["conv1_cos"], width, label="Conv1 Cosine Sim", color="#1f77b4", alpha=0.85)
    rects2 = ax_bar.bar(x, metrics["conv2_cos"], width, label="Conv2 Cosine Sim", color="#ff7f0e", alpha=0.85)
    rects3 = ax_bar.bar(x + width, metrics["conv3_cos"], width, label="Conv3 Cosine Sim", color="#2ca02c", alpha=0.85)

    ax_bar.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="Clean Baseline (1.0)")
    ax_bar.set_ylabel("Representation Cosine Similarity (vs Clean)", fontsize=9, fontweight="bold")
    ax_bar.set_title("Layer-by-Layer Representation Drift (Cosine Similarity vs Clean Image)", fontsize=11, fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics["noise_name"], rotation=30, ha="right", fontsize=8)
    ax_bar.set_ylim(0.0, 1.1)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.5)
    ax_bar.legend(loc="lower right", fontsize=8)

    # Panel B: Energy Drift Ratio
    energy_colors = ["#1b9e77" if c else "#d95f02" for c in metrics["correct"]]
    ax_energy.bar(x, metrics["conv1_energy_ratio"], width=0.55, color=energy_colors, alpha=0.85, edgecolor="#333", linewidth=0.5)
    ax_energy.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Clean Energy Ratio (1.0)")
    ax_energy.set_ylabel("Conv1 L2 Energy Ratio (||A_noise|| / ||A_clean||)", fontsize=9, fontweight="bold")
    ax_energy.set_title(
        "Activation Magnitude Distortion (Green = Correct Classification, Red = Misclassified)",
        fontsize=11,
        fontweight="bold",
    )
    ax_energy.set_xticks(x)
    ax_energy.set_xticklabels(metrics["noise_name"], rotation=30, ha="right", fontsize=8)
    ax_energy.grid(axis="y", linestyle=":", alpha=0.5)
    ax_energy.legend(loc="upper right", fontsize=8)

    fig2.suptitle(
        f"Robustness Diagnostics Across 14 Noise Corruptions (Severity {severity}): {model_name}\n"
        f"Target Image: {img_basename} ({clean_label_str})",
        fontsize=12,
        fontweight="bold",
    )

    path_chart = os.path.join(image_out_dir, f"02_noise_layer_degradation_metrics.png")
    fig2.savefig(path_chart, dpi=dpi, bbox_inches="tight")
    saved_files.append(path_chart)
    print(f"  [Saved] Degradation Metrics -> {path_chart}")
    if show:
        plt.show()
    else:
        plt.close(fig2)

    # =========================================================================
    # FIGURE 3: Spatial Difference Heatmaps (|A_noise - A_clean|)
    # =========================================================================
    # Pick the most prominent rim/edge channel (top_c1[0])
    rim_channel = top_c1[0]
    clean_rim_map = clean_acts["conv1"][rim_channel]

    fig3, axes = plt.subplots(3, 5, figsize=(15, 8.5), constrained_layout=True)
    axes_flat = axes.flatten()

    # Position 0: Clean reference feature map
    im0 = axes_flat[0].imshow(clean_rim_map, cmap="viridis", interpolation="nearest")
    axes_flat[0].set_title(f"Clean Reference\n(Conv1 Ch #{rim_channel})", fontsize=8, fontweight="bold")
    axes_flat[0].axis("off")

    diff_vmax = 0.0
    for noise_name in noise.NOISE_TYPES:
        for n, _, acts, _, _, _, _ in conditions:
            if n == noise_name:
                diff = np.abs(acts["conv1"][rim_channel] - clean_rim_map)
                diff_vmax = max(diff_vmax, float(diff.max()))

    for idx, noise_name in enumerate(noise.NOISE_TYPES):
        ax = axes_flat[idx + 1]
        for n, _, acts, pred, conf, _, _ in conditions:
            if n == noise_name:
                diff = np.abs(acts["conv1"][rim_channel] - clean_rim_map)
                im = ax.imshow(diff, cmap="magma", vmin=0.0, vmax=max(diff_vmax, 1e-4), interpolation="nearest")
                is_correct = pred == clean_pred
                badge = "✓" if is_correct else "✗"
                c_name = noise_name.replace("_", " ").title()
                ax.set_title(f"{c_name} (Sev {severity})\nDiff MSE={np.mean(diff**2):.4f} [{badge}]", fontsize=7.5)
                ax.axis("off")

    cbar = fig3.colorbar(im0, ax=axes, orientation="vertical", fraction=0.015, pad=0.02)
    cbar.set_label(f"Absolute Activation Difference (|A_noise - A_clean|)", fontsize=8)

    fig3.suptitle(
        f"Spatial Feature Distortion Heatmaps: Conv1 Channel #{rim_channel} Across All 14 Corruptions\n"
        f"Image: {img_basename} [Class {clean_pred}: {clean_label_str}] - Severity {severity}",
        fontsize=12,
        fontweight="bold",
    )

    path_diff = os.path.join(image_out_dir, f"03_feature_diff_heatmaps.png")
    fig3.savefig(path_diff, dpi=dpi, bbox_inches="tight")
    saved_files.append(path_diff)
    print(f"  [Saved] Difference Heatmaps -> {path_diff}")
    if show:
        plt.show()
    else:
        plt.close(fig3)

    print(f"[NoiseVisualizer] Generated all 3 artifact suites in: '{image_out_dir}'\n")
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Visualize feature maps for a single image across all 14 benchmark corruptions."
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default="data/Test/00001.png",
        help="Path to input image (default: 'data/Test/00001.png').",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="output/baseline/baseline_model.pth",
        help="Path to .pth checkpoint (default: 'output/baseline/baseline_model.pth').",
    )
    parser.add_argument(
        "--severity",
        "-s",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Corruption severity level from 1 (mild) to 5 (extreme) (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output/noise_feature_maps",
        help="Directory to save generated visual figures (default: 'output/noise_feature_maps').",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI resolution (default: 300).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively with matplotlib popup window.",
    )

    args = parser.parse_args()

    generate_noise_feature_maps_matrix(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        severity=args.severity,
        output_dir=args.output_dir,
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
