"""
Kernel Visualization Suite for Convolutional Neural Network Checkpoints.

Inspects PyTorch *.pth checkpoints in the output/ directory (or user-specified paths),
extracts convolutional weight tensors, and produces comprehensive visual diagnostics:
1. First-Layer RGB Filters: Color patches and RGB channel breakdowns.
2. Kernel Slices: 2D weight slice matrices across output and input channels with diverging colormaps.
3. Channel Connectivity: Frobenius norm heatmaps showing (C_out, C_in) layer coupling.
4. Filter Energy / Norms: L2 norm distributions across individual filters.
5. Weight Value Distributions: Histograms and statistical diagnostics (mean, std, min, max, sparsity).
6. Cross-Model Kernel Comparison: Side-by-side filters, weight distributions, and cosine similarity.

Can be run directly via CLI or imported as a reusable module:
    python visualize_kernels.py
    python visualize_kernels.py --checkpoint output/baseline/baseline_model.pth
    python visualize_kernels.py --compare output/baseline/baseline_model.pth output/adabn/adabn_model.pth
"""

import os
import sys
import glob
import argparse
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib
# Use Agg backend if non-interactive (default) to speed up rendering and avoid GUI blocks
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def find_checkpoints(root_dir: str = "output") -> List[str]:
    """
    Recursively scans the root directory for all .pth checkpoint files,
    ignoring virtual environments (.venv, env, site-packages).
    """
    if not os.path.isdir(root_dir):
        return []

    found = []
    for dirpath, _, filenames in os.walk(root_dir):
        # Skip virtual environment folders if any
        if any(skip in dirpath.lower() for skip in ["site-packages", "env", ".venv", "__pycache__"]):
            continue
        for fname in filenames:
            if fname.lower().endswith(".pth"):
                found.append(os.path.abspath(os.path.join(dirpath, fname)))
    return sorted(found)


def load_checkpoint_weights(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Loads a .pth file safely on CPU and extracts the parameter dictionary.
    Handles raw state dicts, wrapper dictionaries, and serialized nn.Modules.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load on CPU
    data = torch.load(checkpoint_path, map_location="cpu")

    # If it's a full nn.Module instance
    if isinstance(data, nn.Module):
        return data.state_dict()

    # If it's an OrderedDict or dict
    if isinstance(data, (dict, OrderedDict)):
        # Check for nested state dict keys
        for key in ["state_dict", "model", "model_state_dict", "weights", "net"]:
            if key in data and isinstance(data[key], (dict, OrderedDict)):
                return data[key]
        # Check if values are tensors
        if any(isinstance(v, torch.Tensor) for v in data.values()):
            return data

    raise ValueError(f"Unrecognized checkpoint format in '{checkpoint_path}'. Expected state_dict or nn.Module.")


def extract_conv_weights(state_dict: Dict[str, torch.Tensor]) -> "OrderedDict[str, np.ndarray]":
    """
    Filters only 4D convolutional weight tensors (C_out, C_in, kH, kW).
    Returns an OrderedDict mapping layer name to numpy float32 array.
    """
    conv_layers = OrderedDict()
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            conv_layers[name] = tensor.detach().cpu().numpy().astype(np.float32)
    return conv_layers


def get_model_name_from_path(checkpoint_path: str) -> str:
    """Extracts a clean model label from checkpoint path or folder name."""
    fname = os.path.basename(checkpoint_path)
    parent = os.path.basename(os.path.dirname(checkpoint_path))
    # If the file is named baseline_model.pth or similar
    name = os.path.splitext(fname)[0]
    if name.endswith("_model"):
        name = name[:-6]
    if parent and parent not in ["output", ".", ""]:
        return f"{parent} ({name})" if parent.lower() != name.lower() else parent
    return name


class KernelVisualizer:
    """
    Visualizes convolutional kernels from a checkpoint state dictionary.
    """

    def __init__(self, checkpoint_path_or_dict: Union[str, Dict[str, torch.Tensor]], model_name: Optional[str] = None):
        if isinstance(checkpoint_path_or_dict, str):
            self.checkpoint_path = os.path.abspath(checkpoint_path_or_dict)
            self.model_name = model_name or get_model_name_from_path(self.checkpoint_path)
            self.state_dict = load_checkpoint_weights(self.checkpoint_path)
        elif isinstance(checkpoint_path_or_dict, (dict, OrderedDict)):
            self.checkpoint_path = None
            self.model_name = model_name or "CustomModel"
            self.state_dict = checkpoint_path_or_dict
        else:
            raise TypeError("Expected filepath or dictionary of tensors.")

        self.conv_weights = extract_conv_weights(self.state_dict)
        if not self.conv_weights:
            raise ValueError(f"No 4D convolutional weights found in model '{self.model_name}'.")

    def print_summary(self):
        """Prints a summary table of convolutional layers in the model."""
        print(f"\n{'='*70}")
        print(f" Convolutional Layers Summary: {self.model_name}")
        print(f"{'='*70}")
        print(f"{'Layer Name':<28} {'Shape (C_out, C_in, H, W)':<24} {'Params':<10} {'Mean':<8} {'Std':<8}")
        print(f"{'-'*70}")
        total_params = 0
        for name, w in self.conv_weights.items():
            params = w.size
            total_params += params
            shape_str = f"({w.shape[0]}, {w.shape[1]}, {w.shape[2]}, {w.shape[3]})"
            print(f"{name:<28} {shape_str:<24} {params:<10} {w.mean():<8.4f} {w.std():<8.4f}")
        print(f"{'-'*70}")
        print(f"Total Conv Parameters: {total_params:,}")
        print(f"{'='*70}\n")

    def plot_conv1_rgb(
        self,
        layer_name: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = False,
        dpi: int = 300,
    ) -> Optional[plt.Figure]:
        """
        Visualizes first-layer 3-channel (RGB) filters as color patches and individual R, G, B channels.
        """
        if layer_name is None:
            # Pick first conv layer with in_channels == 3
            candidates = [name for name, w in self.conv_weights.items() if w.shape[1] == 3]
            if not candidates:
                print(f"[Warning] No 3-channel input conv layer found in '{self.model_name}'. Skipping RGB plot.")
                return None
            layer_name = candidates[0]

        weights = self.conv_weights[layer_name]  # Shape: (N, 3, H, W)
        if weights.shape[1] != 3:
            raise ValueError(f"Layer '{layer_name}' has {weights.shape[1]} input channels; expected 3 for RGB.")

        num_filters = weights.shape[0]
        # Transpose to (N, H, W, 3) for RGB plotting
        rgb_weights = np.transpose(weights, (0, 2, 3, 1))

        # Setup 2-part figure:
        # Top: Grid of all RGB filters
        # Bottom: Breakdown of first 8 filters into [Composite, Red, Green, Blue]
        cols = 8
        rows = int(np.ceil(num_filters / cols))

        fig = plt.figure(figsize=(15, 2.2 * rows + 6), constrained_layout=True)
        gs = GridSpec(2, 1, height_ratios=[2.2 * rows, 5.5], figure=fig)

        # 1. Main RGB Grid
        sub_gs_top = gs[0].subgridspec(rows, cols, wspace=0.15, hspace=0.3)
        for i in range(num_filters):
            r = i // cols
            c = i % cols
            ax = fig.add_subplot(sub_gs_top[r, c])
            filt = rgb_weights[i]
            # Normalize to [0, 1] per filter
            f_min, f_max = filt.min(), filt.max()
            filt_norm = (filt - f_min) / (f_max - f_min + 1e-8)

            ax.imshow(filt_norm, interpolation="nearest")
            l2_norm = np.linalg.norm(weights[i])
            ax.set_title(f"F#{i}\n||w||={l2_norm:.2f}", fontsize=8)
            ax.axis("off")

        # Hide any unused axes in top grid
        for i in range(num_filters, rows * cols):
            r = i // cols
            c = i % cols
            ax = fig.add_subplot(sub_gs_top[r, c])
            ax.axis("off")

        # 2. RGB Channel Decomposition (first min(8, num_filters) filters)
        decomp_count = min(8, num_filters)
        sub_gs_bot = gs[1].subgridspec(4, decomp_count, wspace=0.15, hspace=0.25)
        channel_names = ["RGB Composite", "Red (R)", "Green (G)", "Blue (B)"]
        colormaps = [None, "Reds", "Greens", "Blues"]

        for col_idx in range(decomp_count):
            filt = rgb_weights[col_idx]
            f_norm = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)

            for row_idx in range(4):
                ax = fig.add_subplot(sub_gs_bot[row_idx, col_idx])
                if row_idx == 0:
                    ax.imshow(f_norm, interpolation="nearest")
                    ax.set_title(f"Filter #{col_idx}", fontsize=9, fontweight="bold")
                else:
                    ch_idx = row_idx - 1
                    ch_data = filt[:, :, ch_idx]
                    # Diverging or channel-specific colormap
                    ax.imshow(ch_data, cmap=colormaps[row_idx], interpolation="nearest")

                if col_idx == 0:
                    ax.set_ylabel(channel_names[row_idx], fontsize=8, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(
            f"First-Layer RGB Filters: {self.model_name} [{layer_name}] ({num_filters} filters, {weights.shape[2]}x{weights.shape[3]})",
            fontsize=13,
            fontweight="bold",
        )

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"  [Saved] RGB Filters -> {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_layer_kernel_slices(
        self,
        layer_name: str,
        num_out: int = 8,
        num_in: int = 8,
        save_path: Optional[str] = None,
        show: bool = False,
        dpi: int = 300,
    ) -> Optional[plt.Figure]:
        """
        Plots a 2D matrix of 3x3 filter slices for multi-channel conv layers (C_out x C_in).
        Uses a centered diverging colormap (coolwarm) where 0 is neutral white.
        """
        if layer_name not in self.conv_weights:
            raise KeyError(f"Layer '{layer_name}' not found. Available: {list(self.conv_weights.keys())}")

        weights = self.conv_weights[layer_name]  # (C_out, C_in, H, W)
        c_out, c_in, kh, kw = weights.shape

        sel_out = min(num_out, c_out)
        sel_in = min(num_in, c_in)

        # Slice of interest: (sel_out, sel_in, H, W)
        slice_data = weights[:sel_out, :sel_in]
        vmax = max(abs(slice_data.min()), abs(slice_data.max()), 1e-5)

        fig, axes = plt.subplots(
            sel_out,
            sel_in,
            figsize=(1.4 * sel_in + 1.5, 1.4 * sel_out + 1.0),
            squeeze=False,
            constrained_layout=True,
        )

        im = None
        for i in range(sel_out):
            for j in range(sel_in):
                ax = axes[i, j]
                kernel = slice_data[i, j]
                im = ax.imshow(kernel, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])

                if i == 0:
                    ax.set_title(f"In #{j}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"Out #{i}", fontsize=8)

        # Add shared colorbar
        cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.03)
        cbar.set_label("Kernel Weight Value", fontsize=9)

        fig.suptitle(
            f"Kernel Slices Matrix: {self.model_name} [{layer_name}]\n(Showing {sel_out} Out Filters × {sel_in} In Channels of {c_out}×{c_in})",
            fontsize=12,
            fontweight="bold",
        )

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"  [Saved] Kernel Slices -> {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_single_filter_channels(
        self,
        layer_name: str,
        filter_idx: int = 0,
        max_channels: int = 32,
        save_path: Optional[str] = None,
        show: bool = False,
        dpi: int = 300,
    ) -> Optional[plt.Figure]:
        """
        Shows all input channel receptive slices for a single designated filter.
        Reveals which input feature maps this specific neuron responds to.
        """
        if layer_name not in self.conv_weights:
            raise KeyError(f"Layer '{layer_name}' not found.")

        weights = self.conv_weights[layer_name]  # (C_out, C_in, H, W)
        c_out, c_in, kh, kw = weights.shape
        if filter_idx >= c_out:
            raise ValueError(f"filter_idx={filter_idx} exceeds C_out={c_out}.")

        channels_to_plot = min(c_in, max_channels)
        cols = 8
        rows = int(np.ceil(channels_to_plot / cols))

        f_weights = weights[filter_idx, :channels_to_plot]  # (channels, H, W)
        vmax = max(abs(f_weights.min()), abs(f_weights.max()), 1e-5)

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(1.5 * cols + 1.2, 1.5 * rows + 0.8),
            squeeze=False,
            constrained_layout=True,
        )

        im = None
        for idx in range(channels_to_plot):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            kernel = f_weights[idx]
            im = ax.imshow(kernel, cmap="seismic", vmin=-vmax, vmax=vmax, interpolation="nearest")
            l2 = np.linalg.norm(kernel)
            ax.set_title(f"Ch #{idx}\n||w||={l2:.2f}", fontsize=7)
            ax.axis("off")

        # Hide extra subplots
        for idx in range(channels_to_plot, rows * cols):
            r = idx // cols
            c = idx % cols
            axes[r, c].axis("off")

        cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.03)
        cbar.set_label("Kernel Weight", fontsize=8)

        fig.suptitle(
            f"Filter #{filter_idx} Input Channel Breakdown: {self.model_name} [{layer_name}] ({channels_to_plot}/{c_in} channels)",
            fontsize=11,
            fontweight="bold",
        )

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"  [Saved] Filter Channel Breakdown -> {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_channel_connectivity(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
        dpi: int = 300,
    ) -> Optional[plt.Figure]:
        """
        Plots Frobenius norm heatmaps ||W_{i, j, :, :}||_F across (C_out, C_in) for all conv layers.
        Visualizes inter-channel coupling, sparsity, and routing strength.
        """
        multi_channel_layers = {
            name: w for name, w in self.conv_weights.items() if w.shape[1] > 1
        }
        if not multi_channel_layers:
            print("[Warning] No multi-channel conv layers found for connectivity matrix.")
            return None

        n_layers = len(multi_channel_layers)
        fig, axes = plt.subplots(
            1, n_layers, figsize=(6.5 * n_layers, 5.5), squeeze=False, constrained_layout=True
        )

        for idx, (name, w) in enumerate(multi_channel_layers.items()):
            ax = axes[0, idx]
            # Frobenius norm of each 2D kernel slice (C_out, C_in)
            conn_matrix = np.linalg.norm(w, axis=(2, 3))
            c_out, c_in = conn_matrix.shape

            im = ax.imshow(conn_matrix, cmap="magma", aspect="auto", interpolation="nearest")
            ax.set_title(f"{name}\nShape: {c_out} Out × {c_in} In", fontsize=10, fontweight="bold")
            ax.set_xlabel("Input Channel Index", fontsize=9)
            ax.set_ylabel("Output Filter Index", fontsize=9)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Kernel Slice L2 Norm", fontsize=8)

        fig.suptitle(
            f"Channel Connectivity Matrices (Kernel Norms): {self.model_name}",
            fontsize=13,
            fontweight="bold",
        )

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"  [Saved] Channel Connectivity -> {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_filter_energy(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
        dpi: int = 300,
    ) -> Optional[plt.Figure]:
        """
        Computes and plots the L2 energy ||W_i||_2 for every output filter across all conv layers.
        Highlights dominant vs pruned or low-magnitude feature channels.
        """
        n_layers = len(self.conv_weights)
        fig, axes = plt.subplots(
            n_layers, 1, figsize=(10, 2.5 * n_layers), squeeze=False, constrained_layout=True
        )

        colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a"]

        for idx, (name, w) in enumerate(self.conv_weights.items()):
            ax = axes[idx, 0]
            # Filter L2 norms: sqrt(sum(w_i^2))
            filter_norms = np.linalg.norm(w.reshape(w.shape[0], -1), axis=1)
            n_filters = len(filter_norms)
            x = np.arange(n_filters)

            c = colors[idx % len(colors)]
            ax.bar(x, filter_norms, color=c, alpha=0.8, edgecolor="none", width=0.85)

            # Overlay mean and std band
            mean_val = np.mean(filter_norms)
            std_val = np.std(filter_norms)
            ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.3f}")
            ax.axhspan(mean_val - std_val, mean_val + std_val, color="red", alpha=0.1, label=f"±1 Std: {std_val:.3f}")

            ax.set_xlim(-0.5, n_filters - 0.5)
            ax.set_ylabel("Filter L2 Norm", fontsize=9)
            ax.set_title(f"{name} ({n_filters} Filters, shape {w.shape})", fontsize=10, fontweight="bold")
            ax.grid(axis="y", linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", fontsize=8)

            if idx == n_layers - 1:
                ax.set_xlabel("Filter Index (Output Channel)", fontsize=9)

        fig.suptitle(f"Filter Energy Profiles (L2 Norm per Filter): {self.model_name}", fontsize=13, fontweight="bold")

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"  [Saved] Filter Energy -> {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_weight_distributions(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
        dpi: int = 300,
    ) -> Optional[plt.Figure]:
        """
        Plots weight value histograms, mean, variance, and sparsity metrics for all conv layers.
        """
        n_layers = len(self.conv_weights)
        cols = 3
        rows = int(np.ceil(n_layers / cols))

        fig, axes = plt.subplots(
            rows, cols, figsize=(4.5 * cols, 3.2 * rows), squeeze=False, constrained_layout=True
        )

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for idx, (name, w) in enumerate(self.conv_weights.items()):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]

            flat_w = w.flatten()
            mean = np.mean(flat_w)
            std = np.std(flat_w)
            w_min, w_max = np.min(flat_w), np.max(flat_w)
            near_zero = np.mean(np.abs(flat_w) < 0.01) * 100

            color = colors[idx % len(colors)]
            counts, bins, _ = ax.hist(
                flat_w, bins=60, density=True, color=color, alpha=0.65, edgecolor="black", linewidth=0.3
            )

            # Gaussian approximation curve
            x_vals = np.linspace(w_min, w_max, 200)
            if std > 1e-6:
                gaussian = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mean) / std) ** 2)
                ax.plot(x_vals, gaussian, "r--", linewidth=1.5, label="Normal fit")

            ax.axvline(0, color="gray", linestyle=":", alpha=0.8)
            ax.axvline(mean, color="black", linestyle="-", linewidth=1.2, label=f"Mean: {mean:.3f}")

            stats_text = (
                f"Std: {std:.4f}\n"
                f"Range: [{w_min:.2f}, {w_max:.2f}]\n"
                f"Near Zero (<0.01): {near_zero:.1f}%\n"
                f"Params: {len(flat_w):,}"
            )
            ax.text(
                0.96,
                0.94,
                stats_text,
                transform=ax.transAxes,
                fontsize=7.5,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#ccc"),
            )

            ax.set_title(f"{name}\nShape: {w.shape}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Weight Value", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.grid(axis="y", linestyle=":", alpha=0.5)

        # Hide extra subplots
        for idx in range(n_layers, rows * cols):
            r = idx // cols
            c = idx % cols
            axes[r, c].axis("off")

        fig.suptitle(f"Kernel Weight Distributions & Regularization: {self.model_name}", fontsize=13, fontweight="bold")

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"  [Saved] Weight Distributions -> {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_feature_maps(
        self,
        image_path: str,
        save_path: Optional[str] = None,
        show: bool = False,
        dpi: int = 300,
    ) -> Optional[plt.Figure]:
        """
        Feeds an actual image into the convolutional layers and visualizes the resulting
        feature map activations (how the filters actually 'see' the world).
        Shows original image, Conv1 feature maps, and deeper layer activations.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Input image not found at '{image_path}'")

        # Load and preprocess image
        pil_img = Image.open(image_path).convert("RGB")
        resized_img = pil_img.resize((49, 49), Image.BILINEAR)
        t_img = transforms.ToTensor()(resized_img).unsqueeze(0)  # (1, 3, 49, 49)

        # Run sequential convolutions across available conv layers
        current = t_img
        conv_outputs = []
        layer_names = []

        # Find ordered conv weights and biases in state_dict
        conv_keys = [k for k in self.state_dict.keys() if "weight" in k and self.state_dict[k].ndim == 4]

        with torch.no_grad():
            for key in conv_keys:
                w = self.state_dict[key].float()
                # Check for corresponding bias
                b_key = key.replace(".weight", ".bias")
                b = self.state_dict[b_key].float() if b_key in self.state_dict else None

                # Check channel alignment
                if current.shape[1] != w.shape[1]:
                    continue

                # Convolution + activation
                act = F.conv2d(current, w, b, padding=1)
                act = F.gelu(act)
                conv_outputs.append(act.squeeze(0).cpu().numpy())
                layer_names.append(key)

                # Downsample (MaxPool2d 2x2)
                current = F.max_pool2d(act, kernel_size=2, stride=2)

        if not conv_outputs:
            print("[Warning] Could not extract feature maps from checkpoint.")
            return None

        fig = plt.figure(figsize=(16, 8), constrained_layout=True)
        gs = GridSpec(1, 3, width_ratios=[1.2, 3, 3], figure=fig)

        # 1. Original Image
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(resized_img)
        ax_orig.set_title(
            f"Input Traffic Sign\n({os.path.basename(image_path)})\n49x49 RGB",
            fontsize=11,
            fontweight="bold",
        )
        ax_orig.axis("off")

        # 2. Conv 1 Feature Maps (32 maps in 8x4 grid)
        c1 = conv_outputs[0]  # (32, 49, 49)
        n_maps1 = min(32, c1.shape[0])
        sub_gs1 = gs[0, 1].subgridspec(8, 4, wspace=0.1, hspace=0.2)
        for i in range(n_maps1):
            ax = fig.add_subplot(sub_gs1[i // 4, i % 4])
            ax.imshow(c1[i], cmap="viridis")
            ax.set_title(f"F#{i}", fontsize=7)
            ax.axis("off")

        # 3. Conv 2 Feature Maps (32 maps in 8x4 grid)
        if len(conv_outputs) > 1:
            c2 = conv_outputs[1]  # (64, 24, 24)
            n_maps2 = min(32, c2.shape[0])
            sub_gs2 = gs[0, 2].subgridspec(8, 4, wspace=0.1, hspace=0.2)
            for i in range(n_maps2):
                ax = fig.add_subplot(sub_gs2[i // 4, i % 4])
                ax.imshow(c2[i], cmap="inferno")
                ax.set_title(f"F#{i}", fontsize=7)
                ax.axis("off")

        fig.suptitle(
            f"Feature Map Activations (What the Filters Actually Detect): {self.model_name}",
            fontsize=13,
            fontweight="bold",
        )

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"  [Saved] Feature Map Activations -> {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def generate_full_report(
        self,
        output_dir: str = "output/kernel_visualizations",
        file_format: str = "png",
        dpi: int = 300,
        show: bool = False,
    ) -> List[str]:
        """
        Runs the full visualization suite for this checkpoint and saves all artifacts.
        Returns list of saved file paths.
        """
        model_subfolder = os.path.join(output_dir, self.model_name.replace(" ", "_").lower())
        os.makedirs(model_subfolder, exist_ok=True)
        saved_files = []

        print(f"\n[KernelVisualizer] Generating visualizations for model '{self.model_name}'...")
        self.print_summary()

        # 1. RGB Filters (Conv1)
        rgb_candidates = [k for k, v in self.conv_weights.items() if v.shape[1] == 3]
        if rgb_candidates:
            conv1_name = rgb_candidates[0]
            rgb_path = os.path.join(model_subfolder, f"01_conv1_rgb_filters.{file_format}")
            self.plot_conv1_rgb(layer_name=conv1_name, save_path=rgb_path, show=show, dpi=dpi)
            saved_files.append(rgb_path)

        # 2. Kernel Slices for multi-channel conv layers
        for idx, (name, w) in enumerate(self.conv_weights.items()):
            clean_name = name.replace(".", "_")
            slices_path = os.path.join(model_subfolder, f"02_slices_{clean_name}.{file_format}")
            self.plot_layer_kernel_slices(layer_name=name, save_path=slices_path, show=show, dpi=dpi)
            saved_files.append(slices_path)

            # Detailed single filter breakdown for deeper layers
            if w.shape[1] > 3:
                single_filter_path = os.path.join(model_subfolder, f"03_filter0_channels_{clean_name}.{file_format}")
                self.plot_single_filter_channels(layer_name=name, filter_idx=0, save_path=single_filter_path, show=show, dpi=dpi)
                saved_files.append(single_filter_path)

        # 3. Channel Connectivity
        conn_path = os.path.join(model_subfolder, f"04_channel_connectivity.{file_format}")
        self.plot_channel_connectivity(save_path=conn_path, show=show, dpi=dpi)
        saved_files.append(conn_path)

        # 4. Filter Energy Profiles
        energy_path = os.path.join(model_subfolder, f"05_filter_energy.{file_format}")
        self.plot_filter_energy(save_path=energy_path, show=show, dpi=dpi)
        saved_files.append(energy_path)

        # 5. Weight Distributions
        dist_path = os.path.join(model_subfolder, f"06_weight_distributions.{file_format}")
        self.plot_weight_distributions(save_path=dist_path, show=show, dpi=dpi)
        saved_files.append(dist_path)

        # 6. Feature Map Activations on Sample Image
        sample_img = None
        for cand in ["data/Test/00000.png", "data/Test/00001.png"]:
            if os.path.isfile(cand):
                sample_img = cand
                break
        if sample_img is None:
            found_test = glob.glob("data/Test/*.png") + glob.glob("data/**/*.png", recursive=True)
            if found_test:
                sample_img = found_test[0]

        if sample_img:
            feat_path = os.path.join(model_subfolder, f"07_feature_maps_sample.{file_format}")
            try:
                self.plot_feature_maps(image_path=sample_img, save_path=feat_path, show=show, dpi=dpi)
                saved_files.append(feat_path)
            except Exception as e:
                print(f"[Warning] Failed to generate feature maps: {e}")

        print(f"[KernelVisualizer] Successfully generated {len(saved_files)} visual artifacts in: '{model_subfolder}'\n")
        return saved_files


def compare_models(
    ckpt_path1: str,
    ckpt_path2: str,
    label1: Optional[str] = None,
    label2: Optional[str] = None,
    output_dir: str = "output/kernel_visualizations/comparison",
    file_format: str = "png",
    dpi: int = 300,
    show: bool = False,
) -> List[str]:
    """
    Performs comparative kernel analysis between two checkpoints (e.g. Baseline vs AdaBN).
    Compares:
    - First layer RGB filters side-by-side.
    - Weight distributions across corresponding layers.
    - Layer-level filter energy profiles.
    - Cosine similarity between corresponding filter weights.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    vis1 = KernelVisualizer(ckpt_path1, model_name=label1)
    vis2 = KernelVisualizer(ckpt_path2, model_name=label2)
    l1 = vis1.model_name
    l2 = vis2.model_name

    print(f"\n[Comparison] Comparing Kernels: '{l1}' vs '{l2}'...")

    # 1. Compare First Layer RGB Filters Side-by-Side
    rgb1_keys = [k for k, v in vis1.conv_weights.items() if v.shape[1] == 3]
    rgb2_keys = [k for k, v in vis2.conv_weights.items() if v.shape[1] == 3]

    if rgb1_keys and rgb2_keys:
        w1 = vis1.conv_weights[rgb1_keys[0]]  # (N1, 3, H, W)
        w2 = vis2.conv_weights[rgb2_keys[0]]  # (N2, 3, H, W)
        n_show = min(16, w1.shape[0], w2.shape[0])

        fig, axes = plt.subplots(4, 8, figsize=(14, 7), constrained_layout=True)
        # Rows 0-1: Model 1
        # Rows 2-3: Model 2
        for i in range(n_show):
            # Model 1
            r1 = i // 8
            c1 = i % 8
            ax1 = axes[r1, c1]
            f1 = np.transpose(w1[i], (1, 2, 0))
            f1_norm = (f1 - f1.min()) / (f1.max() - f1.min() + 1e-8)
            ax1.imshow(f1_norm, interpolation="nearest")
            ax1.set_title(f"F#{i} ({l1[:8]})", fontsize=7)
            ax1.axis("off")

            # Model 2
            r2 = 2 + (i // 8)
            c2 = i % 8
            ax2 = axes[r2, c2]
            f2 = np.transpose(w2[i], (1, 2, 0))
            f2_norm = (f2 - f2.min()) / (f2.max() - f2.min() + 1e-8)
            ax2.imshow(f2_norm, interpolation="nearest")
            ax2.set_title(f"F#{i} ({l2[:8]})", fontsize=7)
            ax2.axis("off")

        fig.suptitle(f"RGB Filter Comparison (First 16 Filters): {l1} (Top) vs {l2} (Bottom)", fontsize=12, fontweight="bold")
        rgb_cmp_path = os.path.join(output_dir, f"compare_rgb_filters.{file_format}")
        fig.savefig(rgb_cmp_path, dpi=dpi, bbox_inches="tight")
        saved_files.append(rgb_cmp_path)
        if show:
            plt.show()
        else:
            plt.close(fig)

    # 2. Compare Weight Distributions Side-by-Side
    # Pair layers by index
    layers1 = list(vis1.conv_weights.items())
    layers2 = list(vis2.conv_weights.items())
    n_pairs = min(len(layers1), len(layers2))

    fig, axes = plt.subplots(1, n_pairs, figsize=(5.5 * n_pairs, 4.2), squeeze=False, constrained_layout=True)
    for idx in range(n_pairs):
        name1, t1 = layers1[idx]
        name2, t2 = layers2[idx]
        ax = axes[0, idx]

        f1 = t1.flatten()
        f2 = t2.flatten()

        ax.hist(f1, bins=50, density=True, alpha=0.5, color="#1f77b4", label=f"{l1} (std={f1.std():.3f})")
        ax.hist(f2, bins=50, density=True, alpha=0.5, color="#ff7f0e", label=f"{l2} (std={f2.std():.3f})")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.7)

        ax.set_title(f"Layer {idx+1}: {name1} vs {name2}\nShapes: {t1.shape} vs {t2.shape}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Weight Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(f"Weight Distribution Comparison: {l1} vs {l2}", fontsize=12, fontweight="bold")
    dist_cmp_path = os.path.join(output_dir, f"compare_weight_distributions.{file_format}")
    fig.savefig(dist_cmp_path, dpi=dpi, bbox_inches="tight")
    saved_files.append(dist_cmp_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    # 3. Filter Cosine Similarity Matrix (Layer 1)
    if rgb1_keys and rgb2_keys:
        w1_flat = w1.reshape(w1.shape[0], -1)  # (N1, D)
        w2_flat = w2.reshape(w2.shape[0], -1)  # (N2, D)
        if w1_flat.shape[1] == w2_flat.shape[1]:
            norm1 = np.linalg.norm(w1_flat, axis=1, keepdims=True) + 1e-8
            norm2 = np.linalg.norm(w2_flat, axis=1, keepdims=True) + 1e-8
            w1_normed = w1_flat / norm1
            w2_normed = w2_flat / norm2
            # Cosine similarity matrix: (N1, N2)
            sim_matrix = np.matmul(w1_normed, w2_normed.T)

            fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
            im = ax.imshow(sim_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
            ax.set_xlabel(f"{l2} Filter Index", fontsize=9)
            ax.set_ylabel(f"{l1} Filter Index", fontsize=9)
            ax.set_title(f"Filter Cosine Similarity Matrix (Conv1)\n{l1} vs {l2}", fontsize=11, fontweight="bold")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Cosine Similarity (-1 to +1)", fontsize=8)

            sim_path = os.path.join(output_dir, f"compare_cosine_similarity.{file_format}")
            fig.savefig(sim_path, dpi=dpi, bbox_inches="tight")
            saved_files.append(sim_path)
            if show:
                plt.show()
            else:
                plt.close(fig)

    print(f"[Comparison] Comparative visual analysis saved to: '{output_dir}'")
    return saved_files


def visualize_all(
    root_dir: str = "output",
    output_dir: str = "output/kernel_visualizations",
    file_format: str = "png",
    dpi: int = 300,
    show: bool = False,
):
    """
    Discovers all .pth files in root_dir, runs full visualization for each,
    and if multiple models are detected, performs comparative kernel analysis.
    """
    checkpoints = find_checkpoints(root_dir)
    if not checkpoints:
        print(f"[KernelVisualizer] No .pth checkpoint files found under '{root_dir}'.")
        return

    print(f"\n[KernelVisualizer] Found {len(checkpoints)} checkpoint file(s) in '{root_dir}':")
    for cp in checkpoints:
        print(f" - {os.path.relpath(cp, os.getcwd())}")

    analyzed = []
    for cp in checkpoints:
        try:
            vis = KernelVisualizer(cp)
            vis.generate_full_report(output_dir=output_dir, file_format=file_format, dpi=dpi, show=show)
            analyzed.append((cp, vis.model_name))
        except Exception as e:
            print(f"[Error] Failed to visualize '{cp}': {e}")

    # If we have at least 2 checkpoints, compare the first two
    if len(analyzed) >= 2:
        ckpt1, name1 = analyzed[0]
        ckpt2, name2 = analyzed[1]
        compare_models(
            ckpt_path1=ckpt1,
            ckpt_path2=ckpt2,
            label1=name1,
            label2=name2,
            output_dir=os.path.join(output_dir, "comparison"),
            file_format=file_format,
            dpi=dpi,
            show=show,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and inspect convolutional kernels in PyTorch .pth model checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Path to a specific .pth checkpoint file. If omitted, scans --dir.",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default="output",
        help="Root directory to search for .pth files (default: 'output').",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("CKPT1", "CKPT2"),
        help="Compare two checkpoints side-by-side (e.g. output/baseline/baseline_model.pth output/adabn/adabn_model.pth).",
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        metavar=("LABEL1", "LABEL2"),
        help="Labels for compared checkpoints (e.g. Baseline AdaBN).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output/kernel_visualizations",
        help="Directory to save generated visualization figures (default: 'output/kernel_visualizations').",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf"],
        help="Image format to save ('png' or 'pdf', default: 'png').",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution in DPI (default: 300).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively with matplotlib GUI popup window.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an input image (e.g. traffic sign) to visualize live feature map activations.",
    )
    parser.add_argument(
        "--noises",
        action="store_true",
        help="Apply all 14 benchmark noise corruptions to --image and generate feature map degradation matrix.",
    )
    parser.add_argument(
        "--severity",
        "-s",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Corruption severity level (1 to 5, default: 3).",
    )

    args = parser.parse_args()

    # Mode 0A: Visualize all 14 noise corruptions on a single image
    if args.noises:
        from visualize_noise_feature_maps import generate_noise_feature_maps_matrix
        img_target = args.image or "data/Test/00001.png"
        ckpt_path = args.checkpoint or "output/baseline/baseline_model.pth"
        generate_noise_feature_maps_matrix(
            image_path=img_target,
            checkpoint_path=ckpt_path,
            severity=args.severity,
            output_dir=os.path.join(args.output_dir, "noise_matrix"),
            dpi=args.dpi,
            show=args.show,
        )
        return

    # Mode 0B: Visualize feature map activations for a given clean input image
    if args.image:
        ckpt_path = args.checkpoint
        if not ckpt_path:
            discovered = find_checkpoints(args.dir)
            if not discovered:
                raise FileNotFoundError(f"No checkpoint found in '{args.dir}' to run feature maps.")
            ckpt_path = discovered[0]
        vis = KernelVisualizer(ckpt_path)
        img_stem = os.path.splitext(os.path.basename(args.image))[0]
        out_name = f"feature_maps_{img_stem}.{args.format}"
        save_dest = os.path.join(args.output_dir, vis.model_name.replace(" ", "_").lower(), out_name)
        vis.plot_feature_maps(image_path=args.image, save_path=save_dest, show=args.show, dpi=args.dpi)
        return

    # Mode 1: Compare two specific checkpoints
    if args.compare:
        labels = args.labels if args.labels else (None, None)
        compare_models(
            ckpt_path1=args.compare[0],
            ckpt_path2=args.compare[1],
            label1=labels[0],
            label2=labels[1],
            output_dir=os.path.join(args.output_dir, "comparison"),
            file_format=args.format,
            dpi=args.dpi,
            show=args.show,
        )
    # Mode 2: Single checkpoint specified
    elif args.checkpoint:
        vis = KernelVisualizer(args.checkpoint)
        vis.generate_full_report(
            output_dir=args.output_dir, file_format=args.format, dpi=args.dpi, show=args.show
        )
    # Mode 3: Automatic discovery in directory
    else:
        visualize_all(
            root_dir=args.dir,
            output_dir=args.output_dir,
            file_format=args.format,
            dpi=args.dpi,
            show=args.show,
        )


if __name__ == "__main__":
    main()
