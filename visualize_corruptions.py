import os
import argparse
import numpy as np

# NumPy 2.0+ compatibility shim for imagecorruptions
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128

import matplotlib.pyplot as plt
from PIL import Image
from imagecorruptions import corrupt

# The 14 corruptions used in GTSRB robustness evaluation
CORRUPTION_TYPES = [
    'gaussian_noise', 'impulse_noise', 'shot_noise', 'snow', 'fog',
    'frost', 'motion_blur', 'zoom_blur', 'defocus_blur', 'brightness',
    'contrast', 'pixelate', 'elastic_transform', 'jpeg_compression'
]

def get_default_sample_image():
    """
    Finds the base image from data/Train/0:
    First image in data/Train/0.
    """
    class_0_dir = os.path.join('data', 'Train', '0')
    if os.path.exists(class_0_dir):
        files = [f for f in sorted(os.listdir(class_0_dir)) if f.lower().endswith(('.png', '.jpg', '.ppm'))]
        if files:
            return os.path.join(class_0_dir, files[0])
            
    # Fallback: scan Train folder for first available image
    train_dir = os.path.join('data', 'Train')
    if os.path.exists(train_dir):
        for root, _, files in os.walk(train_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.ppm')):
                    return os.path.join(root, file)
    raise FileNotFoundError("Could not find any sample image in the data directory.")

def apply_corruption_then_downscale(image_pil, corruption_name, severity, target_size=(49, 49)):
    """
    Applies noise corruption FIRST on native/high-resolution image,
    then downscales the result to target_size.
    Ensures image is at least target_size / 32px for imagecorruptions library.
    """
    w, h = image_pil.size
    if w < target_size[0] or h < target_size[1]:
        img_to_corrupt = image_pil.resize((max(w, target_size[0]), max(h, target_size[1])))
    else:
        img_to_corrupt = image_pil

    img_np = np.array(img_to_corrupt)
    corrupted_np = corrupt(img_np, corruption_name=corruption_name, severity=severity)
    corrupted_pil = Image.fromarray(corrupted_np).resize(target_size)
    return np.array(corrupted_pil)

def plot_single_corruption(image_pil, corruption_name, target_size=(49, 49), output_path=None, show=False):
    """
    Plots the clean image alongside all 5 severity levels for a single corruption.
    Noise is applied FIRST, then downscaled to target_size.
    """
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
    
    # Column 0: Original downscaled to target_size
    clean_np = np.array(image_pil.resize(target_size))
    axes[0].imshow(clean_np, interpolation='nearest')
    axes[0].set_title(f"Clean ({target_size[0]}x{target_size[1]})", fontsize=11, fontweight='bold', pad=8)
    axes[0].axis('off')
    
    # Columns 1 to 5: Severity Levels
    for severity in range(1, 6):
        corrupted_downscaled = apply_corruption_then_downscale(
            image_pil, corruption_name=corruption_name, severity=severity, target_size=target_size
        )
        ax = axes[severity]
        ax.imshow(corrupted_downscaled, interpolation='nearest')
        ax.set_title(f"Severity {severity}", fontsize=11, pad=8)
        ax.axis('off')
        
    title = corruption_name.replace('_', ' ').title()
    fig.suptitle(f"Corruption: {title} (Noise First -> Downscale {target_size[0]}x{target_size[1]})", fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to: {output_path}")
    if show:
        plt.show()
    plt.close()

def plot_all_corruptions(image_pil, target_size=(49, 49), output_path=None, show=False):
    """
    Generates a 14 x 6 comprehensive grid showing all corruptions at all 5 severities.
    """
    num_corruptions = len(CORRUPTION_TYPES)
    fig, axes = plt.subplots(num_corruptions, 6, figsize=(18, 3.2 * num_corruptions))
    clean_np = np.array(image_pil.resize(target_size))
    
    for row_idx, corruption_name in enumerate(CORRUPTION_TYPES):
        row_title = corruption_name.replace('_', ' ').title()
        
        # Column 0: Original
        axes[row_idx, 0].imshow(clean_np, interpolation='nearest')
        if row_idx == 0:
            axes[row_idx, 0].set_title(f"Clean ({target_size[0]}x{target_size[1]})", fontsize=13, fontweight='bold', pad=10)
        
        # Explicit bold row label on the left
        axes[row_idx, 0].text(
            -0.20, 0.5, row_title,
            transform=axes[row_idx, 0].transAxes,
            fontsize=13, fontweight='bold',
            va='center', ha='right', color='#111111'
        )
        axes[row_idx, 0].axis('off')
        
        # Columns 1-5: Severities
        for severity in range(1, 6):
            corrupted_downscaled = apply_corruption_then_downscale(
                image_pil, corruption_name=corruption_name, severity=severity, target_size=target_size
            )
            ax = axes[row_idx, severity]
            ax.imshow(corrupted_downscaled, interpolation='nearest')
            if row_idx == 0:
                ax.set_title(f"Severity {severity}", fontsize=12, fontweight='bold')
            ax.axis('off')

    fig.suptitle(
        f"GTSRB Robustness Benchmark: 14 Noise Types Across 5 Severity Levels (Noise First -> {target_size[0]}x{target_size[1]})",
        fontsize=18,
        fontweight='bold',
        y=1.003
    )
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=120)
        print(f"Saved full corruptions grid to: {output_path}")
    if show:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize image corruption levels (1 to 5) with noise-first downscaling.")
    parser.add_argument('--image_path', type=str, default=None, help="Path to input image (defaults to sample from data/Train/0).")
    parser.add_argument('--corruption', '--noise', dest='corruption', type=str, default='fog', help=f"Corruption name (e.g. 'fog', 'gaussian_noise') or 'all'. Choices: {CORRUPTION_TYPES}")
    parser.add_argument('--resolution', '--resize', dest='resolution', type=int, default=49, help="Target downscaled resolution (default: 49 for 49x49).")
    parser.add_argument('--output', type=str, default=None, help="Optional custom output path. Defaults to output/noise/sample_<noise>_<res>x<res>.png.")
    parser.add_argument('--show', action='store_true', help="Display the plot interactively using plt.show().")
    
    args = parser.parse_args()
    
    # 1. Resolve image path
    image_path = args.image_path if args.image_path else get_default_sample_image()
    print(f"Loading image from: {image_path}")
    pil_image = Image.open(image_path).convert('RGB')
    
    target_size = (args.resolution, args.resolution)
    res_str = f"{args.resolution}x{args.resolution}"
    noise_output_dir = os.path.join('output', 'noise')
    os.makedirs(noise_output_dir, exist_ok=True)
    
    # 2. Render visualization
    if args.corruption.lower() == 'all':
        out_path = args.output if args.output else os.path.join(noise_output_dir, f"sample_all_{res_str}.png")
        plot_all_corruptions(pil_image, target_size=target_size, output_path=out_path, show=args.show)
    else:
        if args.corruption not in CORRUPTION_TYPES:
            print(f"Warning: '{args.corruption}' is not one of the 14 standard benchmark corruptions.")
            print(f"Available standard corruptions: {', '.join(CORRUPTION_TYPES)}")
        out_path = args.output if args.output else os.path.join(noise_output_dir, f"sample_{args.corruption}_{res_str}.png")
        plot_single_corruption(pil_image, args.corruption, target_size=target_size, output_path=out_path, show=args.show)

if __name__ == '__main__':
    main()
