"""
Simple comparison of static vs dynamic RF scenes.
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(path):
    """Load image and convert to float32."""
    img = Image.open(path)
    return np.array(img, dtype=np.float32) / 255.0


def compute_difference(static_img, dynamic_img):
    """Compute absolute and relative differences."""
    # Convert to grayscale (average of RGB channels)
    if static_img.ndim == 3:
        static_gray = static_img.mean(axis=2)
    else:
        static_gray = static_img
    
    if dynamic_img.ndim == 3:
        dynamic_gray = dynamic_img.mean(axis=2)
    else:
        dynamic_gray = dynamic_img
    
    # Compute differences
    abs_diff = np.abs(dynamic_gray - static_gray)
    
    return static_gray, dynamic_gray, abs_diff


def compute_statistics(static_gray, dynamic_gray, abs_diff):
    """Compute statistics of the comparison."""
    stats = {
        'static_mean': float(static_gray.mean()),
        'static_std': float(static_gray.std()),
        'static_min': float(static_gray.min()),
        'static_max': float(static_gray.max()),
        'dynamic_mean': float(dynamic_gray.mean()),
        'dynamic_std': float(dynamic_gray.std()),
        'dynamic_min': float(dynamic_gray.min()),
        'dynamic_max': float(dynamic_gray.max()),
        'diff_mean': float(abs_diff.mean()),
        'diff_std': float(abs_diff.std()),
        'diff_max': float(abs_diff.max()),
        'diff_percentile_95': float(np.percentile(abs_diff, 95)),
        'diff_percentile_99': float(np.percentile(abs_diff, 99)),
    }
    return stats


def create_comparison_plot(static_gray, dynamic_gray, abs_diff, rx_id, output_path):
    """Create comparison visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Static scene
    im0 = axes[0].imshow(static_gray, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title(f'Static Scene (No Box) - Rx {rx_id}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Dynamic scene
    im1 = axes[1].imshow(dynamic_gray, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title(f'Dynamic Scene (With Box) - Rx {rx_id}', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Difference
    im2 = axes[2].imshow(abs_diff, cmap='hot', vmin=0, vmax=abs_diff.max())
    axes[2].set_title(f'Absolute Difference - Rx {rx_id}', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare static vs dynamic RF scenes")
    parser.add_argument("--static1", type=str, required=True, help="Static RF image for Rx 1")
    parser.add_argument("--static2", type=str, required=True, help="Static RF image for Rx 2")
    parser.add_argument("--dynamic1", type=str, required=True, help="Dynamic RF image for Rx 1")
    parser.add_argument("--dynamic2", type=str, required=True, help="Dynamic RF image for Rx 2")
    parser.add_argument("--output_dir", type=str, default="rf_comparison_results", 
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process both Rx positions
    for rx_id, static_path, dynamic_path in [
        (1, args.static1, args.dynamic1),
        (2, args.static2, args.dynamic2)
    ]:
        print(f"\n{'='*60}")
        print(f"Processing Rx Position {rx_id}")
        print(f"{'='*60}")
        print(f"Static:  {static_path}")
        print(f"Dynamic: {dynamic_path}")
        
        # Load images
        static_img = load_image(static_path)
        dynamic_img = load_image(dynamic_path)
        print(f"Image shape: {static_img.shape}")
        
        # Compute difference
        static_gray, dynamic_gray, abs_diff = compute_difference(static_img, dynamic_img)
        
        # Compute statistics
        stats = compute_statistics(static_gray, dynamic_gray, abs_diff)
        
        print(f"\nStatistics:")
        print(f"  Static  - Mean: {stats['static_mean']:.4f}, Std: {stats['static_std']:.4f}, "
              f"Range: [{stats['static_min']:.4f}, {stats['static_max']:.4f}]")
        print(f"  Dynamic - Mean: {stats['dynamic_mean']:.4f}, Std: {stats['dynamic_std']:.4f}, "
              f"Range: [{stats['dynamic_min']:.4f}, {stats['dynamic_max']:.4f}]")
        print(f"  Diff    - Mean: {stats['diff_mean']:.4f}, Std: {stats['diff_std']:.4f}, "
              f"Max: {stats['diff_max']:.4f}")
        print(f"            95th percentile: {stats['diff_percentile_95']:.4f}")
        print(f"            99th percentile: {stats['diff_percentile_99']:.4f}")
        
        # Save statistics to JSON
        import json
        stats_path = os.path.join(args.output_dir, f"statistics_rx{rx_id}.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved statistics: {stats_path}")
        
        # Create comparison plot
        plot_path = os.path.join(args.output_dir, f"comparison_rx{rx_id}.png")
        create_comparison_plot(static_gray, dynamic_gray, abs_diff, rx_id, plot_path)
        
        # Save difference map as separate image
        diff_path = os.path.join(args.output_dir, f"difference_rx{rx_id}.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(abs_diff, cmap='hot')
        plt.colorbar(label='Absolute Difference')
        plt.title(f'RF Difference Map - Rx {rx_id}\n(Shows metallic box effect)', 
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.savefig(diff_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved difference map: {diff_path}")
    
    print(f"\n{'='*60}")
    print(f"✓ Comparison complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
