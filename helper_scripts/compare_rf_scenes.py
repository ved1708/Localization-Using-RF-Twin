"""
Compare RF Heatmaps: Static vs Dynamic Scene
============================================

This script compares RF propagation patterns between:
- Static scene (without metallic cube)
- Dynamic scene (with metallic cube)

The difference highlights where the metallic obstacle affects RF propagation,
enabling localization of the dynamic object.

Author: RF-3DGS Analysis Tool
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import json

def load_rf_image(image_path):
    """
    Load RF heatmap image and convert to grayscale intensity.
    
    Args:
        image_path: Path to PNG image
        
    Returns:
        rf_data: 2D numpy array of RF intensities (0-1 normalized)
    """
    # Load image (already grayscale/single channel from RF render)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # If RGB, convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalize to 0-1
    rf_data = img.astype(np.float32) / 255.0
    
    return rf_data

def compute_difference_map(static_rf, dynamic_rf):
    """
    Compute difference between dynamic and static RF fields.
    
    Args:
        static_rf: RF heatmap without obstacle
        dynamic_rf: RF heatmap with metallic cube
        
    Returns:
        diff_map: Absolute difference |dynamic - static|
        signed_diff: Signed difference (dynamic - static)
    """
    # Ensure same dimensions
    if static_rf.shape != dynamic_rf.shape:
        raise ValueError(f"Shape mismatch: static {static_rf.shape} vs dynamic {dynamic_rf.shape}")
    
    # Compute signed difference (positive = RF increased, negative = RF decreased)
    signed_diff = dynamic_rf - static_rf
    
    # Absolute difference for change magnitude
    diff_map = np.abs(signed_diff)
    
    return diff_map, signed_diff

def visualize_comparison(static_rf, dynamic_rf, diff_map, signed_diff, 
                        output_path, rx_position, threshold=0.1):
    """
    Create multi-panel visualization comparing static, dynamic, and difference.
    
    Args:
        static_rf: Static scene RF heatmap
        dynamic_rf: Dynamic scene RF heatmap  
        diff_map: Absolute difference map
        signed_diff: Signed difference (shows increase/decrease)
        output_path: Where to save visualization
        rx_position: Rx antenna position for title
        threshold: Highlight differences above this threshold
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'RF Scene Comparison - Rx Position: {rx_position}', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Static, Dynamic, Absolute Difference
    # Static scene
    im0 = axes[0, 0].imshow(static_rf, cmap='viridis', vmin=0, vmax=1)
    axes[0, 0].set_title('Static Scene (No Cube)', fontsize=14)
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label='RF Intensity')
    
    # Dynamic scene
    im1 = axes[0, 1].imshow(dynamic_rf, cmap='viridis', vmin=0, vmax=1)
    axes[0, 1].set_title('Dynamic Scene (With Metallic Cube)', fontsize=14)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label='RF Intensity')
    
    # Absolute difference
    im2 = axes[0, 2].imshow(diff_map, cmap='hot', vmin=0, vmax=np.max(diff_map))
    axes[0, 2].set_title('Absolute Difference |Dynamic - Static|', fontsize=14)
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04, label='Change Magnitude')
    
    # Row 2: Signed Difference, Thresholded Change Regions, Statistics
    # Signed difference (RdBu: red = increase, blue = decrease)
    vmax_signed = np.max(np.abs(signed_diff))
    im3 = axes[1, 0].imshow(signed_diff, cmap='RdBu_r', vmin=-vmax_signed, vmax=vmax_signed)
    axes[1, 0].set_title('Signed Difference (Red=Increase, Blue=Decrease)', fontsize=14)
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04, label='RF Change')
    
    # Thresholded change regions (binary mask of significant changes)
    change_mask = diff_map > threshold
    im4 = axes[1, 1].imshow(change_mask, cmap='gray')
    axes[1, 1].set_title(f'Significant Changes (threshold={threshold:.2f})', fontsize=14)
    axes[1, 1].axis('off')
    
    # Overlay change regions on dynamic image
    overlay = dynamic_rf.copy()
    overlay_rgb = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    overlay_rgb[change_mask] = [255, 0, 0]  # Red highlight for changes
    axes[1, 2].imshow(overlay_rgb)
    axes[1, 2].set_title('Change Localization Overlay', fontsize=14)
    axes[1, 2].axis('off')
    
    # Compute statistics
    mean_diff = np.mean(diff_map)
    max_diff = np.max(diff_map)
    change_percentage = 100 * np.sum(change_mask) / change_mask.size
    
    # Add statistics text
    stats_text = f"""Statistics:
    Mean Difference: {mean_diff:.4f}
    Max Difference: {max_diff:.4f}
    Changed Pixels: {change_percentage:.2f}%
    
    RF Power Change:
    Increased regions: {np.sum(signed_diff > 0)} px
    Decreased regions: {np.sum(signed_diff < 0)} px
    """
    
    fig.text(0.7, 0.15, stats_text, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization: {output_path}")
    print(f"    - Mean difference: {mean_diff:.4f}")
    print(f"    - Max difference: {max_diff:.4f}")
    print(f"    - Changed pixels: {change_percentage:.2f}%")
    
    return {
        'mean_difference': float(mean_diff),
        'max_difference': float(max_diff),
        'change_percentage': float(change_percentage),
        'increased_pixels': int(np.sum(signed_diff > 0)),
        'decreased_pixels': int(np.sum(signed_diff < 0))
    }

def find_change_centroid(diff_map, threshold=0.1):
    """
    Find the centroid (center of mass) of significant changes.
    This can help localize where the dynamic object is affecting RF.
    
    Args:
        diff_map: Difference map
        threshold: Threshold for significant changes
        
    Returns:
        centroid_x, centroid_y: Pixel coordinates of change center
    """
    change_mask = diff_map > threshold
    
    if np.sum(change_mask) == 0:
        return None, None
    
    # Compute weighted centroid
    y_indices, x_indices = np.where(change_mask)
    weights = diff_map[change_mask]
    
    centroid_x = np.average(x_indices, weights=weights)
    centroid_y = np.average(y_indices, weights=weights)
    
    return centroid_x, centroid_y

def main():
    parser = argparse.ArgumentParser(description="Compare static vs dynamic RF scenes")
    parser.add_argument('--static_dir', type=str, required=True,
                       help='Directory with static scene RF images (from trained model)')
    parser.add_argument('--dynamic_dir', type=str, required=True,
                       help='Directory with dynamic scene RF images (ground truth with cube)')
    parser.add_argument('--output_dir', type=str, default='rf_comparison_results',
                       help='Output directory for comparison results')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Threshold for significant RF changes (0-1)')
    parser.add_argument('--rx_ids', type=str, default='1,2',
                       help='Comma-separated Rx IDs to compare (e.g., "1,2")')
    
    args = parser.parse_args()
    
    print("="*70)
    print("RF SCENE COMPARISON: STATIC VS DYNAMIC")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Static RF dir: {args.static_dir}")
    print(f"  Dynamic RF dir: {args.dynamic_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Change threshold: {args.threshold}")
    print(f"  Rx IDs: {args.rx_ids}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse Rx IDs
    rx_ids = [int(x.strip()) for x in args.rx_ids.split(',')]
    
    # Results storage
    all_results = {}
    
    # Process each Rx position
    for rx_id in rx_ids:
        print(f"\n{'='*70}")
        print(f"Processing Rx Position {rx_id}")
        print(f"{'='*70}")
        
        # Find corresponding images
        # Convention: 00001.png for first Rx, 00002.png for second, etc.
        # Adjust based on your naming convention
        static_img_path = os.path.join(args.static_dir, f'{rx_id:05d}.png')
        dynamic_img_path = os.path.join(args.dynamic_dir, f'{rx_id:05d}.png')
        
        # Check if files exist
        if not os.path.exists(static_img_path):
            print(f"  ⚠ Static image not found: {static_img_path}")
            print(f"    Trying alternate naming...")
            # Try to list files and find matching pattern
            static_files = sorted([f for f in os.listdir(args.static_dir) if f.endswith('.png')])
            if rx_id <= len(static_files):
                static_img_path = os.path.join(args.static_dir, static_files[rx_id - 1])
                print(f"    Using: {static_img_path}")
            else:
                print(f"  ✗ Could not find static image for Rx {rx_id}")
                continue
        
        if not os.path.exists(dynamic_img_path):
            print(f"  ⚠ Dynamic image not found: {dynamic_img_path}")
            print(f"    Trying alternate naming...")
            dynamic_files = sorted([f for f in os.listdir(args.dynamic_dir) if f.endswith('.png')])
            if rx_id <= len(dynamic_files):
                dynamic_img_path = os.path.join(args.dynamic_dir, dynamic_files[rx_id - 1])
                print(f"    Using: {dynamic_img_path}")
            else:
                print(f"  ✗ Could not find dynamic image for Rx {rx_id}")
                continue
        
        print(f"  Loading images...")
        print(f"    Static: {os.path.basename(static_img_path)}")
        print(f"    Dynamic: {os.path.basename(dynamic_img_path)}")
        
        # Load RF data
        static_rf = load_rf_image(static_img_path)
        dynamic_rf = load_rf_image(dynamic_img_path)
        
        print(f"  ✓ Loaded RF data: {static_rf.shape}")
        
        # Compute differences
        diff_map, signed_diff = compute_difference_map(static_rf, dynamic_rf)
        
        # Find change centroid
        centroid_x, centroid_y = find_change_centroid(diff_map, args.threshold)
        
        if centroid_x is not None:
            print(f"  Change centroid: ({centroid_x:.1f}, {centroid_y:.1f})")
        else:
            print(f"  No significant changes detected above threshold {args.threshold}")
        
        # Visualize and save
        output_path = os.path.join(args.output_dir, f'comparison_rx{rx_id}.png')
        rx_position = f"Rx {rx_id}"  # Could be enhanced with actual coordinates
        
        stats = visualize_comparison(static_rf, dynamic_rf, diff_map, signed_diff,
                                     output_path, rx_position, args.threshold)
        
        # Add centroid to stats
        stats['centroid_x'] = float(centroid_x) if centroid_x is not None else None
        stats['centroid_y'] = float(centroid_y) if centroid_y is not None else None
        
        all_results[f'rx_{rx_id}'] = stats
    
    # Save summary JSON
    summary_path = os.path.join(args.output_dir, 'comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"COMPARISON COMPLETE")
    print(f"{'='*70}")``
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary JSON: {summary_path}")
    
    # Print overall statistics
    print(f"\nOverall Statistics:")
    for rx_id, stats in all_results.items():
        print(f"  {rx_id}:")
        print(f"    Mean diff: {stats['mean_difference']:.4f}")
        print(f"    Max diff: {stats['max_difference']:.4f}")
        print(f"    Changed: {stats['change_percentage']:.2f}%")
        if stats['centroid_x'] is not None:
            print(f"    Centroid: ({stats['centroid_x']:.1f}, {stats['centroid_y']:.1f})")

if __name__ == "__main__":
    main()
