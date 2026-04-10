#!/usr/bin/env python3
"""
Extract RF heatmaps from a trained RF-3DGS model using proper Gaussian Splatting projection.

This script implements the step-by-step process:
1. Load PLY file containing 3D Gaussians (positions, opacities, colors, scales, rotations)
2. Define camera parameters from the RF dataset (position, orientation, FOV)
3. Project 3D Gaussians to 2D image plane (frustum culling + perspective projection)
4. Accumulate opacity/intensity to render the RF power heatmap

The camera parameters are extracted from your generate_single_rf_dynamic.py:
- Rx positions: [6.7, 2.5, 1.5] and [0.3, 2.2, 1.5]
- Looking at target: [3.5, 2.5, 1.2] (box location)
- FOV: 120° horizontal
- Resolution: 1024x1024

Usage:
    python extract_rf_from_ply.py \
        --ply_path /path/to/RF-3DGS/output/rf_model/point_cloud/iteration_45000/point_cloud.ply \
        --dataset /path/to/dataset_custom_scene_ideal_mpc \
        --output_dir /path/to/output/extracted_rf \
        --views all  # or 'train', 'test', or comma-separated indices like '0,5,10'
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add RF-3DGS to path
rf3dgs_path = Path(__file__).parent.parent / "RF-3DGS"
sys.path.insert(0, str(rf3dgs_path))

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from utils.general_utils import safe_state


def load_gaussians_from_ply(ply_path):
    """
    Step 1: Load the PLY file.
    
    Reads the trained Gaussian Splatting model which contains:
    - Positions (x, y, z): 3D locations of Gaussians
    - Opacities (alpha): Transparency/density values (stored as inverse sigmoid)
    - Colors (f_dc, f_rest): Spherical harmonics coefficients (for RF, repurposed as RF power)
    - Scales: Size of each Gaussian ellipsoid
    - Rotations: Orientation quaternions
    
    Returns:
        gaussians: GaussianModel object containing all parameters
    """
    print(f"Loading Gaussians from {ply_path}")
    
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)
    
    num_gaussians = gaussians.get_xyz.shape[0]
    print(f"  ✓ Loaded {num_gaussians} Gaussians")
    print(f"  ✓ Position range: X=[{gaussians.get_xyz[:, 0].min():.2f}, {gaussians.get_xyz[:, 0].max():.2f}], "
          f"Y=[{gaussians.get_xyz[:, 1].min():.2f}, {gaussians.get_xyz[:, 1].max():.2f}], "
          f"Z=[{gaussians.get_xyz[:, 2].min():.2f}, {gaussians.get_xyz[:, 2].max():.2f}]")
    print(f"  ✓ Opacity range: [{gaussians.get_opacity.min():.4f}, {gaussians.get_opacity.max():.4f}]")
    
    return gaussians


def setup_cameras(dataset_path, images_folder="spectrum"):
    """
    Step 2: Define the camera (the sensor).
    
    Loads camera metadata from the RF dataset which includes:
    - Position: Rx location (e.g., [6.7, 2.5, 1.5])
    - Orientation: Look-at direction (computed from yaw/pitch to target [3.5, 2.5, 1.2])
    - FOV: Field of view (120° horizontal from generate_single_rf_dynamic.py)
    - Intrinsics: Focal length, principal point from cameras.txt
    
    The camera parameters match exactly what was used during RF data generation,
    ensuring proper alignment between rendered and ground-truth heatmaps.
    
    Returns:
        train_cameras: List of Camera objects for training views
        test_cameras: List of Camera objects for test views
    """
    print(f"\nStep 2: Loading camera parameters from {dataset_path}")
    
    # Create a minimal argument namespace to load the scene
    class Args:
        def __init__(self):
            self.source_path = dataset_path
            self.model_path = ""  # Not needed for loading cameras
            self.images = images_folder
            self.resolution = -1
            self.white_background = False
            self.data_device = "cuda"
            self.eval = False
            self.sh_degree = 3
    
    args = Args()
    
    # Load scene to get cameras
    gaussians_dummy = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians_dummy, load_iteration=None, shuffle=False)
    
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    
    print(f"  ✓ Found {len(train_cameras)} training cameras")
    print(f"  ✓ Found {len(test_cameras)} test cameras")
    
    # Display camera info for first training camera
    if train_cameras:
        cam = train_cameras[0]
        print(f"\n  Camera 0 details:")
        print(f"    - Image name: {cam.image_name}")
        print(f"    - Resolution: {cam.image_width} x {cam.image_height}")
        print(f"    - FOV: H={np.degrees(cam.FoVx):.1f}°, V={np.degrees(cam.FoVy):.1f}°")
        # Camera position in world coordinates (from transformation matrix)
        R = cam.R
        T = cam.T
        # Camera center in world coords: C = -R^T * T
        camera_center = -R.T @ T
        print(f"    - Position (world): [{camera_center[0]:.2f}, {camera_center[1]:.2f}, {camera_center[2]:.2f}]")
    
    return train_cameras, test_cameras


def extract_rf_heatmap(gaussians, camera, bg_color):
    """
    Steps 3 & 4: Project Gaussians to 2D and accumulate opacity.
    
    This function performs the core Gaussian Splatting rendering:
    
    Step 3 - Project Points to 2D (Splatting):
        For each Gaussian:
        - Check if inside camera frustum (frustum culling)
        - Transform 3D position (x,y,z) to camera space
        - Project to 2D image plane (u,v) using camera intrinsics
        - Compute 2D Gaussian footprint (ellipse) on image plane
    
    Step 4 - Accumulate Opacity (The "RF Image"):
        Unlike photos where solid objects occlude, RF waves penetrate.
        For each pixel (u,v):
        - Sum contributions from all Gaussians projecting to that pixel
        - Weight by opacity (α) and 2D Gaussian kernel G(u,v)
        - Formula: Pixel_RF(u,v) = Σ α_i · G(u,v) · color_i
        
        For RF data:
        - color_i represents RF power (learned during training)
        - α_i represents signal presence/density
        - Penetration is modeled by alpha-blending (back-to-front compositing)
    
    Args:
        gaussians: Trained GaussianModel with 3D scene representation
        camera: Camera object with pose, intrinsics, and image dimensions
        bg_color: Background color (black [0,0,0] for RF = no signal)
    
    Returns:
        rf_heatmap: (H, W, C) numpy array with RF power values
    """
    # Create pipeline parameters (use defaults from training)
    class PipeArgs:
        def __init__(self):
            self.convert_SHs_python = False  # Use CUDA kernels for speed
            self.compute_cov3D_python = False  # Use CUDA kernels
            self.debug = False
    
    pipe_args = PipeArgs()
    
    # Render the scene from this camera view
    # The 'render' function internally performs:
    #   1. Frustum culling (discard Gaussians outside view)
    #   2. Sort Gaussians by depth (for proper alpha blending)
    #   3. Project each Gaussian to 2D
    #   4. Rasterize with alpha blending (accumulate weighted contributions)
    with torch.no_grad():
        rendering = render(camera, gaussians, pipe_args, bg_color)
    
    # Extract the rendered RF heatmap
    # Shape: (C, H, W) where C is typically 3 (RGB channels)
    # For RF, all 3 channels should be identical (grayscale RF power)
    rf_heatmap = rendering["render"]
    
    # Convert to numpy and transpose to (H, W, C)
    rf_heatmap_np = rf_heatmap.cpu().numpy().transpose(1, 2, 0)
    
    # Clamp to valid range [0, 1]
    rf_heatmap_np = np.clip(rf_heatmap_np, 0.0, 1.0)
    
    return rf_heatmap_np


def save_heatmap(heatmap, output_path, colormap='viridis'):
    """Save RF heatmap as PNG image with optional colormap."""
    # Convert to 8-bit
    heatmap_uint8 = (heatmap * 255.0).astype(np.uint8)
    
    # If single channel, convert to grayscale
    if heatmap_uint8.shape[-1] == 1:
        heatmap_uint8 = heatmap_uint8.squeeze(-1)
    
    # If 3-channel, just save as RGB
    img = Image.fromarray(heatmap_uint8)
    img.save(output_path)


def parse_view_selection(view_str, num_train, num_test):
    """Parse view selection string and return indices."""
    if view_str == "all":
        return list(range(num_train)), list(range(num_test))
    elif view_str == "train":
        return list(range(num_train)), []
    elif view_str == "test":
        return [], list(range(num_test))
    else:
        # Parse comma-separated indices
        indices = [int(x.strip()) for x in view_str.split(",")]
        # Assume they're train indices
        return indices, []


def main():
    parser = argparse.ArgumentParser(description="Extract RF heatmaps from trained RF-3DGS model")
    parser.add_argument("--ply_path", required=True, 
                       help="Path to trained PLY file (e.g., /path/to/rf_model/point_cloud/iteration_45000/point_cloud.ply)")
    parser.add_argument("--dataset", required=True, 
                       help="Path to RF dataset with camera metadata (e.g., dataset_custom_scene_ideal_mpc)")
    parser.add_argument("--output_dir", required=True, 
                       help="Directory to save extracted RF heatmaps")
    parser.add_argument("--views", default="all", 
                       help="Views to extract: 'all', 'train', 'test', or comma-separated indices")
    parser.add_argument("--images_folder", default="spectrum", 
                       help="Folder name containing RF images in dataset")
    args = parser.parse_args()
    
    print("="*70)
    print("RF HEATMAP EXTRACTION FROM TRAINED 3DGS MODEL")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  PLY file: {args.ply_path}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {args.output_dir}")
    print(f"  Views: {args.views}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize CUDA
    print(f"\nInitializing CUDA...")
    safe_state(silent=True)
    
    # STEP 1: Load Gaussians from PLY
    print("\n" + "="*70)
    print("STEP 1: LOAD 3D GAUSSIANS FROM PLY")
    print("="*70)
    gaussians = load_gaussians_from_ply(args.ply_path)
    
    # STEP 2: Load Cameras
    print("\n" + "="*70)
    print("STEP 2: DEFINE CAMERA (THE SENSOR)")
    print("="*70)
    train_cameras, test_cameras = setup_cameras(args.dataset, args.images_folder)
    
    # Parse view selection
    train_indices, test_indices = parse_view_selection(args.views, len(train_cameras), len(test_cameras))
    
    # Background color (black for RF - represents no signal)
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # STEPS 3 & 4: Extract RF heatmaps
    print("\n" + "="*70)
    print("STEPS 3 & 4: PROJECT GAUSSIANS AND ACCUMULATE RF POWER")
    print("="*70)
    print("\nRendering RF heatmaps from trained Gaussians...")
    print("(Frustum culling → 2D projection → Alpha blending)\n")
    
    # Process training views
    if train_indices:
        train_output_dir = os.path.join(args.output_dir, "train")
        os.makedirs(train_output_dir, exist_ok=True)
        
        print(f"Processing {len(train_indices)} training views...")
        for idx in tqdm(train_indices, desc="Training views"):
            camera = train_cameras[idx]
            heatmap = extract_rf_heatmap(gaussians, camera, bg_color)
            
            output_path = os.path.join(train_output_dir, f"{camera.image_name}.png")
            save_heatmap(heatmap, output_path)
    
    # Process test views
    if test_indices:
        test_output_dir = os.path.join(args.output_dir, "test")
        os.makedirs(test_output_dir, exist_ok=True)
        
        print(f"\nProcessing {len(test_indices)} test views...")
        for idx in tqdm(test_indices, desc="Test views"):
            camera = test_cameras[idx]
            heatmap = extract_rf_heatmap(gaussians, camera, bg_color)
            
            output_path = os.path.join(test_output_dir, f"{camera.image_name}.png")
            save_heatmap(heatmap, output_path)
    
    total_extracted = len(train_indices) + len(test_indices)
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Extracted {total_extracted} RF heatmaps to {args.output_dir}")
    
    # Save metadata
    metadata_path = os.path.join(args.output_dir, "extraction_info.txt")
    with open(metadata_path, "w") as f:
        f.write(f"RF Heatmap Extraction Report\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Input PLY: {args.ply_path}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Total Gaussians: {gaussians.get_xyz.shape[0]}\n")
        f.write(f"Training views extracted: {len(train_indices)}\n")
        f.write(f"Test views extracted: {len(test_indices)}\n\n")
        f.write(f"Extraction Method:\n")
        f.write(f"  1. Load 3D Gaussians from trained PLY\n")
        f.write(f"  2. Load camera parameters (Rx positions, orientations, FOV)\n")
        f.write(f"  3. Project Gaussians to 2D image plane (frustum culling)\n")
        f.write(f"  4. Accumulate opacity/RF power: Pixel(u,v) = Σ α_i · G(u,v) · color_i\n")
    
    print(f"✓ Metadata saved to {metadata_path}\n")


if __name__ == "__main__":
    main()
