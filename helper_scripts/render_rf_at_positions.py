"""
Render RF at specific Rx positions using trained RF-3DGS model.
This script creates virtual cameras at exact positions and renders RF heatmaps.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RF-3DGS'))

from scene import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from utils.general_utils import PILtoTorch
from argparse import Namespace


def create_camera_at_position(position, target, fov_deg, width, height, camera_id):
    """
    Create a Camera object at specified position looking at target.
    
    Args:
        position: [x, y, z] camera position
        target: [x, y, z] point to look at
        fov_deg: Field of view in degrees
        width: Image width
        height: Image height
        camera_id: Unique camera ID
    """
    # Convert position and target to numpy arrays
    pos = np.array(position, dtype=np.float32)
    tgt = np.array(target, dtype=np.float32)
    
    # Compute camera orientation (look-at matrix)
    # Forward direction (camera looks along -Z in camera space)
    forward = tgt - pos
    forward = forward / np.linalg.norm(forward)
    
    # Right direction (assuming world up is +Z)
    world_up = np.array([0, 0, 1], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    
    # Up direction
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Build rotation matrix (world to camera)
    # Camera convention: +X right, +Y up, -Z forward
    R = np.array([
        right,
        up,
        -forward
    ])
    
    # Translation (camera position in world)
    T = pos
    
    # Convert FOV to radians
    fov_rad = np.deg2rad(fov_deg)
    
    # Create Camera object
    camera = Camera(
        colmap_id=camera_id,
        R=R,
        T=T,
        FoVx=fov_rad,
        FoVy=fov_rad,
        image=torch.zeros((3, height, width)),  # Dummy image
        gt_alpha_mask=None,
        image_name=f"rx_pos_{camera_id}",
        uid=camera_id
    )
    
    return camera


def load_gaussians_from_ply(ply_path):
    """Load Gaussian model from PLY file."""
    print(f"Loading Gaussians from: {ply_path}")
    
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)
    
    num_points = gaussians.get_xyz.shape[0]
    print(f"Loaded {num_points} Gaussian points")
    
    return gaussians


def render_rf_at_camera(gaussians, camera, bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")):
    """
    Render RF at specified camera position.
    
    Returns:
        rendered_image: [3, H, W] tensor with RF values in color channels
    """
    # Render using Gaussian splatting
    render_pkg = render(camera, gaussians, Namespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
        debug=False
    ), bg_color)
    
    rendered_image = render_pkg["render"]
    return rendered_image


def save_rf_image(rf_tensor, output_path):
    """
    Save RF tensor as PNG image.
    
    Args:
        rf_tensor: [3, H, W] tensor with RF values
        output_path: Path to save PNG
    """
    # Convert to numpy and transpose to [H, W, 3]
    rf_np = rf_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # Normalize to [0, 1] range
    rf_min = rf_np.min()
    rf_max = rf_np.max()
    if rf_max > rf_min:
        rf_normalized = (rf_np - rf_min) / (rf_max - rf_min)
    else:
        rf_normalized = rf_np
    
    # Convert to uint8
    rf_uint8 = (rf_normalized * 255).astype(np.uint8)
    
    # Save as PNG
    img = Image.fromarray(rf_uint8)
    img.save(output_path)
    print(f"Saved RF image to: {output_path}")
    print(f"  RF range: [{rf_min:.4f}, {rf_max:.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Render RF at specific Rx positions")
    parser.add_argument("--ply_path", type=str, required=True, help="Path to trained model PLY file")
    parser.add_argument("--rx_positions", type=str, required=True, 
                        help="Comma-separated Rx positions, format: 'x1,y1,z1;x2,y2,z2'")
    parser.add_argument("--target", type=str, default="3.5,2.5,1.2",
                        help="Target point to look at (default: 3.5,2.5,1.2)")
    parser.add_argument("--fov", type=float, default=120.0,
                        help="Field of view in degrees (default: 120)")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Image resolution (default: 1024)")
    parser.add_argument("--output_dir", type=str, default="rendered_rf_positions",
                        help="Output directory for rendered images")
    
    args = parser.parse_args()
    
    # Parse Rx positions
    rx_positions = []
    for pos_str in args.rx_positions.split(';'):
        coords = [float(x) for x in pos_str.strip().split(',')]
        rx_positions.append(coords)
    print(f"\nRx positions: {rx_positions}")
    
    # Parse target
    target = [float(x) for x in args.target.split(',')]
    print(f"Target point: {target}")
    print(f"FOV: {args.fov}°")
    print(f"Resolution: {args.resolution}x{args.resolution}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Gaussian model
    gaussians = load_gaussians_from_ply(args.ply_path)
    
    # Render at each Rx position
    for i, rx_pos in enumerate(rx_positions, start=1):
        print(f"\n--- Rendering at Rx position {i}: {rx_pos} ---")
        
        # Create camera at this position
        camera = create_camera_at_position(
            position=rx_pos,
            target=target,
            fov_deg=args.fov,
            width=args.resolution,
            height=args.resolution,
            camera_id=i
        )
        
        # Render RF
        with torch.no_grad():
            rf_image = render_rf_at_camera(gaussians, camera)
        
        # Save image
        output_path = os.path.join(args.output_dir, f"static_rf_rx{i}.png")
        save_rf_image(rf_image, output_path)
    
    print(f"\n✓ All RF images rendered successfully!")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
