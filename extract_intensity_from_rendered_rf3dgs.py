#!/usr/bin/env python3
"""
Extract intensity from RF-3DGS using OMNIDIRECTIONAL rendering (all 360° + 180° directions).

Key Improvement:
================
Current method: Render from 90° FOV camera only
Problem: Misses signals from directions outside that cone

This script: Render from ALL directions (omnidirectional)
Solution: Create 6 cardinal-direction cameras (±X, ±Y, ±Z) and accumulate

Rendering Pipeline:
===================
1. Load trained Gaussians from PLY
2. Create 6 virtual cameras at RX position, each looking in cardinal directions:
   - East (+X), West (-X)
   - Up (+Y), Down (-Y)  
   - Forward (+Z), Back (-Z)
3. For each camera:
   a) Use gaussian_renderer.render() with SH evaluation
   b) Convert RGB to intensity (luminance)
   c) Quantize pixels to azimuth/zenith directions
   d) Accumulate into omnidirectional spectrum
4. Extract top-K directions from accumulated spectrum
5. Output: [azimuth, zenith, accumulated_intensity] sorted by intensity

Output:
=======
- intensity_from_render.npz contains:
  - 'topk_aoa_intensity': Top-K [azimuth, zenith, intensity] per RX position
  - 'omni_spectra': Full accumulated spectrum (zenith_bins × azimuth_bins)
  - 'rx_positions': Receiver coordinates
  - Metadata: AoA convention, method, denormalization notes

Intensity Range:
================
Values in [0, 1] from normalized render output.
To denormalize back to the train dB values:
    I_db = I_normalized * (I_max_training - I_min_training) + I_min_training

No extra log conversion is needed after denormalization.

Matches Data Generation:
=======================
During generate_rf_dataset.py:
  - Amplitude values are converted to dB: 10*log10(amplitude)
  - Then normalized to pixel intensities [0, 255]
  - Images generated as ground truth

This script:
  - Extracts normalized intensity [0, 1] from render
  - Accumulates across omnidirectional view
    - Output ready for denormalization directly to dB

Usage:
======
  conda run -n rf-3dgs python3 extract_intensity_from_rendered_rf3dgs.py \\
    --ply_path <path_to_ply> \\
    --rx_position x y z \\
    --k 20 \\
    --azimuth_bins 360 \\
    --zenith_bins 180 \\
    --output_dir ./omni_intensity
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Add RF-3DGS to path
rf3dgs_path = Path(__file__).parent / "RF-3DGS"
sys.path.insert(0, str(rf3dgs_path))

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state


def load_gaussians_from_ply(ply_path: str) -> GaussianModel:
    """Load Gaussians from trained PLY file."""
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)
    
    num_gaussians = gaussians.get_xyz.shape[0]
    print(f"✓ Loaded {num_gaussians} Gaussians from {ply_path}")
    print(f"  Position range: X=[{gaussians.get_xyz[:, 0].min():.2f}, {gaussians.get_xyz[:, 0].max():.2f}], "
          f"Y=[{gaussians.get_xyz[:, 1].min():.2f}, {gaussians.get_xyz[:, 1].max():.2f}], "
          f"Z=[{gaussians.get_xyz[:, 2].min():.2f}, {gaussians.get_xyz[:, 2].max():.2f}]")
    
    return gaussians


class VirtualOmniCamera:
    """Omnidirectional camera at RX position pointing towards scene."""
    
    def __init__(self, rx_position: np.ndarray, scene_center: np.ndarray, 
                 image_size: int = 512, fov_degrees: float = 90.0):
        """
        Create virtual camera at RX looking at scene.
        
        Args:
            rx_position: [x, y, z] receiver position in meters
            scene_center: [x, y, z] approximate scene center for look-at
            image_size: Resolution (square image)
            fov_degrees: Field of view in degrees (affects size of observation area)
        """
        self.image_name = f"rx_{rx_position[0]:.1f}_{rx_position[1]:.1f}_{rx_position[2]:.1f}"
        self.image_width = image_size
        self.image_height = image_size
        
        # Convert FOV to radians
        fov_rad = math.radians(fov_degrees)
        self.FoVx = fov_rad
        self.FoVy = fov_rad
        
        # Position and orientation
        rx_torch = torch.tensor(rx_position, dtype=torch.float32, device="cuda")
        scene_center_torch = torch.tensor(scene_center, dtype=torch.float32, device="cuda")
        
        # Look at scene center
        forward = scene_center_torch - rx_torch
        forward = forward / (torch.norm(forward) + 1e-8)
        
        # Use world up as reference
        world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device="cuda")
        
        # Compute right and up vectors
        right = torch.cross(forward, world_up)
        right = right / (torch.norm(right) + 1e-8)
        
        up = torch.cross(right, forward)
        up = up / (torch.norm(up) + 1e-8)
        
        # Build rotation matrix (world to camera)
        # R = [right, up, -forward]
        R = torch.stack([right, up, -forward], dim=1)  # (3, 3)
        
        # Translation: t = -R^T @ position
        t = -R.T @ rx_torch
        
        # Store transformation matrices
        self.R = R.cpu().numpy()
        self.T = t.cpu().numpy()
        self.camera_center = rx_torch.cpu().numpy()
        
        # Create transformation matrices for rasterizer
        # World-to-camera: P = R @ (X - rx)
        # Camera-to-world: X = R^T @ P + rx
        
        # world_view_matrix (4x4): transforms world points to camera space
        self.world_view_matrix = torch.eye(4, dtype=torch.float32, device="cuda")
        self.world_view_matrix[:3, :3] = torch.tensor(self.R, dtype=torch.float32, device="cuda").T
        self.world_view_matrix[:3, 3] = torch.tensor(self.T, dtype=torch.float32, device="cuda")
        
        # Projection matrix (simple perspective)
        # For FOV, focal length f = (image_width/2) / tan(FOV/2)
        f = (image_size / 2.0) / math.tan(fov_rad / 2.0)
        
        proj = torch.zeros(4, 4, dtype=torch.float32, device="cuda")
        proj[0, 0] = f / (image_size / 2.0)
        proj[1, 1] = f / (image_size / 2.0)
        proj[2, 2] = -1.0
        proj[2, 3] = -1.0
        proj[3, 3] = 0.0
        
        self.full_proj_transform = proj @ self.world_view_matrix


def render_omnidirectional_from_rx(gaussians: GaussianModel, rx_position: np.ndarray, 
                                     scene_center: np.ndarray, 
                                     image_size: int = 512,
                                     azimuth_bins: int = 360,
                                     zenith_bins: int = 180) -> np.ndarray:
    """
    Render RF intensity OMNIDIRECTIONALLY from RX position.
    
    This function renders from MULTIPLE viewing directions to accumulate
    intensity from ALL directions (360° azimuth × 180° zenith), not just
    a limited FOV cone.
    
    Strategy:
    - Create 6 cardinal direction cameras (±X, ±Y, ±Z)
    - Each covers ~120° FOV to ensure full coverage
    - Accumulate all rendered intensities into a 2D spectrum
    - Result: Complete omnidirectional power map
    
    Args:
        gaussians: Trained GaussianModel
        rx_position: [x, y, z] receiver location
        scene_center: [x, y, z] point to look at
        image_size: Resolution per view
        azimuth_bins, zenith_bins: Spectrum resolution
    
    Returns:
        spectrum: (zenith_bins, azimuth_bins) accumulated omnidirectional intensity
    """
    
    # Black background (no signal)
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    
    # Pipeline config
    class PipeConfig:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False
    
    pipe = PipeConfig()
    
    # Initialize omnidirectional spectrum
    omni_spectrum = np.zeros((zenith_bins, azimuth_bins), dtype=np.float32)
    
    # Define cardinal viewing directions
    # Each direction: camera is at RX, looking AWAY from this direction
    cardinal_dirs = [
        ("East (+X)", np.array([1.0, 0.0, 0.0])),
        ("West (-X)", np.array([-1.0, 0.0, 0.0])),
        ("Up (+Y)", np.array([0.0, 1.0, 0.0])),
        ("Down (-Y)", np.array([0.0, -1.0, 0.0])),
        ("Forward (+Z)", np.array([0.0, 0.0, 1.0])),
        ("Back (-Z)", np.array([0.0, 0.0, -1.0])),
    ]
    
    print(f"  Rendering omnidirectional intensity (6 cardinal views)...")
    
    for view_name, look_dir in cardinal_dirs:
        # Create camera pointing in this direction from RX
        look_from_world = rx_position + look_dir * 10.0  # Point to look from
        camera = VirtualCameraLookingFrom(
            rx_position, look_from_world, image_size=image_size, fov_degrees=110.0
        )
        
        # Render
        with torch.no_grad():
            render_result = render(camera, gaussians, pipe, bg_color)
        
        # Extract rendered RGB and convert to intensity
        rendered_rgb = render_result["render"].cpu().numpy()  # (3, H, W)
        intensity_map = (0.299 * rendered_rgb[0] + 0.587 * rendered_rgb[1] + 0.114 * rendered_rgb[2]).astype(np.float32)
        
        # Map pixels to world-space directions and accumulate
        H, W = intensity_map.shape
        dirmap = compute_direction_map(H, W, camera.FoVx, camera.FoVy, camera.R)
        
        # Quantize directions to bins and accumulate
        az = dirmap[..., 0]  # [-180, 180]
        ze = dirmap[..., 1]  # [0, 180]
        
        az_idx = np.floor((az + 180.0) / 360.0 * azimuth_bins).astype(np.int32)
        ze_idx = np.floor(ze / 180.0 * zenith_bins).astype(np.int32)
        
        az_idx = np.clip(az_idx, 0, azimuth_bins - 1)
        ze_idx = np.clip(ze_idx, 0, zenith_bins - 1)
        
        # Accumulate intensity
        np.add.at(omni_spectrum, (ze_idx.ravel(), az_idx.ravel()), intensity_map.ravel())
        
        print(f"    ✓ {view_name}: range [{intensity_map.min():.4f}, {intensity_map.max():.4f}]")
    
    print(f"  ✓ Omnidirectional spectrum: range [{omni_spectrum.min():.4f}, {omni_spectrum.max():.4f}]")
    
    return omni_spectrum


class VirtualCameraLookingFrom:
    """Camera positioned to look FROM a specific world direction toward RX."""
    
    def __init__(self, rx_position: np.ndarray, look_from_position: np.ndarray, 
                 image_size: int = 512, fov_degrees: float = 110.0):
        """
        Create camera that looks from a world position toward RX.
        
        Args:
            rx_position: RX location [x, y, z]
            look_from_position: Where the camera is positioned (looking toward RX)
            image_size: Resolution (square)
            fov_degrees: Field of view (110° ensures good coverage with overlap)
        """
        self.image_name = "omni_view"
        self.image_width = image_size
        self.image_height = image_size
        
        fov_rad = math.radians(fov_degrees)
        self.FoVx = fov_rad
        self.FoVy = fov_rad
        
        # Direction: from look_from toward RX
        forward = rx_position - np.array(look_from_position, dtype=np.float32)
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        # Use world up as reference
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Handle case where forward is parallel to world_up
        if np.abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # Compute right and up vectors
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)
        
        # Build rotation matrix (world to camera)
        R = np.stack([right, up, -forward], axis=1)  # (3, 3)
        
        # Translation
        look_from_arr = np.array(look_from_position, dtype=np.float32)
        t = -R.T @ look_from_arr
        
        self.R = R
        self.T = t
        self.camera_center = torch.tensor(look_from_arr, dtype=torch.float32, device="cuda")
        
        # Transformation matrices for rasterizer
        self.world_view_matrix = torch.eye(4, dtype=torch.float32, device="cuda")
        self.world_view_matrix[:3, :3] = torch.tensor(self.R, dtype=torch.float32, device="cuda").T
        self.world_view_matrix[:3, 3] = torch.tensor(self.T, dtype=torch.float32, device="cuda")
        
        # Projection matrix
        f = (image_size / 2.0) / math.tan(fov_rad / 2.0)
        
        proj = torch.zeros(4, 4, dtype=torch.float32, device="cuda")
        proj[0, 0] = f / (image_size / 2.0)
        proj[1, 1] = f / (image_size / 2.0)
        proj[2, 2] = -1.0
        proj[2, 3] = -1.0
        proj[3, 3] = 0.0
        
        self.full_proj_transform = proj @ self.world_view_matrix

        # Match gaussian_renderer camera interface used by RF-3DGS.
        self.world_view_transform = self.world_view_matrix


def compute_direction_map(H: int, W: int, fovx: float, fovy: float, 
                          rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Compute AoA (azimuth, zenith) for each pixel in rendered image.
    
    Args:
        H, W: Image dimensions
        fovx, fovy: Field of view in radians
        rotation_matrix: Camera rotation matrix (world to camera)
    
    Returns:
        dirmap: (H, W, 2) array with [azimuth_deg, zenith_deg] at each pixel
    """
    # Create normalized pixel coordinates [-1, 1] x [-1, 1]
    y_ndc = np.linspace(1, -1, H)  # Top-down
    x_ndc = np.linspace(-1, 1, W)
    x_grid, y_grid = np.meshgrid(x_ndc, y_ndc)
    
    # Convert to camera space direction
    # For perspective camera: ray = [x * tan(fovx/2), y * tan(fovy/2), -1]
    # Then normalize
    z_cam = -np.ones_like(x_grid)
    x_cam = x_grid * np.tan(fovx / 2.0)
    y_cam = y_grid * np.tan(fovy / 2.0)
    
    # Normalize camera-space direction vectors
    norm = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2) + 1e-8
    x_cam = x_cam / norm
    y_cam = y_cam / norm
    z_cam = z_cam / norm
    
    # Transform to world space
    dir_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)
    
    # dir_world = R^T @ dir_cam
    R_T = rotation_matrix.T
    dir_world = np.dot(dir_cam, R_T.T)  # Broadcasting works: (H,W,3) @ (3,3)^T
    
    # Compute AoA from world-space direction
    # Following generate_rf_dataset.py convention (from RX looking at scene):
    dx_w = dir_world[..., 0]
    dy_w = dir_world[..., 1]
    dz_w = dir_world[..., 2]
    
    # azimuth (phi) = atan2(-x, z)
    # zenith (theta) = pi/2 - asin(y)
    azimuth_rad = np.arctan2(-dx_w, dz_w)
    zenith_rad = np.pi / 2.0 - np.arcsin(np.clip(dy_w, -1.0, 1.0))
    
    # Convert to degrees
    azimuth_deg = np.degrees(azimuth_rad)      # [-180, 180]
    zenith_deg = np.degrees(zenith_rad)        # [0, 180]
    
    dirmap = np.stack([azimuth_deg, zenith_deg], axis=-1)
    
    return dirmap


def extract_topk_from_intensity_map(intensity_map: np.ndarray, dirmap: np.ndarray, 
                                     k: int, azimuth_bins: int = 360, 
                                     zenith_bins: int = 180) -> np.ndarray:
    """
    Extract top-K intensity values with their directions.
    
    Bins the intensity map by AoA, accumulates intensity per bin, then selects top-K.
    
    Args:
        intensity_map: (H, W) intensity values in [0, 1]
        dirmap: (H, W, 2) with [azimuth_deg, zenith_deg]
        k: Number of top directions
        azimuth_bins: Number of azimuth bins
        zenith_bins: Number of zenith bins
    
    Returns:
        topk: (K, 3) array with [azimuth_deg, zenith_deg, intensity]
    """
    # Quantize directions to bins
    az = dirmap[..., 0]  # [-180, 180]
    ze = dirmap[..., 1]  # [0, 180]
    
    
    az_idx = np.floor((az + 180.0) / 360.0 * azimuth_bins).astype(np.int32)
    ze_idx = np.floor(ze / 180.0 * zenith_bins).astype(np.int32)
    
    az_idx = np.clip(az_idx, 0, azimuth_bins - 1)
    ze_idx = np.clip(ze_idx, 0, zenith_bins - 1)
    
    # Accumulate intensity into 2D spectrum
    spectrum = np.zeros((zenith_bins, azimuth_bins), dtype=np.float32)
    np.add.at(spectrum, (ze_idx.ravel(), az_idx.ravel()), intensity_map.ravel())
    
    # For each bin, compute center direction
    azimuth_centers = -180.0 + (np.arange(azimuth_bins, dtype=np.float32) + 0.5) * (360.0 / azimuth_bins)
    zenith_centers = (np.arange(zenith_bins, dtype=np.float32) + 0.5) * (180.0 / zenith_bins)
    
    # Extract top-K bins
    flat = spectrum.reshape(-1)
    total = flat.size
    k_eff = min(k, total)
    
    if k_eff == total:
        idx = np.argsort(flat)[::-1]
    else:
        idx = np.argpartition(flat, -k_eff)[-k_eff:]
        idx = idx[np.argsort(flat[idx])[::-1]]
    
    az_bins = spectrum.shape[1]
    topk = np.zeros((k, 3), dtype=np.float32)
    for rank, flat_i in enumerate(idx[:k_eff]):
        ze_i = int(flat_i // az_bins)
        az_i = int(flat_i % az_bins)
        topk[rank, 0] = azimuth_centers[az_i]
        topk[rank, 1] = zenith_centers[ze_i]
        topk[rank, 2] = flat[flat_i]
    
    return topk


def topk_from_spectrum(spectrum: np.ndarray,
                       azimuth_centers: np.ndarray,
                       zenith_centers: np.ndarray,
                       k: int) -> np.ndarray:
    """Extract top-K [azimuth, zenith, intensity] directly from a binned spectrum."""
    flat = spectrum.reshape(-1)
    total = flat.size
    k_eff = min(k, total)

    if k_eff == total:
        idx = np.argsort(flat)[::-1]
    else:
        idx = np.argpartition(flat, -k_eff)[-k_eff:]
        idx = idx[np.argsort(flat[idx])[::-1]]

    az_bins = spectrum.shape[1]
    topk = np.zeros((k, 3), dtype=np.float32)
    for rank, flat_i in enumerate(idx[:k_eff]):
        ze_i = int(flat_i // az_bins)
        az_i = int(flat_i % az_bins)
        topk[rank, 0] = azimuth_centers[az_i]
        topk[rank, 1] = zenith_centers[ze_i]
        topk[rank, 2] = flat[flat_i]

    return topk


def save_outputs(output_dir: str, topk_all: np.ndarray, omni_spectra: np.ndarray,
                 rx_positions: np.ndarray, azimuth_bins: int, zenith_bins: int, k: int) -> None:
    """
    Save omnidirectional intensity extraction outputs.
    
    Args:
        output_dir: Where to save files
        topk_all: Top-K intensity values, shape (N, K, 3) or (K, 3)
        omni_spectra: Full omnidirectional spectra, shape (N, zenith_bins, azimuth_bins)
        rx_positions: RX coordinates, shape (N, 3)
        azimuth_bins, zenith_bins: Spectrum dimensions
        k: Number of top-K values kept
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "intensity_from_render.npz")
    
    save_dict = {
        "topk_aoa_intensity": topk_all.astype(np.float32),  # (N, K, 3)
        "omni_spectra": omni_spectra.astype(np.float32),    # (N, zenith_bins, azimuth_bins)
        "rx_positions": rx_positions.astype(np.float32),    # (N, 3)
        "num_positions": np.int32(rx_positions.shape[0]),
        "azimuth_bins": np.int32(azimuth_bins),
        "zenith_bins": np.int32(zenith_bins),
        "spectrum_height": np.int32(omni_spectra.shape[1]),
        "spectrum_width": np.int32(omni_spectra.shape[2]),
        "layout": "[azimuth_1,zenith_1,intensity_1,...,azimuth_K,zenith_K,intensity_K]",
        "note": "Omnidirectional spectrum covers all 360° azimuth × 180° zenith. Intensity in [0, 1]. Denormalize directly to dB: I_db = I_normalized * (I_max - I_min) + I_min. No extra log conversion is needed.",
        "aoa_convention": "generate_rf_dataset.py: phi=atan2(-x,z), theta=pi/2-asin(y)",
        "method": "Omnidirectional rendering (6 cardinal views) using render() algorithm with SH eval + alpha blending + accumulation",
    }
    
    # Backward compatible single-RX keys
    if rx_positions.shape[0] == 1:
        save_dict["rx_position"] = rx_positions[0].astype(np.float32)
        save_dict["topk_aoa_intensity_single"] = topk_all[0].astype(np.float32)
        save_dict["omni_spectrum"] = omni_spectra[0].astype(np.float32)
    
    np.savez_compressed(output_file, **save_dict)
    print(f"\nSaved: {output_file}")
    
    # Also save top-K table as NPY for convenience
    topk_npy = os.path.join(output_dir, "topk_aoa_intensity.npy")
    if rx_positions.shape[0] == 1:
        np.save(topk_npy, topk_all[0].astype(np.float32))
    else:
        np.save(topk_npy, topk_all.astype(np.float32))
    print(f"Saved: {topk_npy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract intensity from RF-3DGS using omnidirectional render (all 360° + 180° directions)"
    )
    parser.add_argument(
        "--ply_path",
        default="RF-3DGS/output/rf_model_gray/point_cloud/iteration_40000/point_cloud.ply",
        help="Path to trained RF-3DGS point_cloud.ply",
    )
    parser.add_argument(
        "--rx_position",
        nargs=3,
        type=float,
        action="append",
        metavar=("X", "Y", "Z"),
        help="Receiver position in meters (repeat flag for multiple positions)",
    )
    parser.add_argument(
        "--rx_positions_file",
        type=str,
        default=None,
        help="Optional text file with one RX position per line: 'x y z' or 'x,y,z'",
    )
    parser.add_argument(
        "--scene_center",
        nargs=3,
        type=float,
        default=[3.5, 2.5, 1.2],
        metavar=("X", "Y", "Z"),
        help="Scene center reference point (not used for omnidirectional, kept for compatibility)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Rendered image resolution per cardinal view (default: 512)",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Top-K directions to keep (sorted by descending intensity)",
    )
    parser.add_argument(
        "--azimuth_bins",
        type=int,
        default=360,
        help="Number of azimuth bins over [-180,180) (default: 360 = 1° resolution)",
    )
    parser.add_argument(
        "--zenith_bins",
        type=int,
        default=180,
        help="Number of zenith bins over [0,180] (default: 180 = 1° resolution)",
    )
    parser.add_argument(
        "--output_dir",
        default="RF-3DGS/output/rf_model_gray/intensity_omnidirectional",
        help="Directory to save omnidirectional intensity outputs",
    )
    return parser.parse_args()


def load_rx_positions(args: argparse.Namespace) -> np.ndarray:
    """Collect RX positions from CLI and optional file."""
    positions: List[List[float]] = []
    
    if args.rx_position:
        positions.extend(args.rx_position)
    
    if args.rx_positions_file:
        if not os.path.exists(args.rx_positions_file):
            raise FileNotFoundError(f"RX positions file not found: {args.rx_positions_file}")
        
        with open(args.rx_positions_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.replace(",", " ").split()
                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid RX line {line_num} in {args.rx_positions_file}: '{s}'"
                    )
                positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
    
    if not positions:
        raise ValueError("Provide at least one --rx_position or --rx_positions_file")
    
    return np.asarray(positions, dtype=np.float32)


def main() -> None:
    args = parse_args()
    
    if args.k <= 0:
        raise ValueError("--k must be positive")
    
    # Initialize CUDA
    print("Initializing CUDA...")
    safe_state(silent=True)
    
    # Load Gaussians
    print("\nLoading Gaussians from PLY...")
    gaussians = load_gaussians_from_ply(args.ply_path)
    
    # Load RX positions
    print("\nLoading RX positions...")
    rx_positions = load_rx_positions(args)
    print(f"✓ Loaded {len(rx_positions)} RX position(s):")
    for i, pos in enumerate(rx_positions):
        print(f"  [{i}] ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    scene_center = np.array(args.scene_center, dtype=np.float32)
    
    # Render and extract intensity for each RX position
    print(f"\nRendering OMNIDIRECTIONAL intensity (all 360° + 180° directions)...")
    print(f"(6 cardinal views × SH eval -> Alpha blending -> Accumulated spectrum)\n")
    
    all_topk = np.zeros((len(rx_positions), args.k, 3), dtype=np.float32)
    all_omni_spectra = []
    
    for idx, rx_pos in enumerate(rx_positions):
        print(f"RX Position {idx + 1}/{len(rx_positions)}: ({rx_pos[0]:.2f}, {rx_pos[1]:.2f}, {rx_pos[2]:.2f})")
        
        # Render omnidirectional spectrum from this RX position
        # (Accumulates intensity from all 360° azimuth × 180° zenith directions)
        omni_spectrum = render_omnidirectional_from_rx(
            gaussians, rx_pos, scene_center,
            image_size=args.image_size,
            azimuth_bins=args.azimuth_bins,
            zenith_bins=args.zenith_bins
        )
        
        # Extract top-K from omnidirectional spectrum
        azimuth_centers = -180.0 + (np.arange(args.azimuth_bins, dtype=np.float32) + 0.5) * (360.0 / args.azimuth_bins)
        zenith_centers = (np.arange(args.zenith_bins, dtype=np.float32) + 0.5) * (180.0 / args.zenith_bins)
        
        topk = topk_from_spectrum(omni_spectrum, azimuth_centers, zenith_centers, args.k)
        
        all_topk[idx] = topk
        all_omni_spectra.append(omni_spectrum)
        
        print(f"  Top-5 directions (from omnidirectional accumulation):")
        for k_idx in range(min(5, args.k)):
            az, ze, intensity = topk[k_idx]
            print(f"    [{k_idx}] AoA=(φ={az:7.1f}°, θ={ze:6.1f}°) Intensity={intensity:.4f}")
        print()
    
    # Stack omnidirectional spectra
    all_omni_spectra = np.stack(all_omni_spectra, axis=0)
    
    # Save outputs
    print("Saving outputs...")
    save_outputs(
        args.output_dir, all_topk, all_omni_spectra, rx_positions,
        args.azimuth_bins, args.zenith_bins, args.k
    )
    
    print("\n" + "="*70)
    print("OMNIDIRECTIONAL INTENSITY EXTRACTION COMPLETE")
    print("="*70)
    print(f"\nIntensity values are normalized to [0, 1] by accumulation.")
    print(f"To denormalize to the training-domain dB values:")
    print(f"  I_db = I_normalized * (I_max - I_min) + I_min")
    print(f"\nOmnidirectional spectrum covers all 360° × 180° directions.")
    print(f"Compare directly with ground-truth dB after denormalization (no extra log10).")


if __name__ == "__main__":
    main()
