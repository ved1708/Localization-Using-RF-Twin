#!/usr/bin/env python3
"""Render equirectangular RF images from a trained RF-3DGS point cloud.

This uses the same core rendering algorithm as RF-3DGS/render.py:
  - gaussian_renderer.render(...)
  - GaussianModel SH evaluation + rasterization

Difference:
    - Instead of dataset pinhole cameras, it builds 6 directional cameras at each RX,
        renders each view, and accumulates intensity into angular (azimuth, zenith) bins.
    - This avoids cubemap stitching artifacts and aligns with AoA-spectrum workflows.
"""

import os
import math
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add RF-3DGS to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "RF-3DGS"))

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class SimpleCamera:
    """Minimal camera object compatible with gaussian_renderer.render."""

    def __init__(self, position, forward, up_hint, image_size=512, fov_degrees=90.0):
        self.image_name = "equirect_face"
        self.image_width = image_size
        self.image_height = image_size

        fov_rad = math.radians(fov_degrees)
        self.FoVx = fov_rad
        self.FoVy = fov_rad
        self.znear = 0.01
        self.zfar = 100.0

        pos = np.asarray(position, dtype=np.float32)
        fwd = np.asarray(forward, dtype=np.float32)
        fwd = fwd / (np.linalg.norm(fwd) + 1e-8)

        up = np.asarray(up_hint, dtype=np.float32)
        up = up / (np.linalg.norm(up) + 1e-8)

        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            right = np.cross(fwd, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, fwd)
        up = up / (np.linalg.norm(up) + 1e-8)

        # Match RF-3DGS camera convention used in custom scripts.
        R = np.stack([right, up, -fwd], axis=1)
        T = -R.T @ pos

        self.R = R
        self.T = T

        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T), dtype=torch.float32, device="cuda"
        ).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).to("cuda")
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)).squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def cubemap_faces():
    """Return canonical directions and up vectors for 6-view coverage."""
    return [
        ("posx", np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ("negx", np.array([-1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ("posy", np.array([0.0, 1.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32)),
        ("negy", np.array([0.0, -1.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ("posz", np.array([0.0, 0.0, 1.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ("negz", np.array([0.0, 0.0, -1.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]


def compute_direction_map(camera, height, width):
    """Compute per-pixel AoA map using Sionna/RF convention.

    Returns array (H, W, 2): [azimuth_deg, zenith_deg].
    """
    y_ndc = np.linspace(1.0, -1.0, height)
    x_ndc = np.linspace(-1.0, 1.0, width)
    x_grid, y_grid = np.meshgrid(x_ndc, y_ndc)

    z_cam = -np.ones_like(x_grid)
    x_cam = x_grid * np.tan(camera.FoVx / 2.0)
    y_cam = y_grid * np.tan(camera.FoVy / 2.0)

    norm = np.sqrt(x_cam ** 2 + y_cam ** 2 + z_cam ** 2) + 1e-8
    x_cam /= norm
    y_cam /= norm
    z_cam /= norm

    dir_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    # Match the convention used in extract_intensity_from_rendered_rf3dgs.py
    # (camera-space directions transformed with R, not R^T).
    dir_world = np.dot(dir_cam, camera.R)

    dx = dir_world[..., 0]
    dy = dir_world[..., 1]
    dz = dir_world[..., 2]

    azimuth_rad = np.arctan2(-dx, dz)
    zenith_rad = np.pi / 2.0 - np.arcsin(np.clip(dy, -1.0, 1.0))

    azimuth_deg = np.degrees(azimuth_rad)
    zenith_deg = np.degrees(zenith_rad)
    return np.stack([azimuth_deg, zenith_deg], axis=-1)


def render_equirect_angular_accumulation(gaussians, rx_pos, image_size, eq_height, pipeline, bg_color):
    """Render and accumulate into equirectangular angular bins.

    Uses 6 directional renders and bins each pixel by AoA direction.
    """
    eq_width = 2 * eq_height
    spectrum = np.zeros((eq_height, eq_width), dtype=np.float32)

    with torch.no_grad():
        for _, forward, up in cubemap_faces():
            cam = SimpleCamera(rx_pos, forward, up, image_size=image_size, fov_degrees=110.0)
            out = render(cam, gaussians, pipeline, bg_color)["render"]
            rgb = out.detach().cpu().numpy()  # (3, H, W)

            # RF-3DGS is already effectively single-channel; use direct channel 0.
            intensity = np.clip(rgb[0], 0.0, 1.0)

            dirmap = compute_direction_map(cam, image_size, image_size)
            az = dirmap[..., 0]  # [-180, 180]
            ze = dirmap[..., 1]  # [0, 180]

            az_idx = np.floor((az + 180.0) / 360.0 * eq_width).astype(np.int32)
            ze_idx = np.floor(ze / 180.0 * eq_height).astype(np.int32)
            az_idx = np.clip(az_idx, 0, eq_width - 1)
            ze_idx = np.clip(ze_idx, 0, eq_height - 1)

            np.add.at(spectrum, (ze_idx.ravel(), az_idx.ravel()), intensity.ravel())

    max_val = float(np.max(spectrum))
    if max_val > 0:
        spectrum /= max_val
    return np.clip(spectrum, 0.0, 1.0)


def parse_args():
    p = argparse.ArgumentParser(description="Render equirectangular images from RF-3DGS model")
    p.add_argument("--ply_path", required=True, help="Path to point_cloud.ply")
    p.add_argument("--rx_position", nargs=3, type=float, action="append", required=True,
                   metavar=("X", "Y", "Z"), help="Receiver position; pass multiple times")
    p.add_argument("--output_dir", required=True, help="Output folder for equirect PNGs")
    p.add_argument("--face_size", type=int, default=512, help="Cubemap face resolution")
    p.add_argument("--eq_height", type=int, default=512, help="Equirectangular height (width=2*height)")
    p.add_argument("--scene_center", nargs=3, type=float, default=None,
                   help="Compatibility argument (not used in angular accumulation mode)")
    p.add_argument("--white_background", action="store_true", help="Use white background")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.ply_path):
        raise FileNotFoundError(f"PLY not found: {args.ply_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    safe_state(silent=True)

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(args.ply_path)

    class PipeConfig:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False

    pipe = PipeConfig()

    bg = [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    rx_positions = np.asarray(args.rx_position, dtype=np.float32)

    print(f"Loaded Gaussians: {gaussians.get_xyz.shape[0]}")
    print(f"Rendering {len(rx_positions)} equirect image(s)...")

    for i, rx in enumerate(tqdm(rx_positions, desc="Equirect render")):
        eq = render_equirect_angular_accumulation(
            gaussians=gaussians,
            rx_pos=rx,
            image_size=args.face_size,
            eq_height=args.eq_height,
            pipeline=pipe,
            bg_color=background,
        )

        out_name = f"spectrum_{i:04d}_equirect_mpc.png"
        out_path = os.path.join(args.output_dir, out_name)
        plt.imsave(out_path, eq, cmap="gray", vmin=0.0, vmax=1.0)

    print(f"Saved equirect PNGs to: {args.output_dir}")


if __name__ == "__main__":
    main()
