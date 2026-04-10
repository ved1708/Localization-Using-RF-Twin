#!/usr/bin/env python3
"""
Compute AoA/amplitude fingerprints directly from a trained RF-3DGS point_cloud.ply.

This avoids image rasterization and is typically much faster for localization tasks.

What it does:
1) Loads Gaussians from PLY (xyz, SH coeffs, opacity).
2) For a given RX position, computes incoming direction from each Gaussian to RX.
3) Converts direction to AoA bins (theta, phi).
4) Computes per-Gaussian amplitude proxy from SH (or DC only), weighted by opacity and path loss.
5) Accumulates a 2D AoA spectrum and reports top-K peaks.

Notes:
- This is a compact fingerprinting proxy, not exact physical MPC extraction.
- In the stored PLY, opacity is pre-sigmoid; we apply sigmoid before weighting.
"""

import argparse
import csv
import math
import os
from typing import Tuple

import numpy as np
from plyfile import PlyData


# SH constants from RF-3DGS/utils/sh_utils.py
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def eval_sh_numpy(deg: int, sh: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """
    Evaluate SH with hardcoded basis up to degree 3.

    Args:
        deg: SH degree in [0, 3]
        sh:  shape (N, 3, (deg+1)^2)
        dirs: shape (N, 3), unit vectors

    Returns:
        rgb-like values, shape (N, 3)
    """
    if deg < 0 or deg > 3:
        raise ValueError("Only SH degree 0..3 is supported in this script")

    x = dirs[:, 0:1]
    y = dirs[:, 1:2]
    z = dirs[:, 2:3]

    result = C0 * sh[:, :, 0]

    if deg > 0:
        result = (
            result
            - C1 * y * sh[:, :, 1]
            + C1 * z * sh[:, :, 2]
            - C1 * x * sh[:, :, 3]
        )

    if deg > 1:
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        yz = y * z
        xz = x * z
        result = (
            result
            + C2[0] * xy * sh[:, :, 4]
            + C2[1] * yz * sh[:, :, 5]
            + C2[2] * (2.0 * zz - xx - yy) * sh[:, :, 6]
            + C2[3] * xz * sh[:, :, 7]
            + C2[4] * (xx - yy) * sh[:, :, 8]
        )

    if deg > 2:
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        yz = y * z
        xz = x * z
        result = (
            result
            + C3[0] * y * (3.0 * xx - yy) * sh[:, :, 9]
            + C3[1] * xy * z * sh[:, :, 10]
            + C3[2] * y * (4.0 * zz - xx - yy) * sh[:, :, 11]
            + C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[:, :, 12]
            + C3[4] * x * (4.0 * zz - xx - yy) * sh[:, :, 13]
            + C3[5] * z * (xx - yy) * sh[:, :, 14]
            + C3[6] * x * (xx - 3.0 * yy) * sh[:, :, 15]
        )

    return result


def load_gaussians(ply_path: str, sh_degree: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(ply_path)
    v = ply.elements[0].data

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    f_dc = f_dc[:, :, None]  # (N,3,1)

    coeff = (sh_degree + 1) ** 2
    rest_needed = 3 * (coeff - 1)
    rest_names = sorted(
        [p.name for p in ply.elements[0].properties if p.name.startswith("f_rest_")],
        key=lambda n: int(n.split("_")[-1]),
    )
    if len(rest_names) < rest_needed:
        raise ValueError(
            f"PLY has {len(rest_names)} f_rest entries, need at least {rest_needed} for sh_degree={sh_degree}"
        )

    rest = np.stack([v[name] for name in rest_names[:rest_needed]], axis=1).astype(np.float32)
    rest = rest.reshape((-1, 3, coeff - 1))

    sh = np.concatenate([f_dc, rest], axis=2)  # (N,3,coeff)

    raw_opacity = np.asarray(v["opacity"]).astype(np.float32)
    opacity = sigmoid(raw_opacity)

    return xyz, sh, opacity, raw_opacity


def compute_aoa_spectrum(
    xyz: np.ndarray,
    sh: np.ndarray,
    opacity: np.ndarray,
    rx: np.ndarray,
    sh_degree: int,
    phi_bins: int,
    theta_bins: int,
    path_loss_exp: float,
    use_dc_only: bool,
    min_opacity: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vec = rx[None, :] - xyz  # incoming direction toward receiver
    dist = np.linalg.norm(vec, axis=1) + 1e-8
    dirs = vec / dist[:, None]

    # Spherical coordinates: phi in [-180,180], theta in [0,180]
    phi = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0]))
    theta = np.degrees(np.arccos(np.clip(dirs[:, 2], -1.0, 1.0)))

    if use_dc_only:
        # SH degree-0 (DC) => amplitude proxy that ignores directionality.
        rgb = C0 * sh[:, :, 0] + 0.5
    else:
        rgb = eval_sh_numpy(sh_degree, sh, dirs) + 0.5

    rgb = np.clip(rgb, 0.0, None)
    amp = rgb.mean(axis=1)

    weight = opacity * amp / np.power(dist, path_loss_exp)
    valid = opacity >= min_opacity

    phi_idx = np.floor((phi + 180.0) / 360.0 * phi_bins).astype(np.int32)
    theta_idx = np.floor(theta / 180.0 * theta_bins).astype(np.int32)

    phi_idx = np.clip(phi_idx, 0, phi_bins - 1)
    theta_idx = np.clip(theta_idx, 0, theta_bins - 1)

    spec = np.zeros((theta_bins, phi_bins), dtype=np.float32)
    np.add.at(spec, (theta_idx[valid], phi_idx[valid]), weight[valid])

    return spec, theta, phi, weight


def topk_peaks(spec: np.ndarray, k: int) -> list:
    flat = spec.reshape(-1)
    if k >= flat.size:
        idx = np.argsort(flat)[::-1]
    else:
        idx = np.argpartition(flat, -k)[-k:]
        idx = idx[np.argsort(flat[idx])[::-1]]

    peaks = []
    h, w = spec.shape
    for i in idx:
        t = i // w
        p = i % w
        peaks.append((int(t), int(p), float(spec[t, p])))
    return peaks


def save_outputs(spec: np.ndarray, out_prefix: str, peaks: list) -> None:
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    np.save(out_prefix + "_aoa_spectrum.npy", spec)

    with open(out_prefix + "_topk.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "theta_bin", "phi_bin", "amplitude"])
        for i, (tb, pb, amp) in enumerate(peaks, start=1):
            writer.writerow([i, tb, pb, amp])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute AoA/amplitude directly from RF-3DGS PLY")
    parser.add_argument("--ply", required=True, help="Path to point_cloud.ply")
    parser.add_argument("--rx", required=True, help="Receiver position as x,y,z")
    parser.add_argument("--out_prefix", required=True, help="Output prefix, e.g. output/aoa/rx0")

    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree stored in the model (default: 3)")
    parser.add_argument("--phi_bins", type=int, default=360, help="Azimuth bins over [-180,180]")
    parser.add_argument("--theta_bins", type=int, default=180, help="Polar bins over [0,180]")
    parser.add_argument("--path_loss_exp", type=float, default=2.0, help="Distance attenuation exponent")
    parser.add_argument("--min_opacity", type=float, default=0.01, help="Ignore Gaussians below this opacity")
    parser.add_argument("--topk", type=int, default=20, help="Number of strongest AoA bins to report")
    parser.add_argument(
        "--use_dc_only",
        action="store_true",
        help="Use degree-0 SH only for amplitude (faster, less directional)",
    )

    args = parser.parse_args()

    rx = np.array([float(v) for v in args.rx.split(",")], dtype=np.float32)
    if rx.shape[0] != 3:
        raise ValueError("--rx must be x,y,z")

    xyz, sh, opacity, _ = load_gaussians(args.ply, args.sh_degree)

    spec, theta, phi, weight = compute_aoa_spectrum(
        xyz=xyz,
        sh=sh,
        opacity=opacity,
        rx=rx,
        sh_degree=args.sh_degree,
        phi_bins=args.phi_bins,
        theta_bins=args.theta_bins,
        path_loss_exp=args.path_loss_exp,
        use_dc_only=args.use_dc_only,
        min_opacity=args.min_opacity,
    )

    peaks = topk_peaks(spec, args.topk)
    save_outputs(spec, args.out_prefix, peaks)

    print(f"Loaded Gaussians: {xyz.shape[0]}")
    print(f"RX position: {rx.tolist()}")
    print(f"Saved: {args.out_prefix}_aoa_spectrum.npy")
    print(f"Saved: {args.out_prefix}_topk.csv")
    print("Top 5 peaks (theta_bin, phi_bin, amplitude):")
    for t, p, a in peaks[:5]:
        print(f"  ({t}, {p}, {a:.6e})")

    print("\nTip:")
    print("  theta_deg ~ theta_bin * (180/theta_bins)")
    print("  phi_deg   ~ phi_bin * (360/phi_bins) - 180")


if __name__ == "__main__":
    main()
