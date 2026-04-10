#!/usr/bin/env python3
"""
Extract a CSI-like top-K AoA/amplitude vector from a trained RF-3DGS point cloud.

The script:
1) Loads Gaussians from a `point_cloud.ply`.
2) Computes incoming AoA per Gaussian using the same convention as generate_rf_dataset.py:
    - azimuth (phi)  = atan2(-x, z)
    - zenith  (theta)= pi/2 - asin(y)
3) Builds a 2D AoA spectrum over (zenith, azimuth) by accumulating amplitude weights.
4) Outputs top-K [azimuth_deg, zenith_deg, amplitude] sorted by descending amplitude.

Output files:
- rf3dgs_csi.npz              : consolidated output containing AoA and amplitude top-K data
- rf3dgs_csi.npy      : top-K table (shape: (N,K,3) or (K,3) for single-RX compatibility)
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np


# SH degree-0 constant used in 3DGS for DC term.
C0 = 0.28209479177387814


PLY_TYPE_TO_NUMPY = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def read_binary_ply_vertices(ply_path: str) -> np.ndarray:
    """Read binary little-endian PLY vertex table into a structured NumPy array."""
    with open(ply_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY: unexpected EOF before end_header")
            text = line.decode("ascii", errors="strict").strip()
            header_lines.append(text)
            if text == "end_header":
                break

        if not header_lines or header_lines[0] != "ply":
            raise ValueError("Invalid PLY: missing 'ply' magic header")

        fmt = None
        vertex_count = None
        in_vertex = False
        props = []

        for line in header_lines[1:]:
            if line.startswith("format "):
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid format line: {line}")
                fmt = parts[1]
            elif line.startswith("element "):
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Invalid element line: {line}")
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif line.startswith("property ") and in_vertex:
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(
                        "Only scalar vertex properties are supported in this script"
                    )
                ply_type, name = parts[1], parts[2]
                if ply_type not in PLY_TYPE_TO_NUMPY:
                    raise ValueError(f"Unsupported PLY property type: {ply_type}")
                props.append((name, "<" + PLY_TYPE_TO_NUMPY[ply_type]))

        if fmt != "binary_little_endian":
            raise ValueError(
                f"Unsupported PLY format: {fmt}. This script supports binary_little_endian only."
            )
        if vertex_count is None:
            raise ValueError("PLY has no vertex element")
        if not props:
            raise ValueError("PLY vertex element has no scalar properties")

        dtype = np.dtype(props)
        data = np.fromfile(f, dtype=dtype, count=vertex_count)
        if data.shape[0] != vertex_count:
            raise ValueError(
                f"PLY vertex read mismatch: expected {vertex_count}, got {data.shape[0]}"
            )
        return data


def load_gaussian_fields(ply_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load xyz, opacity, and grayscale DC feature from RF-3DGS PLY."""
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    vertex = read_binary_ply_vertices(ply_path)

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

    if "opacity" not in vertex.dtype.names:
        raise ValueError("PLY is missing required field: opacity")
    opacity = sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))

    # RF grayscale training typically stores energy in SH DC channels.
    dc_fields = [name for name in ("f_dc_0", "f_dc_1", "f_dc_2") if name in vertex.dtype.names]
    if not dc_fields:
        raise ValueError("PLY is missing SH DC fields (f_dc_0/f_dc_1/f_dc_2)")

    dc = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in dc_fields], axis=1)
    # Convert SH DC to a non-negative amplitude proxy.
    amp_dc = np.clip(C0 * dc + 0.5, 0.0, None).mean(axis=1)

    return xyz, opacity, amp_dc


def build_aoa_spectrum(
    xyz: np.ndarray,
    opacity: np.ndarray,
    amp_dc: np.ndarray,
    rx: np.ndarray,
    azimuth_bins: int,
    zenith_bins: int,
    min_opacity: float,
    path_loss_exp: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 2D AoA spectrum over (zenith, azimuth) with generate_rf_dataset.py convention."""
    vec = rx[None, :] - xyz
    dist = np.linalg.norm(vec, axis=1) + 1e-8
    dirs = vec / dist[:, None]

    # Match generate_rf_dataset.py camera-local spherical convention:
    # phi   = atan2(-x, z)                  -> azimuth in [-180, 180]
    # theta = pi/2 - asin(y)                -> zenith  in [0, 180]
    azimuth_deg = np.degrees(np.arctan2(-dirs[:, 0], dirs[:, 2]))
    zenith_deg = np.degrees(np.pi / 2.0 - np.arcsin(np.clip(dirs[:, 1], -1.0, 1.0)))

    az_idx = np.floor((azimuth_deg + 180.0) / 360.0 * azimuth_bins).astype(np.int32)
    ze_idx = np.floor(zenith_deg / 180.0 * zenith_bins).astype(np.int32)
    az_idx = np.clip(az_idx, 0, azimuth_bins - 1)
    ze_idx = np.clip(ze_idx, 0, zenith_bins - 1)

    weight = opacity * amp_dc / np.power(dist, path_loss_exp)
    valid = opacity >= min_opacity

    spectrum = np.zeros((zenith_bins, azimuth_bins), dtype=np.float32)
    np.add.at(spectrum, (ze_idx[valid], az_idx[valid]), weight[valid])

    azimuth_centers = -180.0 + (np.arange(azimuth_bins, dtype=np.float32) + 0.5) * (360.0 / azimuth_bins)
    zenith_centers = (np.arange(zenith_bins, dtype=np.float32) + 0.5) * (180.0 / zenith_bins)
    return spectrum, azimuth_centers, zenith_centers


def topk_from_spectrum(
    spectrum: np.ndarray,
    azimuth_centers: np.ndarray,
    zenith_centers: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return top-K rows as [azimuth_deg, zenith_deg, amplitude], sorted by descending amplitude."""
    if k <= 0:
        raise ValueError("K must be positive")

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


def save_outputs(
    output_dir: str,
    topk_all: np.ndarray,
    rx_positions: np.ndarray,
    azimuth_bins: int,
    zenith_bins: int,
    k: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    meta_npz = os.path.join(output_dir, "rf3dgs_csi.npz")
    topk_npy = os.path.join(output_dir, "rf3dgs_csi.npy")

    # Save NPY top-K AoA/amplitude table.
    if rx_positions.shape[0] == 1:
        np.save(topk_npy, topk_all[0].astype(np.float32))
    else:
        np.save(topk_npy, topk_all.astype(np.float32))

    save_dict = {
        "topk_aoa_amplitude": topk_all.astype(np.float32),
        "rx_positions": rx_positions.astype(np.float32),
        "num_positions": np.int32(rx_positions.shape[0]),
        "azimuth_bins": np.int32(azimuth_bins),
        "zenith_bins": np.int32(zenith_bins),
        "layout": "[azimuth_1,zenith_1,amp_1,...,azimuth_K,zenith_K,amp_K]",
        "aoa_convention": "generate_rf_dataset.py: phi=atan2(-x,z), theta=pi/2-asin(y)",
    }

    # Backward-compatible keys when only one RX is requested.
    if rx_positions.shape[0] == 1:
        save_dict["rx_position"] = rx_positions[0].astype(np.float32)
        save_dict["topk_aoa_amplitude_single"] = topk_all[0].astype(np.float32)

    np.savez_compressed(meta_npz, **save_dict)

    print(f"Saved: {meta_npz}")
    print(f"Saved: {topk_npy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract top-K AoA/amplitude CSI vector from RF-3DGS point_cloud.ply"
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
        "--k",
        type=int,
        required=True,
        help="Top-K AoA bins to keep (sorted by descending amplitude)",
    )
    parser.add_argument(
        "--azimuth_bins",
        type=int,
        default=360,
        help="Number of azimuth bins over [-180,180) (default: 360)",
    )
    parser.add_argument(
        "--zenith_bins",
        type=int,
        default=180,
        help="Number of zenith bins over [0,180] (default: 180)",
    )
    parser.add_argument(
        "--min_opacity",
        type=float,
        default=0.01,
        help="Ignore Gaussians with opacity lower than this",
    )
    parser.add_argument(
        "--path_loss_exp",
        type=float,
        default=2.0,
        help="Distance attenuation exponent in weight = amp / dist^exp",
    )
    parser.add_argument(
        "--output_dir",
        default="RF-3DGS/output/rf_model_gray/csi_topk",
        help="Directory to save CSI top-K outputs",
    )
    return parser.parse_args()


def load_rx_positions(args: argparse.Namespace) -> np.ndarray:
    """Collect RX positions from repeated --rx_position and/or --rx_positions_file."""
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
        raise ValueError("--k must be a positive integer")
    if args.azimuth_bins <= 0:
        raise ValueError("--azimuth_bins must be a positive integer")
    if args.zenith_bins <= 0:
        raise ValueError("--zenith_bins must be a positive integer")

    rx_positions = load_rx_positions(args)

    xyz, opacity, amp_dc = load_gaussian_fields(args.ply_path)
    topk_all = np.zeros((rx_positions.shape[0], args.k, 3), dtype=np.float32)

    for i, rx in enumerate(rx_positions):
        spectrum, azimuth_centers, zenith_centers = build_aoa_spectrum(
            xyz=xyz,
            opacity=opacity,
            amp_dc=amp_dc,
            rx=rx,
            azimuth_bins=args.azimuth_bins,
            zenith_bins=args.zenith_bins,
            min_opacity=args.min_opacity,
            path_loss_exp=args.path_loss_exp,
        )
        topk_all[i] = topk_from_spectrum(spectrum, azimuth_centers, zenith_centers, args.k)

    save_outputs(
        args.output_dir,
        topk_all,
        rx_positions,
        args.azimuth_bins,
        args.zenith_bins,
        args.k,
    )

    print(f"\nLoaded gaussians: {xyz.shape[0]}")
    print(f"RX positions: {rx_positions.shape[0]}")
    print(f"Top-K requested: {args.k}")
    print("Top 5 [azimuth_deg, zenith_deg, amplitude] for first RX:")
    for row in topk_all[0, :5]:
        print(f"  [{row[0]:.2f}, {row[1]:.2f}, {row[2]:.6e}]")


if __name__ == "__main__":
    main()
