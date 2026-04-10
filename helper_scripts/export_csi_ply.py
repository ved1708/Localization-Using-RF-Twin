#!/usr/bin/env python3
"""
Export a colored PLY where vertex colors encode a per-point CSI-related metric
from a trained RF-3DGS model's point_cloud PLY.

Usage:
  python scripts/export_csi_ply.py --model_dir output/rf_model --iteration 30000 \
      --metric f_rest_mag --out output/rf_model/iteration_30000/csi_colored.ply

Metrics: 'opacity', 'f_dc_mag', 'f_rest_mag', 'f_dc_0' (any specific attribute name)
"""
import argparse
import os
import numpy as np
from plyfile import PlyData, PlyElement

try:
    import matplotlib.cm as cm
except Exception:
    cm = None


def load_ply(path):
    ply = PlyData.read(path)
    props = ply.elements[0].data
    names = ply.elements[0].properties
    # Build dict of arrays
    data = {}
    for name in props.dtype.names:
        data[name] = np.asarray(props[name])
    return data


def compute_metric(data, metric):
    if metric == 'opacity':
        if 'opacity' in data:
            vals = data['opacity'].astype(np.float32)
        else:
            raise ValueError('opacity not found in PLY')
    elif metric == 'f_dc_mag':
        fdc = [k for k in data.keys() if k.startswith('f_dc_')]
        if not fdc:
            raise ValueError('no f_dc_ fields found')
        arr = np.stack([data[k].astype(np.float32) for k in sorted(fdc)], axis=1)
        vals = np.linalg.norm(arr, axis=1)
    elif metric == 'f_rest_mag':
        frest = [k for k in data.keys() if k.startswith('f_rest_')]
        if not frest:
            raise ValueError('no f_rest_ fields found')
        arr = np.stack([data[k].astype(np.float32) for k in sorted(frest)], axis=1)
        vals = np.linalg.norm(arr, axis=1)
    else:
        # try direct attribute name
        if metric in data:
            vals = data[metric].astype(np.float32)
        else:
            raise ValueError(f'metric {metric} not found')

    # flatten if necessary
    vals = vals.reshape(-1)
    return vals


def map_to_rgb(vals, colormap='viridis'):
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    if np.isclose(vmax, vmin):
        norm = np.zeros_like(vals)
    else:
        norm = (vals - vmin) / (vmax - vmin)
    if cm is not None:
        cmap = cm.get_cmap(colormap)
        mapped = (cmap(norm)[:, :3] * 255.0).astype(np.uint8)
    else:
        # fallback grayscale
        g = (norm * 255.0).astype(np.uint8)
        mapped = np.stack((g, g, g), axis=1)
    return mapped, vmin, vmax


def write_full_gaussian_ply(out_path, original_ply_path, rgb):
    # Load original ply to preserve all structure
    ply = PlyData.read(original_ply_path)
    vertex = ply.elements[0]
    
    # Gaussian Splatting uses SH (Spherical Harmonics) for color.
    # The first 3 coefficients (f_dc_0, 1, 2) correspond to the base color in SH space.
    # SH = (RGB - 0.5) / 0.28209
    rgb_float = rgb.astype(np.float32) / 255.0
    sh_dc = (rgb_float - 0.5) / 0.28209479177387814
    
    # Create copy of data to modify
    data = vertex.data.copy()
    
    # Overwrite f_dc_0, f_dc_1, f_dc_2
    data['f_dc_0'] = sh_dc[:, 0]
    data['f_dc_1'] = sh_dc[:, 1]
    data['f_dc_2'] = sh_dc[:, 2]
    
    # Zero out all f_rest coefficients so we have flat colors
    f_rest_names = [p.name for p in vertex.properties if p.name.startswith("f_rest_")]
    for name in f_rest_names:
        data[name] = 0.0
        
    # Write back
    PlyData([PlyElement.describe(data, 'vertex')], text=False).write(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='model folder containing point_cloud/iteration_X/point_cloud.ply')
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--metric', default='f_rest_mag', help="metric: 'opacity','f_dc_mag','f_rest_mag' or attribute name")
    parser.add_argument('--out', required=True)
    parser.add_argument('--colormap', default='viridis')
    parser.add_argument('--mode', default='minimal', choices=['minimal', 'full'], 
                        help="minimal: XYZ-RGB PLY for MeshLab. full: A proper Gaussian PLY readable by 3DGS viewers.")
    args = parser.parse_args()

    ply_path = os.path.join(args.model_dir, 'point_cloud', f'iteration_{args.iteration}', 'point_cloud.ply')
    if not os.path.exists(ply_path):
        raise FileNotFoundError(ply_path)

    data = load_ply(ply_path)
    xyz = np.stack((data['x'], data['y'], data['z']), axis=1).astype(np.float32)
    vals = compute_metric(data, args.metric)
    rgb, vmin, vmax = map_to_rgb(vals, args.colormap)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.mode == 'full':
        write_full_gaussian_ply(args.out, ply_path, rgb)
    else:
        write_rgb_ply(args.out, xyz, rgb)
    print(f'Wrote {args.out} (mode {args.mode}, metric {args.metric} range {vmin:.6g}..{vmax:.6g})')


if __name__ == '__main__':
    main()
