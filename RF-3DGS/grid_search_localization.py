import os
import subprocess
import argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim


# ──────────────────────────────────────────────────────────────────────────────
# POSE GRID CREATION
# ──────────────────────────────────────────────────────────────────────────────

def create_poses(x_range, y_range, z_range, yaws, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    images_path  = os.path.join(out_dir, "images.txt")
    cameras_path = os.path.join(out_dir, "cameras.txt")

    with open(cameras_path, 'w') as f:
        f.write("1 PINHOLE 600 600 173.20508075688778 173.20508075688778 300.0 300.0\n")

    poses_info = []
    with open(images_path, 'w') as f:
        img_id = 1
        for ix, x in enumerate(x_range):
            for iy, y in enumerate(y_range):
                for iz, z in enumerate(z_range):
                    for itheta, theta in enumerate(yaws):
                        yaw   = np.radians(theta)
                        pitch = 0.0
                        roll  = 0.0

                        R_posz2posx = R.from_euler('ZYX', [-np.pi / 2, 0.0, -np.pi / 2])
                        R_posx2array = R.from_euler('ZYX', [yaw, pitch, roll])

                        R_w2c   = (R_posx2array * R_posz2posx).inv().as_matrix()
                        P_world = np.array([x, y, z])
                        T_w2c   = -R_w2c @ P_world

                        quat = R.from_matrix(R_w2c).as_quat()
                        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]

                        name = f"pos_X{x:.1f}_Y{y:.1f}_Z{z:.1f}_Yaw{theta}.png"
                        f.write(
                            f"{img_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                            f"{T_w2c[0]:.8f} {T_w2c[1]:.8f} {T_w2c[2]:.8f} 1 {name}\n"
                        )
                        poses_info.append({
                            'id':       img_id,
                            'name':     name,
                            'position': (x, y, z),
                            'yaw':      theta,
                            'grid_idx': (ix, iy, iz, itheta),
                            'colmap_line': f"{img_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {T_w2c[0]:.8f} {T_w2c[1]:.8f} {T_w2c[2]:.8f} 1 {name}\n"
                        })
                        img_id += 1

    return out_dir, poses_info


# ──────────────────────────────────────────────────────────────────────────────
# ROBUST PRE-PROCESSING  (the core fix)
# ──────────────────────────────────────────────────────────────────────────────

def robust_preprocess(img_float, target_size=(48, 48)):
    """
    Fixed for 600x600 input images.
    1. Percentile clip  – kills bright/dark outlier noise tiles
    2. Median blur k=19 – large enough to erase ~15-20px blocky patches
    3. Morph close      – fills dark holes left after noise removal
    4. Gaussian blur    – smooth basin for correlation
    5. INTER_AREA downsample – box-average, suppresses residual noise
    """
    amp   = img_float[:, :, 1]
    delay = np.zeros_like(amp)
    valid = amp > 1.0
    delay[valid] = img_float[:, :, 0][valid] / amp[valid]

    def clean_channel(ch):
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        rng = hi - lo
        if rng < 1e-6:
            return np.zeros(target_size, dtype=np.float32)
        ch_u8 = np.clip((ch - lo) / rng * 255.0, 0, 255).astype(np.uint8)
        ch_u8 = cv2.medianBlur(ch_u8, 19)
        ch_u8 = cv2.morphologyEx(ch_u8, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        ch_u8 = cv2.GaussianBlur(ch_u8, (15, 15), 5)
        return cv2.resize(ch_u8.astype(np.float32), target_size,
                          interpolation=cv2.INTER_AREA)

    return clean_channel(amp), clean_channel(delay)


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-SCALE CORRELATION  (robust to remaining artefacts)
# ──────────────────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    fa, fb = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(fa, fb) / (na * nb))


def multiscale_corr(feat_t, feat_p, scales=((48, 48), (24, 24), (12, 12))):
    """
    Compute correlation at multiple resolutions and return weighted average.
    Coarser scales are more noise-tolerant; finer scales add discrimination.
    """
    weights = [1.0, 0.6, 0.3]
    total_w = sum(weights[:len(scales)])
    score   = 0.0
    for (sz, w) in zip(scales, weights):
        r_t = cv2.resize(feat_t, sz, interpolation=cv2.INTER_AREA)
        r_p = cv2.resize(feat_p, sz, interpolation=cv2.INTER_AREA)
        score += w * cosine_sim(r_t, r_p)
    return score / total_w


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def norm_visual(img):
    mx = np.max(img)
    if mx < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    return (255.0 * img / mx).astype(np.uint8)

def write_colmap(poses, colmap_dir):
    os.makedirs(colmap_dir, exist_ok=True)
    images_path  = os.path.join(colmap_dir, "images.txt")
    cameras_path = os.path.join(colmap_dir, "cameras.txt")
    with open(cameras_path, 'w') as f:
        f.write("1 PINHOLE 600 600 173.20508075688778 173.20508075688778 300.0 300.0\n")
    with open(images_path, 'w') as f:
        for p in poses:
            f.write(p['colmap_line'])


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(target_image_path, rendered_folder, poses_info, fast_ssim_only=False):
    verification_dir = os.path.join(rendered_folder, "verification_normalized")
    os.makedirs(verification_dir, exist_ok=True)

    target_img = cv2.imread(target_image_path)
    if target_img is None:
        raise ValueError(f"Could not load target image: {target_image_path}")
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Pre-compute target features once
    t_amp, t_delay = robust_preprocess(target_img)
    t_amp_vis = norm_visual(t_amp)

    cv2.imwrite(os.path.join(verification_dir, "TARGET_amp.png"),   t_amp_vis)
    cv2.imwrite(os.path.join(verification_dir, "TARGET_delay.png"), norm_visual(t_delay))

    # Weight configuration for full eval (neighbor search)
    W_AMP   = 0.1
    W_DELAY = 0.3
    W_SSIM  = 0.6

    results = []
    for p in tqdm(poses_info, desc="Evaluating (Fast SSIM)" if fast_ssim_only else "Evaluating (Full)"):
        img_path = os.path.join(rendered_folder, p['name'])
        if not os.path.exists(img_path):
            continue

        pred_img = cv2.imread(img_path)
        if pred_img is None:
            continue
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB).astype(np.float32)

        p_amp, p_delay = robust_preprocess(pred_img)
        p_amp_vis = norm_visual(p_amp)
        ssim_val  = ssim(t_amp_vis, p_amp_vis, data_range=255)

        if fast_ssim_only:
            # Fast mode: only SSIM matters
            total_dist = 1.0 - max(0, ssim_val)
            results.append({
                'position':   p['position'],
                'yaw':        p['yaw'],
                'error':      float(total_dist),
                'corr_amp':   0.0,
                'corr_delay': 0.0,
                'ssim':       float(ssim_val),
                'name':       p['name'],
                'grid_idx':   p['grid_idx']
            })
        else:
            # Full evaluation
            corr_amp   = multiscale_corr(t_amp,   p_amp)
            corr_delay = multiscale_corr(t_delay, p_delay)
            
            combined_corr = W_AMP * corr_amp + W_DELAY * corr_delay + W_SSIM * max(0, ssim_val)
            total_dist    = 1.0 - combined_corr

            results.append({
                'position':   p['position'],
                'yaw':        p['yaw'],
                'error':      float(total_dist),
                'corr_amp':   float(corr_amp),
                'corr_delay': float(corr_delay),
                'ssim':       float(ssim_val),
                'name':       p['name'],
                'grid_idx':   p['grid_idx']
            })

    results.sort(key=lambda x: x['error'])

    print(f"\nTop 5 Matches ({'Fast SSIM' if fast_ssim_only else 'Full Eval'}):")
    for res in results[:5]:
        x, y, z = res['position']
        print(
            f"  Pos ({x:.1f},{y:.1f},{z:.1f}) Yaw:{res['yaw']:3d}° | "
            f"Dist:{res['error']:.4f}  "
            f"Amp:{res['corr_amp']*100:.1f}%  "
            f"Delay:{res['corr_delay']*100:.1f}%  "
            f"SSIM:{res['ssim']:.3f}"
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",    type=str,   default="output/rf_model")
    parser.add_argument("--iteration",     type=int,   default=40000)
    parser.add_argument("--target_image",  type=str,   required=True)
    parser.add_argument("--spacing",       type=float, default=0.5)
    args = parser.parse_args()

    x_bound = [0.3, 6.7]
    y_bound = [0.3, 4.7]
    z_bound = [0.5, 2.5]

    x_range = np.arange(x_bound[0], x_bound[1] + 1e-4, args.spacing)
    y_range = np.arange(y_bound[0], y_bound[1] + 1e-4, args.spacing)
    z_range = np.arange(z_bound[0], z_bound[1] + 1e-4, args.spacing)
    yaws    = [0, 120, 240]

    print(f"Grid: X={len(x_range)}  Y={len(y_range)}  Z={len(z_range)}  Yaws={len(yaws)}  "
          f"Total={len(x_range)*len(y_range)*len(z_range)*len(yaws)}")

    colmap_dir = "grid_search_colmap"
    _, poses_info = create_poses(x_range, y_range, z_range, yaws, colmap_dir)

    # 1. ANCHOR GENERATION
    anchor_stride = 2
    anchor_indices = set()
    for ix in range(0, len(x_range), anchor_stride):
        for iy in range(0, len(y_range), anchor_stride):
            for iz in range(0, len(z_range), anchor_stride):
                for itheta in range(0, len(yaws)):
                    anchor_indices.add((ix, iy, iz, itheta))
    
    anchor_poses_info = [p for p in poses_info if p['grid_idx'] in anchor_indices]
    
    rendered_folder = os.path.join(args.model_path, "custom_renders")

    def check_and_render_subset(subset_poses, label):
        os.makedirs(rendered_folder, exist_ok=True)
        missing = [p['name'] for p in subset_poses if not os.path.exists(os.path.join(rendered_folder, p['name']))]
        if missing:
            print(f"Missing {len(missing)} out of {len(subset_poses)} {label} renders. Rendering now...")
            colmap_subset_dir = "grid_search_colmap_subset"
            write_colmap(subset_poses, colmap_subset_dir)
            cmd = [
                "python", "render_custom_poses.py",
                "-m", args.model_path,
                "--iteration", str(args.iteration),
                "--colmap_dir", colmap_subset_dir,
            ]
            subprocess.run(cmd, check=True)
        else:
            print(f"All {len(subset_poses)} {label} renders found. Skipping render.")

    print("\n--- PHASE 1: FAST ANCHOR SEARCH ---")
    check_and_render_subset(anchor_poses_info, "anchor")
    anchor_results = evaluate(args.target_image, rendered_folder, anchor_poses_info, fast_ssim_only=True)

    # Group by spatial grid index to find the best yaw for each location
    spatial_best = {}
    for res in anchor_results:
        ix, iy, iz, _ = res['grid_idx']
        loc_key = (ix, iy, iz)
        if loc_key not in spatial_best or res['error'] < spatial_best[loc_key]['error']:
            spatial_best[loc_key] = res
            
    best_spatial_anchors = sorted(list(spatial_best.values()), key=lambda x: x['error'])

    # Pick top 2 spatial anchors
    top_k_anchors = 2
    best_anchors = best_spatial_anchors[:top_k_anchors]
    
    # Phase 2: Extract neighbors
    neighbor_indices = set()
    for anchor in best_anchors:
        ix, iy, iz, itheta = anchor['grid_idx']
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    for n_itheta in range(len(yaws)): # Evaluate all yaw angles
                        n_ix, n_iy, n_iz = ix + dx, iy + dy, iz + dz
                        if 0 <= n_ix < len(x_range) and 0 <= n_iy < len(y_range) and 0 <= n_iz < len(z_range):
                            neighbor_indices.add((n_ix, n_iy, n_iz, n_itheta))
                            
    neighbor_poses_info = [p for p in poses_info if p['grid_idx'] in neighbor_indices]
    
    print(f"\n--- PHASE 2: NEIGHBOR SEARCH (Top {top_k_anchors} anchors -> {len(neighbor_poses_info)} neighbors) ---")
    check_and_render_subset(neighbor_poses_info, "neighbor")
    final_results = evaluate(args.target_image, rendered_folder, neighbor_poses_info, fast_ssim_only=False)

    print("\n================= BEST COARSE POSE =================")
    if final_results:
        b  = final_results[0]
        bx, by, bz = b['position']
        print(f"Position : X={bx:.2f}  Y={by:.2f}  Z={bz:.2f}")
        print(f"Yaw      : {b['yaw']}°")
        print(f"Distance : {b['error']:.4f}")
        print(f"  Amplitude  corr : {b['corr_amp']*100:.2f}%")
        print(f"  Delay      corr : {b['corr_delay']*100:.2f}%")
        print(f"  SSIM            : {b['ssim']:.3f}")


if __name__ == "__main__":
    main()