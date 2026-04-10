import os
import subprocess
import argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def create_poses(x_range, y_range, z_range, yaws, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    images_path = os.path.join(out_dir, "images.txt")
    cameras_path = os.path.join(out_dir, "cameras.txt")
    
    with open(cameras_path, 'w') as f:
        f.write("1 PINHOLE 600 600 173.20508075688778 173.20508075688778 300.0 300.0\n")
    
    poses_info = []
    with open(images_path, 'w') as f:
        img_id = 1
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    for theta in yaws:
                        yaw, pitch, roll = np.radians(theta), 0.0, 0.0
                        R_posz2posx = R.from_euler('ZYX', [-np.pi/2, 0.0, -np.pi/2])
                        R_posx2array = R.from_euler('ZYX', [yaw, pitch, roll])
                        
                        R_w2c = (R_posx2array * R_posz2posx).inv().as_matrix()
                        P_world = np.array([x, y, z])
                        T_w2c = -R_w2c @ P_world
                        
                        quat = R.from_matrix(R_w2c).as_quat() 
                        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
                        
                        name = f"pos_X{x:.1f}_Y{y:.1f}_Z{z:.1f}_Yaw{theta}.png"
                        f.write(f"{img_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                                f"{T_w2c[0]:.8f} {T_w2c[1]:.8f} {T_w2c[2]:.8f} 1 {name}\n")
                        
                        poses_info.append({
                            'id': img_id,
                            'name': name,
                            'position': (x, y, z),
                            'yaw': theta
                        })
                        img_id += 1
                        
    return out_dir, poses_info

def compute_correlation(img1, img2):
    """
    Compute Cosine Similarity (Normalized Cross Correlation).
    Returns value between 0.0 and 1.0. Higher is better overlap.
    """
    flat1 = img1.ravel()
    flat2 = img2.ravel()
    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    return np.dot(flat1, flat2) / (norm1 * norm2)

def norm_visual(img):
    if np.max(img) < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    return (255.0 * img / np.max(img)).astype(np.uint8)

def evaluate(target_image_path, rendered_folder, poses_info):
    verification_dir = os.path.join(rendered_folder, "verification_normalized")
    os.makedirs(verification_dir, exist_ok=True)
    
    target_img = cv2.imread(target_image_path)
    if target_img is None:
        raise ValueError(f"Could not load target image: {target_image_path}")
        
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    # R = delay*amp, G = amp
    # It's better to separate amplitude and delay
    target_amp = target_img[:, :, 1]
    target_delay = np.zeros_like(target_amp)
    valid = target_amp > 1.0
    target_delay[valid] = target_img[:, :, 0][valid] / target_amp[valid]
    
    # Apply a heavy blur so that grid-points don't require perfect sub-pixel alignment!
    blur_k = (91, 91)
    sigma = 20
    t_amp_blur = cv2.GaussianBlur(target_amp, blur_k, sigma)
    t_delay_blur = cv2.GaussianBlur(target_delay, blur_k, sigma)
    
    cv2.imwrite(os.path.join(verification_dir, "TARGET_amp_blur.png"), norm_visual(t_amp_blur))
    cv2.imwrite(os.path.join(verification_dir, "TARGET_delay_blur.png"), norm_visual(t_delay_blur))
    
    results = []
    
    for p in tqdm(poses_info, desc="Evaluating Rendered Images against Target"):
        img_name = p['name']
        img_path = os.path.join(rendered_folder, img_name)
        if not os.path.exists(img_path): continue
            
        pred_img = cv2.imread(img_path)
        if pred_img is None: continue
            
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        pred_amp = pred_img[:, :, 1] 
        pred_delay = np.zeros_like(pred_amp)
        v_pred = pred_amp > 1.0
        pred_delay[v_pred] = pred_img[:, :, 0][v_pred] / pred_amp[v_pred]
        
        p_amp_blur = cv2.GaussianBlur(pred_amp, blur_k, sigma)
        p_delay_blur = cv2.GaussianBlur(pred_delay, blur_k, sigma)
        
        # We want to MAXIMIZE correlation, so distance = 1 - correlation
        corr_amp = compute_correlation(t_amp_blur, p_amp_blur)
        corr_delay = compute_correlation(t_delay_blur, p_delay_blur)
        
        dist_amp = 1.0 - corr_amp
        dist_delay = 1.0 - corr_delay
        
        # Save one sample visual just so we know what prediction looks like
        if dist_amp < 0.99:
            cv2.imwrite(os.path.join(verification_dir, f"PRED_amp_{img_name}"), norm_visual(p_amp_blur))
        
        # total distance is what we want to minimize
        total_dist = dist_amp + dist_delay
        
        results.append({
            'position': p['position'],
            'yaw': p['yaw'],
            'error': float(total_dist),
            'corr_amp': float(corr_amp),
            'corr_delay': float(corr_delay),
            'name': img_name
        })
        
    results.sort(key=lambda x: x['error'])
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="output/rf_model", help="Path to trained RF-3DGS model")
    parser.add_argument("--iteration", type=int, default=40000, help="Model iteration to render")
    parser.add_argument("--target_image", type=str, required=True, help="Path to the target test image")
    parser.add_argument("--spacing", type=float, default=0.5, help="Grid spatial spacing in meters")
    args = parser.parse_args()
    
    x_bound = [0.3, 6.7]
    y_bound = [0.3, 4.7]
    z_bound = [0.5, 2.5]
    
    x_range = np.arange(x_bound[0], x_bound[1] + 1e-4, args.spacing)
    y_range = np.arange(y_bound[0], y_bound[1] + 1e-4, args.spacing)
    z_range = np.arange(z_bound[0], z_bound[1] + 1e-4, args.spacing)
    yaws = [0, 120, 240]
    
    print(f"Creating Grid: X ({len(x_range)}), Y ({len(y_range)}), Z ({len(z_range)})")
    colmap_dir = "grid_search_colmap"
    _, poses_info = create_poses(x_range, y_range, z_range, yaws, colmap_dir)
    
    rendered_folder = os.path.join(args.model_path, "custom_renders")
    
    needs_render = True
    if os.path.exists(rendered_folder):
        if any(f.endswith('.png') for f in os.listdir(rendered_folder) if not f.startswith("TARGET")):
            needs_render = False
    
    if needs_render:
        print(f"Starting batch rendering of {len(poses_info)} grid locations...")
        cmd = ["python", "render_custom_poses.py", "-m", args.model_path, "--iteration", str(args.iteration), "--colmap_dir", colmap_dir]
        subprocess.run(cmd, check=True)
    else:
        print(f"Render outputs found. Skipping rendering phase...")
        
    results = evaluate(args.target_image, rendered_folder, poses_info)
    
    print("\n================= BEST COARSE POSE FOUND =================")
    if len(results) > 0:
        best_match = results[0]
        best_pos = best_match['position']
        best_yaw = best_match['yaw']
        print(f"Pose_0 = (Position: (X: {best_pos[0]:.2f}, Y: {best_pos[1]:.2f}, Z: {best_pos[2]:.2f}), Rotation/Yaw: {best_yaw}°)")
        print(f"Lowest Distance (1-Corr): {best_match['error']:.4f}")
        print(f"   ∟ Amplitude Correlation: {best_match['corr_amp']*100:.2f}%")
        print(f"   ∟ Delay Correlation:     {best_match['corr_delay']*100:.2f}%")
        print("\nTop 3 Alternatives:")
        for res in results[1:4]:
            print(f"Pos ({res['position'][0]:.1f}, {res['position'][1]:.1f}, {res['position'][2]:.1f}) Yaw: {res['yaw']}° -> Dist: {res['error']:.4f} (Amp {res['corr_amp']*100:.1f}%)")
    
if __name__ == "__main__":
    main()
