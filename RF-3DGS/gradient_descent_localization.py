#!/usr/bin/env python3
#
# Gradient Descent-based Refinement for RF-3DGS Localization
#
# This script takes a coarse pose estimate from grid_search_localization.py
# and refines it using gradient descent on a combined L1 + SSIM loss.
#

import torch
import os
import sys
import cv2
import numpy as np
import subprocess
import re
import math
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import torchvision

# RF-3DGS imports
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.loss_utils import ssim
from utils.general_utils import safe_state


class CustomCam:
    """
    A lightweight camera class for rendering custom poses.
    """
    def __init__(self, R_mat, T_vec, FoVy, FoVx, width, height, uid):
        self.uid = uid
        self.image_width = width
        self.image_height = height
        self.FoVy = FoVy
        self.FoVx = FoVx
        self.znear = 0.01
        self.zfar = 100.0
        
        # Build world-to-view transform
        self.world_view_transform = torch.tensor(getWorld2View2(R_mat, T_vec)).transpose(0, 1).cuda()
        
        # Build projection matrix
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, 
                                                      fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        
        # Build full projection transform
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def pose_to_matrix(position, yaw_rad):
    """
    Convert position (tx, ty, tz) and yaw (in radians) to rotation matrix R and translation vector T
    suitable for world-to-camera transformation.
    """
    tx, ty, tz = position[0], position[1], position[2]
    yaw = yaw_rad
    
    # Sionna coordinate transformation
    pitch, roll = torch.tensor(0.0, device="cuda"), torch.tensor(0.0, device="cuda")
    
    # PyTorch based Euler to Rotation matrix
    # R_posz2posx = R.from_euler('ZYX', [-np.pi/2, 0.0, -np.pi/2])
    # R_posx2array = R.from_euler('ZYX', [yaw, pitch, roll])
    
    # Simple workaround since SciPy R doesn't support autograd:
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    
    # R_ZYX = R_z(yaw) * R_y(pitch) * R_x(roll)
    R_yaw = torch.stack([
        torch.stack([cy, -sy, torch.tensor(0.0, device="cuda")]),
        torch.stack([sy, cy, torch.tensor(0.0, device="cuda")]),
        torch.stack([torch.tensor(0.0, device="cuda"), torch.tensor(0.0, device="cuda"), torch.tensor(1.0, device="cuda")])
    ])
    
    # Static rotation equivalent to R.from_euler('ZYX', [-np.pi/2, 0.0, -np.pi/2])
    R_posz2posx = torch.tensor([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0]
    ], device="cuda")
    
    R_posx2array = R_yaw
    
    R_w2c = torch.inverse(torch.matmul(R_posx2array, R_posz2posx))
    
    P_world = torch.stack([tx, ty, tz])
    T_w2c = -torch.matmul(R_w2c, P_world)
    
    return R_w2c.detach().cpu().numpy(), T_w2c.detach().cpu().numpy() # this breaks gradients


def run_grid_search(target_image, model_path, iteration):
    """
    Run grid_search_localization.py and parse the output to extract coarse pose.
    Returns: (position_tuple, yaw_deg)
    """
    print("\n" + "="*60)
    print("STEP 1: Run Grid Search for Coarse Pose Estimation... (This might take a moment)")
    print("="*60)
    
    cmd = [
        sys.executable, "grid_search_localization.py",
        "--target_image", target_image,
        "--model_path", model_path,
        "--iteration", str(iteration)
    ]
    
    # Instead of fighting python's stdout pipe buffering on nested processes,
    # we just run the command natively in the shell with `tee` to show progress
    # completely properly, and write it to a temp file to parse the answer.
    tmp_out = "temp_grid_output.txt"
        
    cmd_str = f'"{sys.executable}" grid_search_localization.py --target_image "{target_image}" --model_path "{model_path}" --iteration {iteration}'
    
    process = subprocess.Popen(
        f"{cmd_str} 2>&1 | tee {tmp_out}",
        shell=True,
        executable="/bin/bash"
    )
    process.wait()
    
    with open(tmp_out, "r") as f:
        output_lines = f.readlines()
        
    # Clean up right after
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
        
    candidates = []
    for line in output_lines:
        if "Pose_0" in line and "Position:" in line:
            pos_match = re.search(r'X:\s*([\d.-]+).*Y:\s*([\d.-]+).*Z:\s*([\d.-]+)', line)
            yaw_match = re.search(r'Rotation/Yaw:\s*([\d.-]+)°', line)
            
            if pos_match and yaw_match:
                tx, ty, tz = float(pos_match.group(1)), float(pos_match.group(2)), float(pos_match.group(3))
                yaw_deg = float(yaw_match.group(1))
                candidates.insert(0, ((tx, ty, tz), yaw_deg))
                
        elif "Pos (" in line and "Yaw:" in line:
            pos_match = re.search(r'Pos \(([\d.-]+),\s*([\d.-]+),\s*([\d.-]+)\)\s+Yaw:\s*([\d.-]+)°', line)
            if pos_match:
                tx, ty, tz = float(pos_match.group(1)), float(pos_match.group(2)), float(pos_match.group(3))
                yaw_deg = float(pos_match.group(4))
                candidates.append(((tx, ty, tz), yaw_deg))
                
    if len(candidates) == 0:
        raise RuntimeError(f"Could not parse grid search output. Unexpected format.")
        
    print("\n" + "="*60)
    print(f"STEP 2: ACQUIRED {len(candidates)} CANDIDATE POSES FROM GRID SEARCH")
    print("="*60)
    for i, c in enumerate(candidates):
        print(f"  Candidate {i+1}: X={c[0][0]:.4f}m, Y={c[0][1]:.4f}m, Z={c[0][2]:.4f}m, Yaw={c[1]:.2f}°")
    print("="*60)
    
    return candidates


def load_target_image(target_image_path, width, height):
    img = cv2.imread(target_image_path)
    if img is None:
        raise ValueError(f"Could not load target image: {target_image_path}")
    
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
    return img_tensor


def compute_loss(rendered, target, lambda_weight=0.7):
    l1 = torch.abs(target - rendered).mean()
    ssim_value = ssim(rendered, target, window_size=11, size_average=True)
    loss = (1.0 - lambda_weight) * l1 + lambda_weight * (1.0 - ssim_value)
    
    return loss



def load_cameras_txt(camera_file):
    cameras = {}
    if not os.path.exists(camera_file):
        return cameras
    with open(camera_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            if model == "PINHOLE":
                fx = float(parts[4])
                fy = float(parts[5])
                fovx = 2.0 * np.arctan(width / (2.0 * fx))
                fovy = 2.0 * np.arctan(height / (2.0 * fy))
                cameras[cam_id] = {'width': width, 'height': height, 'fovx': fovx, 'fovy': fovy}
    return cameras

from scipy.optimize import minimize

def extract_gt_from_filename(filename):
    base = os.path.basename(filename)
    floats = re.findall(r'-?\d+\.\d+', base)
    if len(floats) >= 3:
        return np.array([float(floats[0]), float(floats[1]), float(floats[2])])
    return None

def iterative_optimization_refinement(gaussians, pipeline, background, target_image_tensor, 
                                 coarse_candidates, yaws_to_test=[0, 120, 240], num_iterations=100, 
                                 lambda_weight=0.8, fovx=0.8, fovy=0.8, width=600, height=600):
                                 
    print("\n" + "="*60)
    print("STEP 4: Deep Verification & Iterative Direction-Based Optimization")
    print("="*60 + "\n")
    
    loss_history = []
    
    def evaluate_loss_at(x, y, z, yaw_deg):
        yaw = np.radians(yaw_deg)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R_yaw = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device="cuda")
        R_posz2posx = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float32, device="cuda")
        R_w2c = torch.inverse(torch.matmul(R_yaw, R_posz2posx))
        P_world = torch.tensor([x, y, z], dtype=torch.float32, device="cuda")
        T_w2c = -torch.matmul(R_w2c, P_world)
        
        wv_trans = torch.zeros((4, 4), device="cuda")
        wv_trans[:3, :3] = R_w2c.transpose(0, 1)
        wv_trans[3, :3] = T_w2c
        wv_trans[3, 3] = 1.0
        
        znear, zfar = 0.01, 100.0
        proj_mat = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy)
        if isinstance(proj_mat, torch.Tensor):
            projection_matrix = proj_mat.clone().detach().to("cuda").transpose(0, 1)
        else:
            projection_matrix = torch.tensor(proj_mat, device="cuda").transpose(0, 1)
        
        full_proj_transform = torch.matmul(wv_trans, projection_matrix)
        camera_center = torch.inverse(wv_trans)[3, :3]
        
        class DiffCam: pass
        cam = DiffCam()
        cam.image_width, cam.image_height = width, height
        cam.FoVy, cam.FoVx = fovy, fovx
        cam.znear, cam.zfar = znear, zfar
        cam.world_view_transform = wv_trans
        cam.full_proj_transform = full_proj_transform
        cam.camera_center = camera_center
        
        with torch.no_grad():
            rend = render(cam, gaussians, pipeline, background)["render"]
            loss = compute_loss(rend.unsqueeze(0), target_image_tensor, lambda_weight=lambda_weight)
        
        return loss.item()

    MAX_DIRECTION_CHECKS = 7
    MIN_CONSECUTIVE_SAME_YAW = 5

    print(f"--- Phase 1: Exploring Best Angle and Error Basin for each Candidate ---")
    
    best_candidate_post_phase1_loss = float('inf')
    best_candidate_post_phase1_pos = None
    best_candidate_post_phase1_yaw = None
    
    for i, (pos, yaw) in enumerate(coarse_candidates):
        print(f"\n  [Candidate {i+1}]: Initial Coords={pos}, Yaw={yaw}°")
        current_pos = np.array(pos)
        best_yaw_history = []
        best_loss = float('inf')
        best_yaw = yaw
        
        for step in range(MAX_DIRECTION_CHECKS):
            best_loss = float('inf')
            best_yaw = 0
            
            for y in yaws_to_test:
                l = evaluate_loss_at(current_pos[0], current_pos[1], current_pos[2], y)
                if l < best_loss:
                    best_loss = l
                    best_yaw = y
                    
            best_yaw_history.append(best_yaw)
            
            # Check if stabilized
            if len(best_yaw_history) >= MIN_CONSECUTIVE_SAME_YAW and len(set(best_yaw_history[-MIN_CONSECUTIVE_SAME_YAW:])) == 1:
                break
                
            def step_objective(p):
                return evaluate_loss_at(p[0], p[1], p[2], best_yaw)
                
            res = minimize(step_objective, current_pos, method='Nelder-Mead', options={'maxiter': 5, 'maxfev': 15})
            current_pos = res.x
            
        print(f"    -> After basin exploration: Coords=[{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}], Best Yaw={best_yaw}°, Deep Loss={best_loss:.6f}")
        
        if best_loss < best_candidate_post_phase1_loss:
            best_candidate_post_phase1_loss = best_loss
            best_candidate_post_phase1_pos = current_pos
            best_candidate_post_phase1_yaw = best_yaw

    print(f"\n  --> WINNER CANDIDATE FOR PHASE 2: Coords=[{best_candidate_post_phase1_pos[0]:.4f}, {best_candidate_post_phase1_pos[1]:.4f}, {best_candidate_post_phase1_pos[2]:.4f}], Yaw={best_candidate_post_phase1_yaw}°, Loss={best_candidate_post_phase1_loss:.6f}")

    current_pos = np.array(best_candidate_post_phase1_pos)
    final_target_yaw = best_candidate_post_phase1_yaw

    print(f"\n--- Phase 2: Full Rx Position Optimization (Fixed at Dir {final_target_yaw}°) ---")
    pbar = tqdm(total=num_iterations, desc="Optimization")
    it_count = [0]
    
    def final_objective(pos):
        l = evaluate_loss_at(pos[0], pos[1], pos[2], final_target_yaw)
        loss_history.append(l)
        it_count[0] += 1
        if it_count[0] <= num_iterations:
            pbar.update(1)
            pbar.set_postfix({"loss": f"{l:.4f}", "x": f"{pos[0]:.3f}", "y": f"{pos[1]:.3f}", "z": f"{pos[2]:.3f}"})
        return l
        
    final_res = minimize(final_objective, current_pos, method='Nelder-Mead', options={'maxiter': num_iterations, 'maxfev': num_iterations})
    pbar.close()
    
    optimized_position = final_res.x
    
    return optimized_position, final_target_yaw, loss_history

def main():

    parser = ArgumentParser(description="Gradient Descent Localization Refinement for RF-3DGS")
    parser.add_argument("--target_image", type=str, required=True)
    # model_path is implicitly added by ModelParams
    # parser.add_argument("--iteration", type=int, default=40000) # Often implicit too, let's keep it safe by trying to let pipeline grab it
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--lambda_weight", type=float, default=0.6)
    
    # In some forks explicit iteration is added, but ModelParams usually has it or pipeline.
    # We will use parse_known_args
    
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=40000)
    
    args = get_combined_args(parser)
    # safe_state(silent=True) # removed to prevent timestamp hijacking of stdout
    
    if not os.path.exists(args.target_image):
        raise FileNotFoundError(f"Target image not found: {args.target_image}")
    
    coarse_candidates = run_grid_search(args.target_image, args.model_path, args.iteration)
    
    print("\n" + "="*60)
    print("STEP 3: Loading RF-3DGS Scene and Model for Gradient Descent...")
    print("="*60)
    
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Load camera parameters from colmap explicitly
    cam_file = "grid_search_colmap/cameras.txt"
    cameras = load_cameras_txt(cam_file)
    cam_id = 1
    if cam_id in cameras:
        fovx = cameras[cam_id]['fovx']
        fovy = cameras[cam_id]['fovy']
        width = cameras[cam_id]['width']
        height = cameras[cam_id]['height']
    else:
        print("Warning: cameras.txt not found in grid_search_colmap. Using default 120-deg parameters.")
        fovx, fovy = 2.0943951, 2.0943951  # roughly 120 degrees
        width, height = 600, 600
        
    target_tensor = load_target_image(args.target_image, width, height)
    
    gt_pos = extract_gt_from_filename(args.target_image)
    
    with torch.no_grad():
        optimized_position, optimized_yaw, loss_history = iterative_optimization_refinement(
            gaussians, pipeline, background, target_tensor,
            coarse_candidates, yaws_to_test=[0, 120, 240], num_iterations=args.num_iterations,
            lambda_weight=args.lambda_weight, fovx=fovx, fovy=fovy, width=width, height=height
        )
        coarse_pose, coarse_yaw = coarse_candidates[0] # just for diff logs
    
    print("\n" + "="*80)
    print("STEP 5: FINAL OPTIMIZED RECEIVER POSE")
    print("="*80)
    print(f"\nOptimized Position:")
    print(f"  X: {optimized_position[0]:.6f} m")
    print(f"  Y: {optimized_position[1]:.6f} m")
    print(f"  Z: {optimized_position[2]:.6f} m")
    print(f"\nOptimized Yaw: {optimized_yaw:.6f}°")
    
    if gt_pos is not None:
        dist_error = np.sqrt(np.sum((optimized_position - gt_pos)**2))
        print(f"\n=== ACCURACY METRICS ===")
        print(f"  Real Position (Ground Truth): X={gt_pos[0]:.4f}m, Y={gt_pos[1]:.4f}m, Z={gt_pos[2]:.4f}m")
        print(f"  Distance from Real Position (Error): {dist_error:.6f} meters")
    else:
        print(f"\n[!] Ground Truth position could not be extracted from filename.")
    
    print(f"\n--- Coarse-to-Fine Refinement Delta ---")
    print(f"  ΔX: {(optimized_position[0] - coarse_pose[0]):.4f} m")
    print(f"  ΔY: {(optimized_position[1] - coarse_pose[1]):.4f} m")
    print(f"  ΔZ: {(optimized_position[2] - coarse_pose[2]):.4f} m")
    print(f"  ΔYaw: {(optimized_yaw - coarse_yaw):.6f}°")
    
    print(f"\n--- Optimization Statistics ---")
    print(f"  Initial Loss: {loss_history[0]:.6f}")
    print(f"  Final Loss:   {loss_history[-1]:.6f}")
    if loss_history[0] > 1e-8:
        reduction_pct = ((loss_history[0] - loss_history[-1]) / loss_history[0] * 100)
        print(f"  Loss Reduction: {reduction_pct:.2f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
