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
import time
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
from lpipsPyTorch import LPIPS


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
    
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    
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
    
    return R_w2c, T_w2c


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
    
    # Instead of writing to a temp file, we read from stdout pipe directly line by line.
    cmd_str = [sys.executable, "grid_search_localization.py", "--target_image", target_image, "--model_path", model_path, "--iteration", str(iteration)]
    
    process = subprocess.Popen(
        cmd_str,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    output_lines = []
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
        
    process.wait()
            
    # The actual top matches will be populated from the final list or from the Best coarse pose print
    # Let's cleanly grab the new "Top 5 Matches" output
    
    candidates = []
    in_top_5 = False
    for line in output_lines:
        if "Top 5 Matches (Full Eval):" in line or "Top 5 Matches:" in line:
            in_top_5 = True
            continue
        elif in_top_5:
            if "Pos (" in line and "Yaw:" in line:
                pos_match = re.search(r'Pos \(([\d.-]+),\s*([\d.-]+),\s*([\d.-]+)\)\s+Yaw:\s*([\d.-]+)°', line)
                if pos_match:
                    tx, ty, tz = float(pos_match.group(1)), float(pos_match.group(2)), float(pos_match.group(3))
                    yaw_deg = float(pos_match.group(4))
                    candidates.append(((tx, ty, tz), yaw_deg))
            elif line.strip() == "" or "BEST COARSE POSE" in line:
                in_top_5 = False
                
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


def compute_loss(rendered, target, lpips_net, lambda_weight=0.6):
    lpips_val = lpips_net(rendered, target).mean()
    ssim_value = ssim(rendered, target, window_size=11, size_average=True)
    # L1 replaced with LPIPS. 0.4 * LPIPS + 0.6 * SSIM (in loss form: 1 - SSIM)
    loss = (1.0 - lambda_weight) * lpips_val + lambda_weight * (1.0 - ssim_value)
    
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
                                 lambda_weight=0.6, fovx=0.8, fovy=0.8, width=600, height=600,
                                 learning_rate=0.01):
                                 
    print("\n" + "="*60)
    print("STEP 4: Optimization Refinement")
    print("="*60 + "\n")
    
    loss_history = []
    
    best_candidate_post_phase1_pos, best_candidate_post_phase1_yaw = coarse_candidates[0]
    
    pos_tensor = torch.tensor(best_candidate_post_phase1_pos, dtype=torch.float32, device="cuda")
    x = torch.nn.Parameter(pos_tensor[0])
    y = torch.nn.Parameter(pos_tensor[1])
    z = torch.nn.Parameter(pos_tensor[2])
    
    optimizer = torch.optim.Adam([
        {'params': x, 'lr': learning_rate},
        {'params': y, 'lr': learning_rate},
        {'params': z, 'lr': learning_rate}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)
    
    final_target_yaw = best_candidate_post_phase1_yaw
    yaw_tensor = torch.tensor(np.radians(final_target_yaw), dtype=torch.float32, device="cuda")
    
    lpips_net = LPIPS(net_type='alex').cuda()
    
    # Pre-compute static matrices outside the evaluation loop for significant speedup
    cy = torch.cos(yaw_tensor)
    sy = torch.sin(yaw_tensor)
    R_yaw = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], device="cuda")
    R_posz2posx = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], device="cuda")
    R_w2c_static = torch.inverse(torch.matmul(R_yaw, R_posz2posx))
    R_w2c_T = R_w2c_static.transpose(0, 1).contiguous()
    
    znear, zfar = 0.01, 100.0
    proj_mat = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy)
    if isinstance(proj_mat, torch.Tensor):
        projection_matrix = proj_mat.clone().detach().to("cuda").transpose(0, 1).contiguous()
    else:
        projection_matrix = torch.tensor(proj_mat, device="cuda").transpose(0, 1).contiguous()

    base_wv_trans = torch.zeros((4, 4), device="cuda")
    base_wv_trans[:3, :3] = R_w2c_T
    base_wv_trans[3, 3] = 1.0

    class DiffCam: pass
    shared_cam = DiffCam()
    shared_cam.image_width, shared_cam.image_height = width, height
    shared_cam.FoVy, shared_cam.FoVx = fovy, fovx
    shared_cam.znear, shared_cam.zfar = znear, zfar
    
    def evaluate_loss_fw(x_val, y_val, z_val):
        P_world = torch.stack([x_val, y_val, z_val])
        T_w2c = -torch.matmul(R_w2c_static, P_world)
        
        wv_trans = base_wv_trans.clone()
        wv_trans[3, :3] = T_w2c
        
        shared_cam.world_view_transform = wv_trans
        shared_cam.full_proj_transform = torch.matmul(wv_trans, projection_matrix)
        shared_cam.camera_center = P_world
        
        with torch.no_grad():
            rend = render(shared_cam, gaussians, pipeline, background)["render"]
            loss = compute_loss(rend.unsqueeze(0), target_image_tensor, lpips_net, lambda_weight=lambda_weight)
        return loss

    print(f"\n--- Full Rx Position Optimization (Fixed at Dir {final_target_yaw}°) ---")
    
    num_iterations = min(num_iterations, 100)
    pbar = tqdm(total=num_iterations, desc="Optimization")
    
    eps = 1e-3
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    min_delta = 1e-4

    for it in range(num_iterations):
        optimizer.zero_grad()
        
        l_center = evaluate_loss_fw(x, y, z)
        
        l_x_pos = evaluate_loss_fw(x + eps, y, z)
        l_y_pos = evaluate_loss_fw(x, y + eps, z)
        l_z_pos = evaluate_loss_fw(x, y, z + eps)
        
        x.grad = (l_x_pos - l_center) / eps
        y.grad = (l_y_pos - l_center) / eps
        z.grad = (l_z_pos - l_center) / eps
        
        optimizer.step()
        scheduler.step()
        
        l_item = l_center.item()
        loss_history.append(l_item)
        
        pbar.update(1)
        pbar.set_postfix({"loss": f"{l_item:.4f}", "x": f"{x.item():.3f}", "y": f"{y.item():.3f}", "z": f"{z.item():.3f}"})
        
        if l_item < best_loss - min_delta:
            best_loss = l_item
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at iteration {it+1}: loss hasn't decreased significantly in {patience} steps.")
            break
        
    pbar.close()
    
    optimized_position = np.array([x.item(), y.item(), z.item()])
    
    return optimized_position, final_target_yaw, loss_history

def main():

    parser = ArgumentParser(description="Gradient Descent Localization Refinement for RF-3DGS")
    parser.add_argument("--target_image", type=str, required=True)
    # model_path is implicitly added by ModelParams
    # parser.add_argument("--iteration", type=int, default=40000) # Often implicit too, let's keep it safe by trying to let pipeline grab it
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--lambda_weight", type=float, default=0.6)
    parser.add_argument("--coarse_x", type=float, default=None, help="Optional coarse X position")
    parser.add_argument("--coarse_y", type=float, default=None, help="Optional coarse Y position")
    parser.add_argument("--coarse_z", type=float, default=None, help="Optional coarse Z position")
    parser.add_argument("--coarse_yaw", type=float, default=0.0, help="Optional coarse Yaw (degrees)")
    
    # In some forks explicit iteration is added, but ModelParams usually has it or pipeline.
    # We will use parse_known_args
    
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=40000)
    
    args = get_combined_args(parser)
    # safe_state(silent=True) # removed to prevent timestamp hijacking of stdout
    
    if not os.path.exists(args.target_image):
        raise FileNotFoundError(f"Target image not found: {args.target_image}")
    
    coarse_x = getattr(args, 'coarse_x', None)
    coarse_y = getattr(args, 'coarse_y', None)
    coarse_z = getattr(args, 'coarse_z', None)
    coarse_yaw = getattr(args, 'coarse_yaw', 0.0)

    if coarse_x is not None and coarse_y is not None and coarse_z is not None:
        print("\n" + "="*60)
        print("STEP 1 & 2: Using provided coarse pose. Skipping grid search.")
        print("="*60)
        coarse_candidates = [((coarse_x, coarse_y, coarse_z), coarse_yaw)]
    else:
        coarse_candidates = run_grid_search(args.target_image, args.model_path, args.iteration)
    
    print("\n" + "="*60)
    print("STEP 3: Loading RF-3DGS Scene and Model for Gradient Descent...")
    print("="*60)
    print("\nLoading Gaussian model from iteration {}".format(args.iteration))
    gaussians = GaussianModel(args.sh_degree)
    
    model_path = args.model_path
    iteration = args.iteration
    point_cloud_path = os.path.join(model_path, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply")
    gaussians.load_ply(point_cloud_path)
    
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
    
    start_time = time.time()
    optimized_position, optimized_yaw, loss_history = iterative_optimization_refinement(
        gaussians, pipeline, background, target_tensor,
        coarse_candidates, yaws_to_test=[0, 120, 240], num_iterations=args.num_iterations,
        lambda_weight=args.lambda_weight, fovx=fovx, fovy=fovy, width=width, height=height,
        learning_rate=args.learning_rate
    )
    refinement_time = time.time() - start_time
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
    
    print(f"\n--- Optimization Statistics ---")
    print(f"  Initial Loss: {loss_history[0]:.6f}")
    print(f"  Final Loss:   {loss_history[-1]:.6f}")
    if loss_history[0] > 1e-8:
        reduction_pct = ((loss_history[0] - loss_history[-1]) / loss_history[0] * 100)
        print(f"  Loss Reduction: {reduction_pct:.2f}%")
    print(f"  Refinement Time: {refinement_time:.2f} seconds")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
