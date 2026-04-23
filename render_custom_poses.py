import torch
import os
import numpy as np
import torchvision
from argparse import ArgumentParser
from tqdm import tqdm
import sys
from os import makedirs

# Add RF-3DGS to path since we moved the script out
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "RF-3DGS"))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scene.colmap_loader import qvec2rotmat

class CustomCam:
    """
    A lightweight camera class to use for rendering custom poses.
    """
    def __init__(self, R, T, FoVy, FoVx, width, height, uid):
        self.uid = uid
        self.image_width = width
        self.image_height = height
        self.FoVy = FoVy
        self.FoVx = FoVx
        self.znear = 0.01
        self.zfar = 100.0
        
        # Build world-to-view transform
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        
        # Build projection matrix
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        
        # Build full projection transform
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def load_cameras_txt(camera_file):
    """
    Load camera intrinsics from a COLMAP-like cameras.txt.
    Returns a dictionary mapping camera_id to its properties.
    """
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
            
            # PINHOLE format: fx, fy, cx, cy
            if model == "PINHOLE":
                fx = float(parts[4])
                fy = float(parts[5])
                # Calculate FOV from focal length
                fovx = 2.0 * np.arctan(width / (2.0 * fx))
                fovy = 2.0 * np.arctan(height / (2.0 * fy))
                cameras[cam_id] = {'width': width, 'height': height, 'fovx': fovx, 'fovy': fovy}
    return cameras

def load_poses(pose_file):
    """
    Load a list of poses from a text file formatted like COLMAP's images.txt,
    but handling the user's custom 8-element format:
    IMAGE_ID QW QX QY QZ TX TY TZ
    or the standard 10-element format:
    IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    """
    poses = []
    render_idx = 0
    with open(pose_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or comments
            if not line or line.startswith("#"):
                continue
            
            parts = line.split()
            if len(parts) >= 8:
                try:
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    
                    # Optional camera_id and name
                    cam_id = int(parts[8]) if len(parts) > 8 else 1
                    name = parts[9] if len(parts) > 9 else f"{render_idx:05d}.png"
                    
                    qvec = np.array([qw, qx, qy, qz])
                    tvec = np.array([tx, ty, tz])
                    
                    R = np.transpose(qvec2rotmat(qvec))
                    T = tvec
                    
                    poses.append((R, T, name, cam_id))
                    render_idx += 1
                except ValueError:
                    continue # Skip lines like 2D point observations
    return poses

def render_custom(dataset: ModelParams, pipeline: PipelineParams, iteration: int, colmap_dir: str, default_fov: float, default_width: int, default_height: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        
        print("Initializing Scene and loading model...")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out_dir = os.path.join(dataset.model_path, "custom_renders")
        makedirs(out_dir, exist_ok=True)
        
        pose_file = os.path.join(colmap_dir, "images.txt")
        camera_file = os.path.join(colmap_dir, "cameras.txt")

        poses = load_poses(pose_file)
        cameras = load_cameras_txt(camera_file)
        
        if len(poses) == 0:
            print("No valid poses found in the file.")
            return

        print(f"Rendering {len(poses)} custom poses to {out_dir}")
        for idx, (R, T, name, cam_id) in enumerate(tqdm(poses, desc="Rendering custom poses")):
            # Get intrinsics from cameras.txt or fallback to defaults
            cam_data = cameras.get(cam_id, {
                'fovx': default_fov, 'fovy': default_fov, 
                'width': default_width, 'height': default_height
            })
            
            # Create a custom camera for this pose
            cam = CustomCam(R, T, FoVy=cam_data['fovy'], FoVx=cam_data['fovx'], 
                            width=cam_data['width'], height=cam_data['height'], uid=idx)

            rendering = render(cam, gaussians, pipeline, background)["render"]
            
            # Use original filename if possible, otherwise zero-padded index
            filename = name if name.endswith(".png") or name.endswith(".jpg") else f"{idx:05d}.png"
            out_path = os.path.join(out_dir, filename)
            torchvision.utils.save_image(rendering, out_path)

if __name__ == "__main__":
    parser = ArgumentParser(description="Render custom poses script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--colmap_dir", type=str, required=True, help="Path to directory containing images.txt and cameras.txt")
    parser.add_argument("--fov", type=float, default=0.8, help="Fallback field of view (radians) if camera_file not found")
    parser.add_argument("--width", type=int, default=800, help="Fallback image width if camera_file not found")
    parser.add_argument("--height", type=int, default=800, help="Fallback image height if camera_file not found")
    
    args = get_combined_args(parser)
    
    render_custom(model.extract(args), pipeline.extract(args), args.iteration, args.colmap_dir, args.fov, args.width, args.height)
