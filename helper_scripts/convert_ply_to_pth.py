import torch
import os
import sys
from argparse import ArgumentParser
# Add RF-3DGS repo to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
rf3dgs_dir = os.path.join(project_dir, "RF-3DGS")
sys.path.insert(0, rf3dgs_dir)

from scene.gaussian_model import GaussianModel
from arguments import OptimizationParams

def convert(args):
    print(f"Loading PLY from {args.ply}")
    
    # 1. Initialize Gaussian Model
    gaussians = GaussianModel(sh_degree=3)
    
    # 2. Setup dummy training parameters to initialize optimizer (needed for capture)
    # We use default params, it doesn't matter much as we just want a valid structure
    parser = ArgumentParser()
    opt_params = OptimizationParams(parser)
    # Extract defaults
    opt = opt_params.extract(parser.parse_args([]))
    
    # Setup training (creates optimizer)
    gaussians.training_setup(opt)
    
    # 3. Load the PLY data into the model (overwriting initialized params)
    gaussians.load_ply(args.ply)
    
    # 4. Capture state (includes the geometry from PLY and the fresh optimizer state)
    iteration = 30000
    save_tuple = (gaussians.capture(), iteration)
    
    # 5. Save .pth
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(save_tuple, args.out)
    print(f"Saved checkpoint to {args.out}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ply", required=True, help="Path to input point_cloud.ply")
    parser.add_argument("--out", required=True, help="Path to output .pth checkpoint")
    args = parser.parse_args()
    
    convert(args)
