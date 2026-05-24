import os
import shutil
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation
import random
import csv
from math import pi
import argparse

# Output directories
base_dir = "rfdt_based_nn/dataset"
rf3dgs_img_dir = os.path.join(base_dir, "rf3dgs/images")
rf3dgs_labels_file = os.path.join(base_dir, "rf3dgs/labels.csv")

os.makedirs(rf3dgs_img_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, "sionna/images"), exist_ok=True)

# Grid parameters
X_MIN, X_MAX, X_STEP = 0.4, 6.6, 0.25
Y_MIN, Y_MAX, Y_STEP = 0.4, 4.6, 0.25
Z_MIN, Z_MAX, Z_STEP = 1.2, 2.5, (2.5-1.2)/8.0  # 9 points roughly
YAW_STEPS = 12 # every 30 degrees
CHUNK_SIZE = 5000

# Jitter 
JITTER_MAG = 0.1

def euler_to_quaternion(euler):
    R_posz2posx = Rotation.from_euler('ZYX', [-np.pi/2,0.0,-np.pi/2])
    yaw, pitch, roll = euler 
    R_posx2array = Rotation.from_euler('ZYX',[yaw, pitch, roll]) 
    R_w2c = R_posx2array * R_posz2posx
    R_c2w = R_w2c.inv()
    q = R_c2w.as_quat()
    return R_c2w, np.array([q[3], q[0], q[1], q[2]]) # [w, x, y, z]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a small test generation (10 poses)")
    parser.add_argument("--rf_model", type=str, default="output/rf_model_delay_3.5ghz")
    args = parser.parse_args()

    xs = np.arange(X_MIN, X_MAX + X_STEP/2, X_STEP)
    ys = np.arange(Y_MIN, Y_MAX + Y_STEP/2, Y_STEP)
    zs = np.linspace(Z_MIN, Z_MAX, 9)
    yaws = np.linspace(0, 2*pi, YAW_STEPS, endpoint=False)
    
    poses = []
    
    print(f"Grid: X({len(xs)}) Y({len(ys)}) Z({len(zs)}) Yaws({len(yaws)})")
    
    for x in xs:
        for y in ys:
            for z in zs:
                for yaw in yaws:
                    # Apply jitter
                    jx = x + random.uniform(-JITTER_MAG, JITTER_MAG)
                    jy = y + random.uniform(-JITTER_MAG, JITTER_MAG)
                    jz = z + random.uniform(-JITTER_MAG, JITTER_MAG)
                    
                    # Ensure within bounds
                    jx = np.clip(jx, X_MIN, X_MAX)
                    jy = np.clip(jy, Y_MIN, Y_MAX)
                    jz = np.clip(jz, Z_MIN, Z_MAX)
                    
                    poses.append((jx, jy, jz, yaw))

    if args.test:
        poses = poses[:10]
        print("Running in TEST mode: limited to 10 poses.")

    print(f"Total poses to render: {len(poses)}")
    
    # Write empty labels if not exists or override
    with open(rf3dgs_labels_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'x', 'y', 'z', 'yaw_deg'])
    
    tmp_colmap = "rfdt_based_nn/tmp_colmap"
    os.makedirs(tmp_colmap, exist_ok=True)
    
    # Save a dummy cameras.txt
    with open(os.path.join(tmp_colmap, 'cameras.txt'), 'w') as f:
        f.write("1 PINHOLE 600 600 300 300 300 300\n")
        
    model_path = args.rf_model
    out_dir = os.path.join(model_path, "custom_renders")
    
    global_idx = 0
    
    for chunk_start in range(0, len(poses), CHUNK_SIZE):
        chunk_poses = poses[chunk_start:chunk_start+CHUNK_SIZE]
        
        # Prepare colmap images.txt for this chunk
        images_txt = os.path.join(tmp_colmap, 'images.txt')
        with open(images_txt, 'w') as f:
            for i, p in enumerate(chunk_poses):
                x, y, z, yaw = p
                R_c2w, qvec_c2w = euler_to_quaternion([yaw, 0, 0])
                rx_loc = np.array([x, y, z])
                tvec_c2w = -R_c2w.apply(rx_loc)
                
                filename = f"image_{global_idx+i:06d}.png"
                
                qw, qx, qy, qz = qvec_c2w
                tx, ty, tz = tvec_c2w
                
                # colmap format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
                f.write(f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {filename}\n")
                
        # Call render script
        print(f"Rendering chunk {chunk_start//CHUNK_SIZE + 1} / {int(np.ceil(len(poses)/CHUNK_SIZE))}")
        
        # We need to render the poses using python render_custom_poses.py
        # You can adjust width, height, and fov to match the expected format
        cmd = [
            "python", "render_custom_poses.py", 
            "-m", model_path,
            "--colmap_dir", tmp_colmap,
            "--width", "600",
            "--height", "600",
            "--fov", "1.5708" # dummy fov if camera isn't used correctly
        ]
        
        subprocess.run(cmd, check=True)
        
        # Move rendered images and append labels
        with open(rf3dgs_labels_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for i, p in enumerate(chunk_poses):
                filename = f"image_{global_idx+i:06d}.png"
                src_path = os.path.join(out_dir, filename)
                dst_path = os.path.join(rf3dgs_img_dir, filename)
                
                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)
                    
                yaw_deg = np.degrees(p[3])
                writer.writerow([filename, p[0], p[1], p[2], yaw_deg])
                
        global_idx += len(chunk_poses)
        
    # Clean up temp files
    shutil.rmtree(tmp_colmap)
    print("Dataset generation completed.")

if __name__ == '__main__':
    main()
