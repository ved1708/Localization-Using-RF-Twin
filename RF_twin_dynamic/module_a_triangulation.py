import numpy as np
import cv2
import os
import sys

# ==========================================
# 1. USER CONFIGURATION
# ==========================================

PROJECT_ROOT = "/home/ved/Ved/Project_1"

# COLMAP Paths
COLMAP_PATH = os.path.join(PROJECT_ROOT, "dynamics_data/rf_data/sparse/0")
IMAGES_TXT = os.path.join(COLMAP_PATH, "images.txt")
CAMERAS_TXT = os.path.join(COLMAP_PATH, "cameras.txt")

# Image Paths
# Static: /home/ved/Ved/Project_1/dataset_custom_scene_ideal_mpc/spectrum/
STATIC_FOLDER = os.path.join(PROJECT_ROOT, "dataset_custom_scene_ideal_mpc", "spectrum")

# Dynamic: /home/ved/Ved/Project_1/dynamics_data/rf_data/spectrum/
DYNAMIC_FOLDER = os.path.join(PROJECT_ROOT, "dynamics_data/rf_data", "spectrum")

# Receivers to use
RX_FILENAME_1 = "00010.png"
RX_FILENAME_2 = "00172.png"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def qvec2rotmat(qvec):
    """Converts Quaternion to Rotation Matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def parse_cameras_txt(path):
    """Parses Camera Intrinsics."""
    cams = {}
    try:
        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split()
                if len(parts) < 5: continue
                cam_id = int(parts[0])
                # We need fx (Focal Length) and cx (Principal Point)
                fx = float(parts[4])
                cx = float(parts[6])
                cams[cam_id] = {'fx': fx, 'cx': cx}
    except FileNotFoundError:
        print(f"Error: {path} not found.")
        sys.exit(1)
    return cams

def parse_images_txt(path):
    """Parses Receiver Poses (Position and Rotation)."""
    poses = {}
    try:
        with open(path, "r") as f:
            lines = [l for l in f.readlines() if not l.startswith("#")]
    except FileNotFoundError:
        print(f"Error: {path} not found.")
        sys.exit(1)
        
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 9: continue
        qvec = np.array(list(map(float, parts[1:5])))
        tvec = np.array(list(map(float, parts[5:8])))
        cam_id = int(parts[8])
        filename = parts[9]
        
        # C = -R^T * t
        R = qvec2rotmat(qvec)
        C = -R.T @ tvec 
        poses[filename] = { 'pos': C, 'R': R, 'cam_id': cam_id }
    return poses

def intersect_lines_3d(p1, v1, p2, v2):
    """
    Mathematical Triangulation: Finds the closest point between two 3D Rays.
    p = origin, v = direction vector
    """
    w0 = p1 - p2
    a = np.dot(v1, v1); b = np.dot(v1, v2)
    c = np.dot(v2, v2); d = np.dot(v1, w0); e = np.dot(v2, w0)
    
    denom = a*c - b*b
    if denom < 1e-6: 
        return None # Lines are parallel

    s = (b*e - c*d) / denom
    t = (a*e - b*d) / denom

    point1 = p1 + s * v1
    point2 = p2 + t * v2
    
    # Return the midpoint between the two closest points on the rays
    return (point1 + point2) / 2

# ==========================================
# 3. PROCESSING LOGIC
# ==========================================

def get_ray_and_debug(filename, pose, cam, static_dir, dyn_dir):
    """
    1. Loads images
    2. Finds difference
    3. Calculates 3D Ray
    4. Returns Ray + Debug Images
    """
    p_stat = os.path.join(static_dir, filename)
    p_dyn = os.path.join(dyn_dir, filename)
    
    img_stat = cv2.imread(p_stat, 0)
    img_dyn = cv2.imread(p_dyn, 0)
    
    if img_stat is None:
        print(f"Error: Missing {p_stat}")
        return None, None, None
    if img_dyn is None:
        print(f"Error: Missing {p_dyn}")
        return None, None, None

    # 1. Difference
    diff = cv2.absdiff(img_dyn, img_stat)
    
    # 2. Threshold (Adjust 40 if your mask is too noisy or empty)
    _, mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    # 3. Find Blob Center
    M = cv2.moments(mask)
    if M["m00"] == 0:
        print(f"Warning: No signal detected in {filename}")
        # Return empty data but keep images for debug
        return None, None, (diff, mask, diff) 
        
    u = M["m10"] / M["m00"]
    v = M["m01"] / M["m00"]
    
    # 4. Create Debug Image (Draw Red Dot on Center)
    debug_img = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    cv2.circle(debug_img, (int(u), int(v)), 5, (0, 0, 255), -1)
    
    # 5. Calculate 3D Ray Direction
    # Pixel (u) -> Normalized Camera Coordinate (x)
    x_local = (u - cam['cx']) / cam['fx']
    
    # Local Vector (Forward is Z)
    local_dir = np.array([x_local, 0, 1.0])
    
    # Rotate to World Space
    world_dir = pose['R'].T @ local_dir
    world_dir = world_dir / np.linalg.norm(world_dir)
    
    return pose['pos'], world_dir, (diff, mask, debug_img)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("-" * 50)
    print(" MODULE A: TRIANGULATION + VISUAL DEBUG")
    print("-" * 50)

    # 1. Parse Metadata
    print("Reading COLMAP data...")
    poses = parse_images_txt(IMAGES_TXT)
    cameras = parse_cameras_txt(CAMERAS_TXT)
    
    # 2. Process Receiver 1
    print(f"Processing Rx 1: {RX_FILENAME_1}...")
    pose1 = poses[RX_FILENAME_1]
    cam1 = cameras[pose1['cam_id']]
    p1, d1, visuals1 = get_ray_and_debug(RX_FILENAME_1, pose1, cam1, STATIC_FOLDER, DYNAMIC_FOLDER)
    
    # 3. Process Receiver 2
    print(f"Processing Rx 2: {RX_FILENAME_2}...")
    pose2 = poses[RX_FILENAME_2]
    cam2 = cameras[pose2['cam_id']]
    p2, d2, visuals2 = get_ray_and_debug(RX_FILENAME_2, pose2, cam2, STATIC_FOLDER, DYNAMIC_FOLDER)

    # 4. Save Visual Debug Image
    if visuals1 and visuals2:
        # Check if we got valid images (visuals is a tuple of 3 images)
        # Create a row for Rx1: [Diff] [Mask] [Result]
        row1 = np.hstack([
            cv2.cvtColor(visuals1[0], cv2.COLOR_GRAY2BGR), # Diff (Gray to BGR)
            cv2.cvtColor(visuals1[1], cv2.COLOR_GRAY2BGR), # Mask (Gray to BGR)
            visuals1[2]                                    # Result (Already BGR)
        ])
        
        # Create a row for Rx2
        row2 = np.hstack([
            cv2.cvtColor(visuals2[0], cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(visuals2[1], cv2.COLOR_GRAY2BGR),
            visuals2[2]
        ])
        
        # Stack vertically
        final_grid = np.vstack([row1, row2])
        cv2.imwrite("debug_masks.png", final_grid)
        print("\n[VISUAL] Saved 'debug_masks.png'. Check this file!")

    # 5. Perform Triangulation
    if p1 is not None and p2 is not None:
        obj_pos = intersect_lines_3d(p1, d1, p2, d2)
        
        print("\n" + "="*40)
        print(" SUCCESS: OBJECT DETECTED")
        print("="*40)
        print(f" Receiver 1 Pos: [{p1[0]:.2f}, {p1[1]:.2f}, {p1[2]:.2f}]")
        print(f" Receiver 2 Pos: [{p2[0]:.2f}, {p2[1]:.2f}, {p2[2]:.2f}]")
        print("-" * 20)
        print(f" DYNAMIC OBJECT LOCATION (X, Y, Z):")
        print(f" [{obj_pos[0]:.4f}, {obj_pos[1]:.4f}, {obj_pos[2]:.4f}]")
        print("="*40)
        
        np.savetxt("dynamic_object_pos.txt", obj_pos)
    else:
        print("\nFAILED: One or both receivers did not detect the signal.")