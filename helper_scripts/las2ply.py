import laspy
import open3d as o3d
import numpy as np
import os

def optimize_and_convert(las_path, ply_path, voxel_size=0.05):
    # Check if input file exists
    if not os.path.exists(las_path):
        print(f"Error: The file '{las_path}' was not found.")
        return

    print(f"1. Loading LAS file: {las_path}...")
    try:
        las = laspy.read(las_path)
    except Exception as e:
        print(f"Failed to read LAS file: {e}")
        return
    
    # Extract XYZ coordinates
    # standard scaling is applied automatically by laspy (las.x, las.y, las.z)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Extract Colors (if available)
    if hasattr(las, 'red'):
        print("   Color data found. Processing...")
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
        # Auto-detect scale (8-bit vs 16-bit)
        if np.max(colors) > 255:
            colors = colors / 65535.0
        else:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        print("   No color data found. Result will be monochrome.")

    original_count = len(pcd.points)
    print(f"   Original Point Count: {original_count}")

    # --- OPTIMIZATION: DOWNSAMPLING ---
    # Reduces the number of points to make it run smoothly on WebGL
    print(f"2. Downsampling (Voxel Size: {voxel_size})...")
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    new_count = len(downpcd.points)
    print(f"   New Point Count: {new_count} (Reduced by {100 - (new_count/original_count)*100:.1f}%)")

    # --- OPTIMIZATION: NORMALS ---
    # Calculates lighting information so the model doesn't look flat
    print("3. Estimating Normals (for lighting/shading)...")
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.5, max_nn=30)
    )
    # Align normals to point outwards
    downpcd.orient_normals_consistent_tangent_plane(k=15)

    # --- SAVE ---
    print(f"4. Saving to {ply_path}...")
    o3d.io.write_point_cloud(ply_path, downpcd, write_ascii=False) # Binary save
    print("Done! You can now upload this PLY to your WebGL viewer.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Input: The raw scan
    input_file = "/home/ved/Downloads/XGrid_artGarage/art_garage.las"
    
    # Output: Will save in the folder where you run this script (unless you add a path)
    output_file = "artpark.ply" 
    
    # Controls detail level. 
    # 0.05 = 5cm spacing (Good balance for buildings/garages in Meters)
    # Increase to 0.1 or 0.2 if the file is still too slow in the browser.
    VOXEL_SIZE = 0.05 
    # ---------------------

    optimize_and_convert(input_file, output_file, voxel_size=VOXEL_SIZE)