import os
import shutil
import glob
import numpy as np

rf_root = "/home/ved/Ved/Project_1/dataset_ideal_delay_3.5ghz"
sparse_dir = os.path.join(rf_root, "sparse", "0")

# 1. Create sparse/0 directory
os.makedirs(sparse_dir, exist_ok=True)
print(f"Created {sparse_dir}")

# 2. Copy cameras.txt and images.txt
for fname in ["cameras.txt", "images.txt"]:
    src = os.path.join(rf_root, fname)
    dst = os.path.join(sparse_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {fname} to {sparse_dir}")
    else:
        print(f"Warning: {fname} not found in {rf_root}")

# 3. Create dummy points3D.txt
points3d_path = os.path.join(sparse_dir, "points3D.txt")
if not os.path.exists(points3d_path):
    with open(points3d_path, "w") as f:
        # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
        # Create one dummy point at origin
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 1\n")
        f.write("1 0.0 0.0 0.0 128 128 128 0.0\n")
    print(f"Created dummy {points3d_path}")

# 4. Generate train_index.txt and test_index.txt
spectrum_dir = os.path.join(rf_root, "spectrum")
images = sorted(glob.glob(os.path.join(spectrum_dir, "*.png")))
image_names = [os.path.basename(img) for img in images]

# Select 20 evenly distributed images for testing
num_test = 20
num_images = len(image_names)
test_step = num_images // num_test if num_test > 0 else num_images

# Select test indices (evenly spaced throughout dataset)
test_indices = [image_names[i * test_step - 1] for i in range(1, num_test + 1) if i * test_step - 1 < num_images]

# Remaining images for training
train_indices = [img for img in image_names if img not in test_indices]

# The dataset_readers.py seems to expect filenames in train_index.txt
# It loads them and strips extension if present in the file content, but here we can just write filenames.
# "base_name = os.path.basename(filename).split(".")[0]"

with open(os.path.join(rf_root, "train_index.txt"), "w") as f:
    for name in train_indices:
        f.write(f"{name}\n")
print(f"Created train_index.txt with {len(train_indices)} images")

with open(os.path.join(rf_root, "test_index.txt"), "w") as f:
    for name in test_indices:
        f.write(f"{name}\n")
print(f"Created test_index.txt with {len(test_indices)} images")
