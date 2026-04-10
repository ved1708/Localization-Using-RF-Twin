TARGET_DIR="/home/ved/Ved/Project_1/localisastion_eval/spectrum"
IMAGES_TXT="/home/ved/Ved/Project_1/localisastion_eval/images.txt"
MODEL_PATH="output/rf_model_delay"
ITERATION=40000

img="$TARGET_DIR/00003.png"
filename=$(basename "$img")

# Extract ground truth coordinates from images.txt using scipy rotation
gt_coords=$(python3 -c "
import sys
from scipy.spatial.transform import Rotation as R
import numpy as np

images_txt = sys.argv[1]
target_name = sys.argv[2]

with open(images_txt, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 10 and parts[9] == target_name:
            # QW QX QY QZ
            qvec = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[1])] # qx qy qz qw for scipy
            tvec_c2w = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
            r_c2w = R.from_quat(qvec)
            tvec_w2c = -r_c2w.inv().apply(tvec_c2w)
            print(f'{tvec_w2c[0]:.6f},{tvec_w2c[1]:.6f},{tvec_w2c[2]:.6f}')
            sys.exit(0)
print('')
" "$IMAGES_TXT" "$filename")

echo "GT coords: $gt_coords"

output=$(python grid_search_localization.py --target_image "$img" --model_path "$MODEL_PATH" --iteration "$ITERATION" 2>&1)
est_line=$(echo "$output" | grep "Pose_0" | grep "Position:" | head -n 1)
echo "EST line: $est_line"
