#!/bin/bash

# Configuration
TARGET_DIR="/home/ved/Ved/Project_1/localization_frames"
MODEL_PATH="output/rf_model_delay"
ITERATION=40000
MAX_POSITIONS=50

# Activate conda environment securely inside script
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rf-3dgs

echo "=================================================================="
echo "Starting Grid Search Evaluation for up to $MAX_POSITIONS positions"
echo "=================================================================="

# Temporary file to store raw results
RESULTS_FILE=$(mktemp)

count=0
# Loop through the delay target images (only delay files to avoid duplicates with phase) 
for img in "$TARGET_DIR"/delay_*.png; do
    if [ "$count" -ge "$MAX_POSITIONS" ]; then
        break
    fi

    # Extract target GT from filename, e.g., delay_0.40_0.40_1.20_0.png -> x=0.40, y=0.40, z=1.20
    filename=$(basename "$img")
    
    # Simple python parser for the filename 
    gt_coords=$(python3 -c "
import re, sys
match = re.findall(r'-?\d+\.\d+', sys.argv[1])
if len(match) >= 3:
    print(f'{float(match[0])},{float(match[1])},{float(match[2])}')
else:
    print('')
" "$filename")

    if [ -z "$gt_coords" ]; then
        echo "Skipping $filename (could not parse GT coordinates)"
        continue
    fi

    gt_x=$(echo "$gt_coords" | cut -d',' -f1)
    gt_y=$(echo "$gt_coords" | cut -d',' -f2)
    gt_z=$(echo "$gt_coords" | cut -d',' -f3)

    echo -ne "Processing [$((count+1))/$MAX_POSITIONS] $filename ... "

    # Run the grid search script
    output=$(python grid_search_localization.py --target_image "$img" --model_path "$MODEL_PATH" --iteration "$ITERATION" 2>&1)

    # We assume 'Pose_0' contains the best estimation
    # Example parsing: "Pose_0 = (Position: (X: 3.50, Y: 2.50, Z: 1.50), Rotation/Yaw: 0°)"
    est_line=$(echo "$output" | grep "Pose_0" | grep "Position:" | head -n 1)

    if [ -z "$est_line" ]; then
        echo "FAILED (Could not parse output)"
        continue
    fi

    est_coords=$(python3 -c "
import re, sys
line = sys.argv[1]
pos_match = re.search(r'X:\s*([\d.-]+).*Y:\s*([\d.-]+).*Z:\s*([\d.-]+)', line)
if pos_match:
    print(f'{float(pos_match.group(1))},{float(pos_match.group(2))},{float(pos_match.group(3))}')
else:
    print('')
" "$est_line")

    if [ -z "$est_coords" ]; then
        echo "FAILED (Regex parsing failed)"
        continue
    fi

    est_x=$(echo "$est_coords" | cut -d',' -f1)
    est_y=$(echo "$est_coords" | cut -d',' -f2)
    est_z=$(echo "$est_coords" | cut -d',' -f3)

    # Compute Euclidean distance (error) and Squared Error in python
    errors=$(python3 -c "
import sys, math
gx, gy, gz = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
ex, ey, ez = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
dx = ex - gx
dy = ey - gy
dz = ez - gz
sq_err = dx*dx + dy*dy + dz*dz
dist = math.sqrt(sq_err)
print(f'{dist:.6f},{sq_err:.6f}')
" "$gt_x" "$gt_y" "$gt_z" "$est_x" "$est_y" "$est_z")

    dist=$(echo "$errors" | cut -d',' -f1)
    sq_err=$(echo "$errors" | cut -d',' -f2)

    echo "Distance Error = $dist m"

    # Store in results file: filename, dist, sq_err, gt_x, gt_y, gt_z, est_x, est_y, est_z
    echo "$filename|$dist|$sq_err|$gt_x|$gt_y|$gt_z|$est_x|$est_y|$est_z" >> "$RESULTS_FILE"

    count=$((count+1))
done

echo ""
echo "=================================================================="
echo "EVALUATION SUMMARY"
echo "=================================================================="

# Check if we processed anything
if [ ! -s "$RESULTS_FILE" ]; then
    echo "No successful evaluations were completed."
    rm -f "$RESULTS_FILE"
    exit 1
fi

# Calculate total MSE
MSE=$(python3 -c "
import sys
lines = sys.stdin.read().strip().split('\n')
if not lines or lines == ['']:
    print('0')
    sys.exit()
total_sq = 0.0
for line in lines:
    parts = line.split('|')
    if len(parts) >= 3:
        total_sq += float(parts[2])
print(f'{total_sq / len(lines):.6f}')
" < "$RESULTS_FILE")

echo "Total Evaluated Positions: $count"
echo "Mean Squared Error (MSE):  $MSE meters²"
echo "Root Mean Sq. Error (RMSE): $(python3 -c "import math; print(f'{math.sqrt($MSE):.6f}')") meters"
echo ""

echo "=================================================================="
echo "TOP 10 BEST RESULTS (Lowest Distance Error)"
echo "=================================================================="
# Sort by the 2nd column (Distance Error) numerically (-n), take top 10
sort -t'|' -k2 -n "$RESULTS_FILE" | head -n 10 | awk -F'|' '
BEGIN {
    printf "%-5s | %-12s | %-30s | %-20s | %-20s\n", "Rank", "Error (m)", "Image File", "Ground Truth (XYZ)", "Estimated (XYZ)"
    printf "------------------------------------------------------------------------------------------------------\n"
}
{
    printf "%-5d | %-12s | %-30s | %s,%s,%s | %s,%s,%s\n", NR, $2, $1, $4, $5, $6, $7, $8, $9
}'

# Clean up
rm -f "$RESULTS_FILE"
