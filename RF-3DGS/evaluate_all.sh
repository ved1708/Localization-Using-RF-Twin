#!/bin/bash

# Directory containing the target images
TARGET_DIR="/home/ved/Ved/Project_1/localisation_frames_3.5ghz_2"

# Model path and iteration arguments
MODEL_PATH="output/rf_model_delay_3.5ghz"
ITERATION=40000

# Create a temporary file for outputs
TMP_OUTPUT="eval_tmp_output.log"
RESULTS_CSV="evaluation_results.csv"

echo "Running evaluations..."
echo ""

# Print Table Header to stdout
printf "%-25s | %-25s | %-10s | %-10s | %-10s\n" "GT Pos (X,Y,Z)" "Est Pos (X,Y,Z)" "Yaw (°)" "Error (m)" "Time (s)"
printf "%s\n" "----------------------------------------------------------------------------------------------------------"

# Write header to CSV
echo "GT_X,GT_Y,GT_Z,Est_X,Est_Y,Est_Z,Yaw,Error,Time_s,Filename" > "$RESULTS_CSV"

total_sq_error=0
total_time=0
count=0
min_error=99
max_error=0

# Loop through all png files in the target directory
for img in "$TARGET_DIR"/*.png; do
    if [ -f "$img" ]; then
        filename=$(basename "$img")
        
        # Print progress to stderr so it doesn't mess up the table if redirected
        echo "Evaluating $filename..." >&2
        
        # Run the python script and capture its output
        python gradient_descent_localization.py \
            --target_image "$img" \
            --model_path "$MODEL_PATH" \
            --iteration "$ITERATION" > "$TMP_OUTPUT" 2>&1
            
        # Extract values using grep and awk
        # We use tail to get the final optimization block output
        est_x=$(grep "  X: " "$TMP_OUTPUT" | tail -n 1 | awk '{print $2}')
        est_y=$(grep "  Y: " "$TMP_OUTPUT" | tail -n 1 | awk '{print $2}')
        est_z=$(grep "  Z: " "$TMP_OUTPUT" | tail -n 1 | awk '{print $2}')
        yaw=$(grep "Optimized Yaw:" "$TMP_OUTPUT" | tail -n 1 | awk '{print $3}' | tr -d '°')
        
        # Ground Truth Values (stripping trailing characters like 'm' or ',')
        gt_x=$(grep "  Real Position" "$TMP_OUTPUT" | tail -n 1 | awk -F'X=' '{print $2}' | awk -F'm' '{print $1}')
        gt_y=$(grep "  Real Position" "$TMP_OUTPUT" | tail -n 1 | awk -F'Y=' '{print $2}' | awk -F'm' '{print $1}')
        gt_z=$(grep "  Real Position" "$TMP_OUTPUT" | tail -n 1 | awk -F'Z=' '{print $2}' | awk -F'm' '{print $1}')
        
        error=$(grep "Distance from Real Position (Error):" "$TMP_OUTPUT" | tail -n 1 | awk '{print $6}')
        time_s=$(grep "Refinement Time:" "$TMP_OUTPUT" | tail -n 1 | awk '{print $3}')

        # Format missing or empty values as N/A in terminal display
        if [[ -z "$gt_x" ]]; then
           gt_str="N/A"
        else
           gt_str="${gt_x}, ${gt_y}, ${gt_z}"
        fi
        
        if [[ -z "$est_x" ]]; then
           est_str="N/A"
        else
           est_str="${est_x}, ${est_y}, ${est_z}"
        fi
        err_str="${error:-N/A}"
        yaw_str="${yaw:-N/A}"
        time_str="${time_s:-N/A}"

        # Print the row in the terminal table
        printf "%-25s | %-25s | %-10s | %-10s | %-10s\n" "$gt_str" "$est_str" "$yaw_str" "$err_str" "$time_str"
        
        # Accumulate metrics
        if [[ -n "$error" ]] && [[ -n "$time_s" ]]; then
            total_sq_error=$(echo "$total_sq_error + ($error * $error)" | bc -l)
            total_time=$(echo "$total_time + $time_s" | bc -l)
            if [ "$(echo "$error < $min_error" | bc -l)" -eq 1 ]; then
                min_error=$error
            fi
            if [ "$(echo "$error > $max_error" | bc -l)" -eq 1 ]; then
                max_error=$error
            fi
            count=$((count + 1))
        fi

        # Append to CSV
        echo "$gt_x,$gt_y,$gt_z,$est_x,$est_y,$est_z,$yaw,$error,$time_s,$filename" >> "$RESULTS_CSV"
    fi
done

# Clean up temp file
if [ -f "$TMP_OUTPUT" ]; then
    rm "$TMP_OUTPUT"
fi

echo ""
if [ "$count" -gt 0 ]; then
    mse=$(echo "scale=6; $total_sq_error / $count" | bc -l)
    avg_time=$(echo "scale=6; $total_time / $count" | bc -l)
    echo "Summary($count evaluations):"
    echo "  Mean Squared Error (MSE): $mse m^2"
    echo "  Average Time: $avg_time s"
    echo "  Min Error: $min_error m"
    echo "  Max Error: $max_error m"
    echo ""
    
    # Append summary to CSV for reference
    echo "MSE,$mse,Avg_Time,$avg_time,Min_Error,$min_error,Max_Error,$max_error" >> "$RESULTS_CSV"
fi

echo "Done! The results have also been saved to $RESULTS_CSV"
