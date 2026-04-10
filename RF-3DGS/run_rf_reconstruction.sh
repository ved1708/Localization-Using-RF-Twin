#!/bin/bash
cd /home/ved/Ved/Project_1/RF-3DGS

# Set paths
VISUAL_DATA="/home/ved/Ved/Project_1/dataset_visual_v2"
RF_DATA="/home/ved/Ved/Project_1/dataset_custom_scene_ideal_mpc"
VISUAL_MODEL_DIR="output/visual_model"
RF_MODEL_DIR="output/rf_model"

# 1. Train Visual Model
echo "Starting Visual Training..."
conda run -n rf-3dgs --no-capture-output python train.py -s "$VISUAL_DATA" -m "$VISUAL_MODEL_DIR"

# 2. Train RF Model (fine-tuning from visual)
echo "Starting RF Training..."
conda run -n rf-3dgs --no-capture-output python train.py -s "$RF_DATA" --images spectrum --start_checkpoint "$VISUAL_MODEL_DIR/chkpnt30000.pth" -m "$RF_MODEL_DIR"

echo "Reconstruction Complete!"
