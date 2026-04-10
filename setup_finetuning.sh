#!/bin/bash
# Quick Setup for Temporal Fine-Tuning Test
# Run this from /home/ved/Ved/Project_1

set -e  # Exit on error

echo "=========================================="
echo "Temporal Fine-Tuning Quick Setup"
echo "=========================================="

# Change to RF-3DGS directory
cd /home/ved/Ved/Project_1/RF-3DGS

echo ""
echo "Step 1: Creating test dataset structure..."
mkdir -p test_finetune/images
mkdir -p test_finetune/spectrum

echo "Step 2: Copying visual images..."
cp ../dynamic_scene_visual/images/dynamic_frame_0000.png test_finetune/images/ || {
    echo "Error: Visual image not found. Did you run generate_single_visual_dynamic.py?"
    exit 1
}

echo "Step 3: Copying RF images..."
cp ../dynamic_scene_rf_multiview/spectrum/dynamic_rf_0001.png test_finetune/spectrum/ || {
    echo "Error: RF images not found. Did you run generate_single_rf_dynamic.py?"
    exit 1
}
cp ../dynamic_scene_rf_multiview/spectrum/dynamic_rf_0002.png test_finetune/spectrum/

# Optional: Copy complex RF data
if [ -f ../dynamic_scene_rf_multiview/spectrum/dynamic_rf_0001_complex.npz ]; then
    echo "Step 4: Copying complex RF data..."
    cp ../dynamic_scene_rf_multiview/spectrum/*.npz test_finetune/spectrum/
fi

echo "Step 5: Copying COLMAP sparse reconstruction..."
cp -r ../dynamic_scene_rf_multiview/sparse test_finetune/ || {
    echo "Error: COLMAP data not found."
    exit 1
}

echo ""
echo "✓ Dataset structure created:"
tree test_finetune -L 2 2>/dev/null || find test_finetune -type f | head -20

echo ""
echo "Step 6: Checking for static model checkpoint..."
CHECKPOINT="../output/rf_model/chkpnt10000.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "⚠ Warning: Static model checkpoint not found at $CHECKPOINT"
    echo "Available checkpoints:"
    find ../output -name "*.pth" -type f 2>/dev/null || echo "None found"
    echo ""
    echo "Please specify the correct checkpoint path when running fine-tuning."
    CHECKPOINT="<YOUR_CHECKPOINT_PATH>"
fi

echo ""
echo "=========================================="
echo "Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Quick fine-tuning test (500 iterations, ~2 minutes):"
echo "   cd /home/ved/Ved/Project_1/RF-3DGS"
echo "   python train.py \\"
echo "     -s test_finetune \\"
echo "     -m ../output/test_finetune_rf \\"
echo "     --start_checkpoint $CHECKPOINT \\"
echo "     --iterations 500 \\"
echo "     --save_iterations 250 500 \\"
echo "     --test_iterations 250 500 \\"
echo "     --images spectrum"
echo ""
echo "2. Render results:"
echo "   python render.py -m ../output/test_finetune_rf --iteration 500"
echo ""
echo "3. Evaluate metrics:"
echo "   python metrics.py -m ../output/test_finetune_rf"
echo ""
echo "4. Visualize:"
echo "   eog ../output/test_finetune_rf/test/ours_500/renders/*.png"
echo ""
echo "=========================================="
