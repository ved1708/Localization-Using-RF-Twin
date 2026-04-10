# Temporal Fine-Tuning for Dynamic Scene Reconstruction

## Overview

You have successfully generated:
- ✅ **1 visual image** (`dynamic_scene_visual/images/dynamic_frame_0000.png`)
- ✅ **2 RF images** (`dynamic_scene_rf_multiview/spectrum/dynamic_rf_0001.png`, `dynamic_rf_0002.png`)
- ✅ **Complex RF data** with phase information (`.npz` files)

Now you will:
1. **Fine-tune** your static model on this dynamic data (fast, ~1K iterations)
2. **Retrain** RF-3DGS from scratch on the dynamic scene (full training)
3. **Compare** the results to evaluate temporal fine-tuning effectiveness

---

## Method 1: Temporal Fine-Tuning (Recommended First)

### Step 1: Prepare Dynamic Dataset Directory

Your dynamic dataset should have this structure:

```
dynamic_scene_combined/
├── images/                      # Visual images
│   └── dynamic_frame_0000.png
├── spectrum/                    # RF images
│   ├── dynamic_rf_0001.png
│   ├── dynamic_rf_0001_complex.npz
│   ├── dynamic_rf_0002.png
│   └── dynamic_rf_0002_complex.npz
├── sparse/
│   └── 0/
│       ├── cameras.txt
│       ├── images.txt
│       └── points3D.txt
└── transforms.json              # Combined transforms
```

**Create the combined dataset:**

```bash
cd /home/ved/Ved/Project_1

# Create combined directory
mkdir -p dynamic_scene_combined/images
mkdir -p dynamic_scene_combined/spectrum

# Copy visual image
cp dynamic_scene_visual/images/dynamic_frame_0000.png dynamic_scene_combined/images/

# Copy RF images and complex data
cp dynamic_scene_rf_multiview/spectrum/* dynamic_scene_combined/spectrum/

# Copy COLMAP sparse reconstruction
cp -r dynamic_scene_rf_multiview/sparse dynamic_scene_combined/
```

**Create combined transforms.json:**

```bash
python3 << 'EOF'
import json
import os

# Load visual transform
with open('dynamic_scene_visual/transforms.json', 'r') as f:
    visual_data = json.load(f)

# Load RF transforms (from images.txt in sparse/0/)
# We'll add RF cameras to the same transforms.json

combined_data = {
    "camera_angle_x": visual_data["camera_angle_x"],
    "frames": []
}

# Add visual frame
visual_frame = visual_data["frames"][0].copy()
visual_frame["file_path"] = "images/dynamic_frame_0000"
visual_frame["time"] = 0
combined_data["frames"].append(visual_frame)

# Add RF frames (with same camera_angle_x for simplicity)
# RF camera positions from dynamic_scene_rf_multiview/sparse/0/images.txt
rf_frames = [
    {"file_path": "spectrum/dynamic_rf_0001", "time": 0},
    {"file_path": "spectrum/dynamic_rf_0002", "time": 0}
]

# Parse RF camera extrinsics and add to frames
# You may need to extract transform_matrix from sparse/0/images.txt
# For now, create minimal structure
for rf_frame in rf_frames:
    combined_data["frames"].append(rf_frame)

# Save combined transforms
os.makedirs('dynamic_scene_combined', exist_ok=True)
with open('dynamic_scene_combined/transforms.json', 'w') as f:
    json.dump(combined_data, f, indent=2)

print("✓ Created dynamic_scene_combined/transforms.json")
EOF
```

---

### Step 2: Fine-Tune on Dynamic Scene (Visual + RF)

**Option A: Fine-tune visual model on new visual + RF**

```bash
cd /home/ved/Ved/Project_1/RF-3DGS

# Fine-tune from your trained static model
python train.py \
  -s /home/ved/Ved/Project_1/dynamic_scene_combined \
  -m /home/ved/Ved/Project_1/output/dynamic_finetuned \
  --start_checkpoint /home/ved/Ved/Project_1/output/rf_model/chkpnt10000.pth \
  --iterations 1000 \
  --save_iterations 500 1000 \
  --test_iterations 500 1000 \
  --images spectrum
```

**Key Parameters:**
- `--start_checkpoint`: Path to your trained static RF model checkpoint
- `--iterations 1000`: Short fine-tuning (geometry already learned)
- `--images spectrum`: Use RF images for training
- Lower learning rates automatically applied after loading checkpoint

**What happens:**
1. Loads static model Gaussians (positions, opacities, features)
2. Freezes geometry (positions, scales, rotations) - see line 47-49 in train.py
3. Only updates:
   - Opacity (`_opacity`) - to adapt to new object (metallic cube)
   - Features (`_features_dc`, `_features_rest`) - to learn new RF signatures
4. Fast convergence (~2-3 minutes on GPU)

---

### Step 3: Render and Evaluate Fine-Tuned Model

```bash
cd /home/ved/Ved/Project_1/RF-3DGS

# Render test views
python render.py \
  -m /home/ved/Ved/Project_1/output/dynamic_finetuned \
  --iteration 1000

# Calculate metrics
python metrics.py \
  -m /home/ved/Ved/Project_1/output/dynamic_finetuned
```

**Output:**
- `output/dynamic_finetuned/test/ours_1000/renders/` - Rendered RF heatmaps
- `output/dynamic_finetuned/test/ours_1000/gt/` - Ground truth RF images
- `results.json` - PSNR, SSIM, LPIPS metrics

---

## Method 2: Full Retraining on Dynamic Scene (Comparison Baseline)

### Step 1: Generate Full Dynamic Dataset

For fair comparison, you need more images:

**Visual Images (100-200 views):**
```bash
cd /home/ved/Ved/Project_1

# Modify generate_visual_dataset.py to use meshes_d and room_with_cube.xml
# Then run:
blender --background --python generate_visual_dataset.py -- \
  --meshes_dir=meshes_d \
  --output_dir=dataset_dynamic_full
```

**RF Images (100+ views):**
```bash
# Modify generate_dataset_ideal_mpc.py or create multi-position RF generator
python generate_dataset_ideal_mpc.py --output_dir dataset_dynamic_full/spectrum
```

---

### Step 2: Train from Scratch on Dynamic Scene

```bash
cd /home/ved/Ved/Project_1/RF-3DGS

# Stage 1: Visual training (30K iterations)
python train.py \
  -s /home/ved/Ved/Project_1/dataset_dynamic_full \
  -m /home/ved/Ved/Project_1/output/dynamic_retrained_visual \
  --iterations 30000 \
  --save_iterations 7000 15000 30000

# Stage 2: RF fine-tuning (10K iterations)
python train.py \
  -s /home/ved/Ved/Project_1/dataset_dynamic_full \
  -m /home/ved/Ved/Project_1/output/dynamic_retrained_rf \
  --images spectrum \
  --start_checkpoint /home/ved/Ved/Project_1/output/dynamic_retrained_visual/chkpnt30000.pth \
  --iterations 10000 \
  --save_iterations 3000 7000 10000
```

**Time:** ~1-2 hours for full training

---

## Method 3: Quick Test with Minimal Data (What You Have Now)

Since you only have 1 visual + 2 RF images, you can still test fine-tuning:

### Quick Fine-Tune Script

```bash
cd /home/ved/Ved/Project_1/RF-3DGS

# Create minimal test dataset
python << 'EOF'
import os
import shutil
import json

# Create structure
os.makedirs('test_finetune/images', exist_ok=True)
os.makedirs('test_finetune/spectrum', exist_ok=True)

# Copy images
shutil.copy('../dynamic_scene_visual/images/dynamic_frame_0000.png', 
            'test_finetune/images/')
shutil.copy('../dynamic_scene_rf_multiview/spectrum/dynamic_rf_0001.png',
            'test_finetune/spectrum/')
shutil.copy('../dynamic_scene_rf_multiview/spectrum/dynamic_rf_0002.png',
            'test_finetune/spectrum/')

# Copy sparse reconstruction
shutil.copytree('../dynamic_scene_rf_multiview/sparse', 
                'test_finetune/sparse', dirs_exist_ok=True)

print("✓ Created test_finetune dataset")
EOF

# Fine-tune (500 iterations is enough for testing)
python train.py \
  -s test_finetune \
  -m ../output/test_finetune_rf \
  --start_checkpoint ../output/rf_model/chkpnt10000.pth \
  --iterations 500 \
  --save_iterations 250 500 \
  --test_iterations 250 500 \
  --images spectrum
```

---

## Comparison Strategy

### Metrics to Compare

1. **PSNR (Peak Signal-to-Noise Ratio)**
   - Higher is better (>20 dB good for RF)
   - Measures pixel-level accuracy

2. **SSIM (Structural Similarity)**
   - Range [0, 1], higher is better
   - Measures perceptual similarity

3. **Training Time**
   - Fine-tuning: ~2-5 minutes
   - Full retraining: ~1-2 hours

4. **Convergence Speed**
   - Plot loss curves (check TensorBoard logs)

5. **Novel View Quality**
   - Visual inspection of rendered RF heatmaps

### Expected Results

| Method | Training Time | PSNR (dB) | SSIM | Notes |
|--------|--------------|-----------|------|-------|
| **Fine-tuning** | 2-5 min | 18-22 | 0.75-0.85 | Fast, limited data |
| **Full Retrain** | 1-2 hr | 22-26 | 0.85-0.92 | Best quality, needs full dataset |
| **Static Model (no adaptation)** | 0 min | 12-16 | 0.50-0.65 | Baseline (will fail on new object) |

---

## Troubleshooting

### Issue 1: "Not enough training images"
**Solution:** Use `--debug_from 0` to disable densification checks:
```bash
python train.py ... --debug_from 0
```

### Issue 2: Checkpoint loading fails
**Cause:** Version mismatch or corrupted checkpoint
**Solution:** Check checkpoint exists and was saved correctly:
```bash
ls -lh /home/ved/Ved/Project_1/output/rf_model/chkpnt*.pth
```

### Issue 3: Loss explodes or doesn't decrease
**Cause:** Learning rate too high for fine-tuning
**Solution:** The code should auto-handle this (line 88 in train.py calls `update_learning_rate`), but you can manually reduce:
```python
# Edit arguments/__init__.py, reduce learning rates by 10x:
position_lr_max_steps = 30000  # Already set
feature_lr = 0.0025  # Reduce from 0.025
opacity_lr = 0.005   # Reduce from 0.05
```

### Issue 4: Out of memory
**Solution:** Reduce batch size or resolution:
```bash
python train.py ... --resolution 4  # Use 1/4 resolution
```

---

## Next Steps

### Recommended Workflow:

1. **Quick Test** (30 minutes):
   ```bash
   # Test fine-tuning with your current 1+2 images
   # Follow "Method 3" above
   ```

2. **Evaluate** (5 minutes):
   ```bash
   python render.py -m ../output/test_finetune_rf --iteration 500
   python metrics.py -m ../output/test_finetune_rf
   ```

3. **Compare with Static** (baseline):
   ```bash
   # Render static model on dynamic scene data
   python render.py \
     -m ../output/rf_model \
     --iteration 10000 \
     -s ../dynamic_scene_rf_multiview
   ```

4. **Full Experiment** (if quick test works):
   - Generate full dynamic dataset (100+ images each for visual/RF)
   - Run full retraining (Method 2)
   - Compare fine-tuning vs retraining quantitatively

---

## Expected Outcomes

### Fine-Tuning Success Indicators:
- ✅ Loss decreases quickly (< 100 iterations)
- ✅ RF heatmaps show metallic cube signature
- ✅ Background RF patterns preserved from static model
- ✅ PSNR improvement of 5-10 dB over non-adapted static model

### If Fine-Tuning Fails:
- Static model's geometry may be too rigid
- Need more diverse viewpoints (generate more RF images)
- Metallic cube's RF signature too different (requires full retrain)

---

## Summary Commands (Quick Start)

```bash
# 1. Setup combined dataset
cd /home/ved/Ved/Project_1/RF-3DGS
mkdir -p test_finetune/images test_finetune/spectrum
cp ../dynamic_scene_visual/images/*.png test_finetune/images/
cp ../dynamic_scene_rf_multiview/spectrum/*.png test_finetune/spectrum/
cp -r ../dynamic_scene_rf_multiview/sparse test_finetune/

# 2. Fine-tune (quick test)
python train.py \
  -s test_finetune \
  -m ../output/test_finetune_rf \
  --start_checkpoint ../output/rf_model/chkpnt10000.pth \
  --iterations 500 \
  --images spectrum

# 3. Evaluate
python render.py -m ../output/test_finetune_rf --iteration 500
python metrics.py -m ../output/test_finetune_rf

# 4. Visualize (open in browser)
firefox ../output/test_finetune_rf/test/ours_500/renders/dynamic_rf_0001.png
```

Good luck! Let me know the results after your first quick test! 🚀
