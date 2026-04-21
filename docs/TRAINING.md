# 3DGS Training and RRF Reconstruction

Complete guide for training 3D Gaussian Splatting models on visual and RF datasets to reconstruct Radio-Frequency Radiance Fields.

## Overview

The training process consists of two stages:

1.  **Stage 1: Visual Training** - Learn scene geometry from RGB images
2.  **Stage 2: RF Fine-tuning** - Learn RF propagation patterns from heatmaps

This two-stage approach leverages transfer learning: geometric knowledge from visual data helps the model learn RF patterns more effectively.

---

## Prerequisites

Ensure you have:

-   Visual dataset generated (`dataset_visual_v2/`)
-   RF dataset generated (`dataset_custom_scene_ideal_mpc/`)
-   RF-3DGS framework installed
-   CUDA 11.8+ and PyTorch 2.0+

---

## Stage 1: Visual Training

### Purpose

Train a 3DGS model to reconstruct RGB appearance and learn scene geometry (positions, shapes, normals) from photorealistic images.

### Command

```bash
cd RF-3DGS
conda activate rf-3dgs

python train.py \
  -s /home/ved/Ved/Project_1/dataset_visual \
  -m output/visual_model \
  --iterations 30000 \
  --save_iterations 15000 30000 \
  --test_iterations 15000 30000 \
  --eval
  ```

### Key Parameters

Parameter

Value

Description

`-s, --source_path`

`dataset_visual_v2/`

Path to visual dataset

`-m, --model_path`

`output/visual_model`

Output directory for checkpoints

`--iterations`

`30000`

Total training iterations

`--save_iterations`

`7000 15000 30000`

Save checkpoints at these iterations

`--test_iterations`

`7000 15000 30000`

Evaluate on test set

`--eval`

flag

Enable evaluation mode

**Additional Options**:

```bash
--resolution 1            # Downsample images (1=full, 2=half, 4=quarter)--white_background        # Use white background (instead of black)--sh_degree 3             # Spherical harmonics degree (0-3, higher=more view-dependent effects)--densify_grad_threshold 0.0002  # Gradient threshold for densification
```

### Training Process

**Initialization (Iteration 0)**:

-   Randomly initialize ~5,000 Gaussians in scene bounds
-   Or initialize from SfM point cloud (if available)

 **Opacity Reset** (every 3,000 iterations):
    
    -   Reset all opacities to prevent premature convergence

**Learning Rates** (exponential decay):

```python
# Position learning ratelr_pos = 0.00016 * (0.01 ** (iter / 30000))# Scaling learning rate  lr_scale = 0.005 * (0.01 ** (iter / 30000))# Rotation learning ratelr_rot = 0.001 * (0.01 ** (iter / 30000))# Opacity learning ratelr_opacity = 0.05 (constant)# SH coefficients learning ratelr_sh = 0.0025 * (0.01 ** (iter / 30000))
```

### Expected Output

```
output/visual_model/├── cameras.json                   # Camera metadata├── cfg_args                       # Training configuration├── input.ply                      # Initial point cloud├── point_cloud/│   ├── iteration_7000/│   │   └── point_cloud.ply       # 7K iteration Gaussians (~50K Gaussians)│   ├── iteration_15000/│   │   └── point_cloud.ply       # 15K iteration Gaussians (~80K Gaussians)│   └── iteration_30000/│       └── point_cloud.ply       # Final Gaussians (~100K-150K Gaussians)└── chkpnt30000.pth               # Full checkpoint (for fine-tuning) - see below
```

**Note**: The training by default only saves `.ply` files, not `.pth` checkpoints. **To generate the `.pth` checkpoint needed for RF fine-tuning, run**:

```bash
conda activate rf-3dgs

python ../helper_scripts/convert_ply_to_pth.py \
  --ply /home/ved/Ved/Project_1/RF-3DGS/output/visual_model/point_cloud/iteration_30000/point_cloud.ply \
  --out /home/ved/Ved/Project_1/RF-3DGS/output/visual_model/chkpnt30000.pth
```

This converts the PLY point cloud into a full checkpoint that includes optimizer state and all parameters required for the next stage.

### Monitoring Training

**Console Output**:

```
Iteration 0:    Loss: 0.5234  L1: 0.3421  SSIM: 0.6543  PSNR: 18.23 dB  Time: 0.12sIteration 100:  Loss: 0.3891  L1: 0.2156  SSIM: 0.7821  PSNR: 22.45 dB  Time: 0.09sIteration 1000: Loss: 0.2134  L1: 0.1234  SSIM: 0.8654  PSNR: 26.78 dB  Time: 0.08s...Iteration 30000: Loss: 0.0543  L1: 0.0321  SSIM: 0.9432  PSNR: 31.23 dB  Time: 0.07s[ITER 7000] Evaluating test: L1 0.0521 PSNR 28.45 SSIM 0.9123[ITER 15000] Evaluating test: L1 0.0412 PSNR 29.87 SSIM 0.9312[ITER 30000] Evaluating test: L1 0.0345 PSNR 31.23 SSIM 0.9456Training complete!
```

**Success Indicators**:

-   Loss decreases steadily (should reach <0.1 by iteration 20,000)
-   PSNR increases to >28 dB on test set
-   SSIM increases to >0.92 on test set
-   Training time: ~2-4 hours on RTX 3080

**Visualize Checkpoints**:

```bash
# Render test views at iteration 30,000python render.py -m output/visual_model --iteration 30000
```

---

## Stage 2: RF Fine-tuning

### Purpose

Fine-tune the visual model to learn RF propagation patterns from RF heatmaps while preserving geometric structure.

### Command

```bash
python train.py   -s /home/ved/Ved/Project_1/dataset_ideal_mpc   -m output/rf_model   --images spectrum   --start_checkpoint output/visual_model/chkpnt30000.pth   --iterations 40000  --save_iterations 40000 --test_iterations 40000 --eval
```

### Key Parameters

Parameter

Value

Description

`-s, --source_path`

`dataset_custom_scene_ideal_mpc/`

Path to RF dataset

`-m, --model_path`

`output/rf_model`

Output directory

`--images`

`spectrum`

Subfolder with RF heatmaps

`--start_checkpoint`

`output/visual_model/chkpnt30000.pth`

Visual checkpoint

`--iterations`

`10000`

Fewer iterations (geometry already learned)

### Fine-tuning Process

**Initialization**:

-   Load Gaussian positions, scales, rotations from visual checkpoint
-   Reset SH coefficients (color → RF power)
-   Keep geometric parameters mostly frozen (small learning rates)

**Key Differences from Stage 1**:

1.  **Lower learning rates**: 10x smaller for positions/scales
2.  **No densification**: Gaussian count stays fixed
3.  **Grayscale input**: RF heatmaps are single-channel
4.  **Different loss weights**: More emphasis on L1 (less on SSIM)

**Learning Rates**:

```python
# Position: 10x lower (geometry mostly fixed)lr_pos = 0.000016# Scaling: 5x lowerlr_scale = 0.001# Rotation: frozenlr_rot = 0.0# SH coefficients: learn RF patternslr_sh = 0.0025
```

### Expected Output

```
output/rf_model/├── cameras.json├── cfg_args├── point_cloud/│   ├── iteration_3000/│   │   └── point_cloud.ply       # 3K iteration RF Gaussians│   ├── iteration_7000/│   │   └── point_cloud.ply       # 7K iteration RF Gaussians│   └── iteration_10000/│       └── point_cloud.ply       # Final RRF model└── chkpnt10000.pth               # Final checkpoint
```

### Monitoring Fine-tuning

**Console Output**:

```
[RF Fine-tuning] Starting from checkpoint: output/visual_model/chkpnt30000.pthLoaded 123,456 GaussiansIteration 0:    Loss: 0.3421  L1: 0.2456  PSNR: 20.12 dB  Time: 0.08sIteration 1000: Loss: 0.1234  L1: 0.0987  PSNR: 25.67 dB  Time: 0.07sIteration 5000: Loss: 0.0621  L1: 0.0512  PSNR: 28.34 dB  Time: 0.07sIteration 10000: Loss: 0.0432  L1: 0.0378  PSNR: 29.87 dB  Time: 0.07s[ITER 3000] Evaluating test: L1 0.0632 PSNR 26.12 SSIM 0.8923[ITER 7000] Evaluating test: L1 0.0521 PSNR 28.45 SSIM 0.9145[ITER 10000] Evaluating test: L1 0.0456 PSNR 29.67 SSIM 0.9234RF Fine-tuning complete!
```

**Success Indicators**:

-   Loss decreases to <0.06 by iteration 10,000
-   PSNR >26 dB on RF test set (lower than visual, expected)
-   RF heatmaps show realistic propagation patterns

**Training time**: ~1-2 hours on RTX 3080

---

## One-Step Training Script

For convenience, use `run_rf_reconstruction.sh` to run both stages:

```bash
cd RF-3DGSbash run_rf_reconstruction.sh
```

**Script Contents**:

```bash
#!/bin/bashcd /home/ved/Ved/Project_1/RF-3DGSVISUAL_DATA="/home/ved/Ved/Project_1/dataset_visual_v2"RF_DATA="/home/ved/Ved/Project_1/dataset_custom_scene_ideal_mpc"VISUAL_MODEL_DIR="output/visual_model"RF_MODEL_DIR="output/rf_model"# Stage 1: Visual Trainingecho "===== Stage 1: Visual Training ====="conda run -n rf-3dgs --no-capture-output   python train.py -s "$VISUAL_DATA" -m "$VISUAL_MODEL_DIR"   --iterations 30000 --eval# Stage 2: RF Fine-tuningecho "===== Stage 2: RF Fine-tuning ====="conda run -n rf-3dgs --no-capture-output   python train.py -s "$RF_DATA" -m "$RF_MODEL_DIR"   --images spectrum   --start_checkpoint "$VISUAL_MODEL_DIR/chkpnt30000.pth"   --iterations 40000 --evalecho "Reconstruction Complete!"
```

---

## Advanced Configuration

### Memory Optimization

For GPUs with <8GB VRAM:

```bash
python train.py   --resolution 2                # Half resolution (400×400)  --densify_grad_threshold 0.0003   # More aggressive pruning  ...
```

### Higher Quality

For better reconstruction quality:

```bash
python train.py   --iterations 50000            # More iterations  --densification_interval 50   # More frequent densification  --sh_degree 3                 # Max SH degree (view-dependent effects)  ...
```

### Debug Mode

Enable debug logging:

```bash
python train.py   --debug   --test_iterations 500 1000 2000 5000 10000   ...
```

### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py   --data_device cuda:0   ...
```

---

## Understanding 3D Gaussians

Each Gaussian is parameterized by:

```python
class Gaussian:    xyz: float[3]           # Center position (x, y, z)    rotation: float[4]      # Quaternion (qw, qx, qy, qz)    scaling: float[3]       # Scale (sx, sy, sz) in log space    opacity: float[1]       # Alpha value in range [0, 1]    features_dc: float[3]   # SH coefficients (degree 0) = base color    features_rest: float[45] # SH coefficients (degrees 1-3) = view-dependent color
```

**For RF Gaussians**:

-   `features_dc`: Encodes RF power at Gaussian center
-   `features_rest`: Encodes angular dependence (directional propagation)
-   `opacity`: Represents RF absorption/transmission

### PLY File Format

Saved Gaussians in PLY format:

```
plyformat binary_little_endian 1.0element vertex 123456property float xproperty float yproperty float zproperty float nxproperty float nyproperty float nzproperty float f_dc_0property float f_dc_1property float f_dc_2property float f_rest_0...property float f_rest_44property float opacityproperty float scale_0property float scale_1property float scale_2property float rot_0property float rot_1property float rot_2property float rot_3end_header<binary data>
```

Load in Python:

```python
from plyfile import PlyDataply = PlyData.read("point_cloud.ply")xyz = np.stack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']], axis=1)print(f"Number of Gaussians: {len(xyz)}")
```

---

## Troubleshooting

### Issue 1: Training Diverges (NaN Loss)

**Symptom**: Loss becomes NaN after a few iterations

**Causes**:

1.  Learning rate too high
2.  Invalid camera poses
3.  Corrupted dataset

**Solutions**:

```bash
# Lower learning ratespython train.py --position_lr_init 0.00008 --scaling_lr 0.0025# Check camera posespython debug_scene.py# Verify dataset integritypython -c "from scene import Scene; Scene(args)"
```

### Issue 2: Poor Test PSNR (<25 dB)

**Symptom**: Visual model achieves low PSNR on test set

**Causes**:

1.  Insufficient training (need more iterations)
2.  Too few images (need more views)
3.  Low image overlap

**Solutions**:

```bash
# Train longerpython train.py --iterations 50000# Check dataset qualitypython render.py -m output/visual_model --iteration 30000# Inspect test/ours_30000/renders/ for artifacts
```

### Issue 3: RF Model Loses Geometry

**Symptom**: RF model produces blurry/distorted heatmaps

**Cause**: Learning rates too high for fine-tuning

**Solution**:

```bash
# Freeze geometry completelypython train.py   --position_lr_init 0.0   --scaling_lr 0.0   --rotation_lr 0.0
```

### Issue 4: Out of Memory

**Symptom**: `CUDA out of memory` error

**Solutions**:

```bash
# Reduce resolutionpython train.py --resolution 2  # or 4# Reduce batch size (if implemented)# Prune more aggressivelypython train.py --densify_grad_threshold 0.0005# Use smaller SH degreepython train.py --sh_degree 1
```

### Issue 5: Slow Training

**Symptom**: >10 seconds per iteration

**Solutions**:

-   **Enable GPU**: Check `nvidia-smi` and PyTorch CUDA
-   **Reduce resolution**: `--resolution 2`
-   **Disable eval**: Remove `--eval` flag during training
-   **Check I/O**: Use SSD for dataset storage

---

## Tips for Best Results

### Dataset Quality

1.  **High overlap**: Neighboring views should share >60% of scene
2.  **Diverse viewpoints**: Cover all angles, heights
3.  **Consistent lighting**: Use HDRI or uniform lighting
4.  **Sharp images**: No motion blur or defocus

---

## Next Steps

After training:

1.  **Render test views**: `python render.py -m output/rf_model`
2.  **Compute metrics**: `python metrics.py -m output/rf_model`
3.  **Visualize in viewer**: Launch SIBR viewer
4.  **Export results**: Generate videos, figures

---

**See also**:

-   [Main README](../README.md) - Complete pipeline
-   [Evaluation Guide](EVALUATION.md) - Quality assessment
-   [Visualization Guide](VISUALIZATION.md) - Interactive viewer

---

**Training Time Estimates** (RTX 3080):

-   Visual (30K iters): 2-4 hours
-   RF Fine-tuning (10K iters): 1-2 hours
-   **Total**: 3-6 hours

**GPU Memory Requirements**:

-   Visual training: ~6-8 GB
-   RF fine-tuning: ~4-6 GB
-   Rendering: ~2-4 GB