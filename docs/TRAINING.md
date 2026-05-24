# 3DGS Training and RRF Reconstruction

Complete guide for training 3D Gaussian Splatting models on visual and RF datasets to reconstruct Radio-Frequency Radiance Fields using the [RF-3DGS](https://github.com/SunLab-UGA/RF-3DGS) framework.

## Overview

The training process consists of two stages:

1.  **Stage 1: Visual Training** - Learn scene geometry from RGB images
2.  **Stage 2: RF Fine-tuning** - Learn RF propagation patterns from heatmaps

---

## Prerequisites

Ensure you have:

-   Visual dataset generated ([dataset_visual](../dataset_visual/))
-   RF dataset generated ([datset_ideal_delay_3.5GHz](../dataset_ideal_delay_3.5ghz/)(used for Localisation) / [dataset_ideal_mpc](../dataset_ideal_mpc/))
-   [RF-3DGS framework](https://github.com/SunLab-UGA/RF-3DGS) installed
-   CUDA 11.8+ and PyTorch 2.0+

---

## Stage 1: Visual Training
Train a 3DGS model to reconstruct RGB appearance and learn scene geometry (positions, shapes, normals) from photorealistic images.

### Command

```bash
cd RF-3DGS
conda activate rf-3dgs

python train.py \
  -s ../dataset_visual \
  -m output/visual_model \
  --iterations 30000 \
  --save_iterations 15000 30000 \
  --test_iterations 15000 30000 \
  --eval
  ```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `-s, --source_path` | `dataset_visual` | Path to visual dataset |
| `-m, --model_path` | `output/visual_model` | Output directory for checkpoints |
| `--iterations` | `30000` | Total training iterations |
| `--save_iterations` | `7000 15000 30000` | Save checkpoints at these iterations |
| `--test_iterations` | `7000 15000 30000` | Evaluate on test set |
| `--eval` | flag | Enable evaluation mode |

**Note**: The training by default only saves `.ply` files, not `.pth` checkpoints. **To generate the `.pth` checkpoint needed for RF fine-tuning, run**:

```bash
conda activate rf-3dgs

python ../helper_scripts/convert_ply_to_pth.py \
  --ply output/visual_model/point_cloud/iteration_30000/point_cloud.ply \
  --out output/visual_model/chkpnt30000.pth
```

This converts the PLY point cloud into a full checkpoint that includes optimizer state and all parameters required for the next stage.

### Monitoring

**Visualize Checkpoints**:

```bash
# Render test views at iteration 30,000
python render.py -m output/visual_model --iteration 30000
```

---

## Stage 2: RF Fine-tuning

Fine-tune the visual model to learn RF propagation patterns from RF heatmaps while preserving geometric structure.

### Command

```bash
python train.py \
  -s ../dataset_ideal_delay_3.5ghz \
  -m output/rf_model_delay_3.5ghz \
  --images spectrum \
  --start_checkpoint output/visual_model/chkpnt30000.pth \
  --iterations 45000 \
  --save_iterations 40000 45000 \
  --test_iterations 40000 45000 \
  --eval
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `-s, --source_path` | `dataset_ideal_mpc` | Path to RF dataset |
| `-m, --model_path` | `output/rf_model` | Output directory |
| `--images` | `spectrum` | Subfolder with RF heatmaps |
| `--start_checkpoint` | `output/visual_model/chkpnt30000.pth` | Visual checkpoint |
| `--iterations` | `10000` | Fewer iterations (geometry already learned) |

### Fine-tuning Process

**Initialization**:

-   Load Gaussian positions, scales, rotations from visual checkpoint and runs after 30000 iterations.
-   Reset SH coefficients (color → RF power)
-   Keep geometric parameters frozen

### Expected Output

```
output/rf_model/├── cameras.json├── cfg_args├── point_cloud/│   ├── iteration_3000/│   │   └── point_cloud.ply       # 3K iteration RF Gaussians│   ├── iteration_7000/│   │   └── point_cloud.ply       # 7K iteration RF Gaussians│   └── iteration_10000/│       └── point_cloud.ply       # Final RRF model└── chkpnt10000.pth               # Final checkpoint
```

---
### Training Process

**Initialization (Iteration 0)**:

-   Randomly initialize ~5,000 Gaussians in scene bounds
-   Or initialize from SfM point cloud (if available)

**Opacity Reset** (every 3,000 iterations):
    
-   Reset all opacities to prevent premature convergence

---

## Troubleshooting

### Issue 1: Poor Test PSNR (<25 dB)

**Symptom**: Visual model achieves low PSNR on test set

**Causes**:

1.  Insufficient training (need more iterations)
2.  Too few images (need more views)
3.  Low image overlap
---

## Next Steps

After training:

1.  **Render test views**: `python render.py -m output/rf_model_delay_3.5ghz`
2.  **Compute metrics**: `python metrics.py -m output/rf_model_delay_3.5ghz`
3.  **Visualize in viewer**: Antimatter 3DGS viewer
---

**See also**:

-   [Main README](../README.md) - Complete pipeline
-   [Evaluation Guide](EVALUATION.md) - Quality assessment
-   [Visualization Guide](VISUALIZATION.md) - Interactive viewer