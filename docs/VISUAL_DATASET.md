# Visual Dataset Generation with Blender

Complete guide for generating photorealistic RGB image datasets for 3D Gaussian Splatting training.

This process creates a synthetic dataset of RGB images with accurate camera poses using Blender's Cycles renderer. The dataset serves as the foundation for training the visual geometry component of the RRF model.

---

## Script: generate_visual_dataset.py

### Purpose

Generate a high-quality RGB dataset (400+ images) from diverse camera viewpoints inside the 3D room scene, specifically tailored for 3DGS training with optimal coverage and feature capture.

### Requirements

-   **Blender 3.6+** (supports up to Blender 5.0+)
-   **Python modules**: `bpy`, `mathutils`, `argparse`, `random` (built into Blender)
-   **Input**: PLY meshes from `Project_1/meshes_d/` directory (created via [SCENE_CREATION.md](SCENE_CREATION.md))
-   **Hardware**: GPU with CUDA/OptiX support

---
## Usage

### Generate Training Dataset
```bash
blender --background --python /home/ved/Ved/Project_1/generate_visual_dataset.py -- \
  --meshes_dir=meshes_d \
  --output_dir=dataset_visual \
  --test_ratio=0.10
```

**Output**:
-   `dataset_visual/images/` (PNG renders)
-   `dataset_visual/transforms_train.json` (Optimized for 3DGS)
-   `dataset_visual/transforms_test.json` (Used for PSNR/SSIM metrics)

Time: ~15 minutes (GPU)

## Command-Line Arguments

### generate_visual_dataset.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--meshes_dir` | `meshes` | Directory containing PLY mesh files |
| `--output_dir` | `dataset_visual_1` | Output directory name |
| `--test_ratio` | `0.10` | Percentage of data for validation |
| `--split_mode` | `spatial` | Split strategy: `spatial`, `block`, or `periodic` |
| `--test_block_size` | `8` | Chunk size for `block` split mode |
| `--spatial_min_dist`| `0.35` | Min meters between test frames in `spatial` mode |
---

## Configuration

Key parameters in `generate_visual_dataset.py`:

```python
# === PATHS ===
BASE_DIR = "/home/ved/Ved/Project_1"
INPUT_MODELS_DIR = os.path.join(BASE_DIR, "meshes_d")
OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "dataset_visual")
```
---

## Workflow Setup

### 1. Scene Setup

Import Meshes:
The script automatically searches for material keywords in filenames:
-   `floor`, `walls` → Concrete (Mat_Grey/White)
-   `ceiling` → Custom bluish-grey 
-   `window` → Semi-transparent glass (Transmission 0.7)
-   `door`, `furniture` → Wood brown
-   `led_tv`, `metallic_cube` → Metallic (Metallic 0.8, Roughness 0.5)

Uses **Textures** Noise to ensure high-feature density for better 3DGS convergence.

### 2. Lighting Setup

Uses a hybrid lighting model for realism, combining ambient fill, a directional sun light, and a large ceiling softbox to ensure uniform, soft illumination across the entire room.

---

## Train/Test Splitting

The script includes sophisticated splitting modes (defaulting to **Spatial**):
-   **Spatial**: Selects test frames based on physical distance to prevent data leakage.
-   **Periodic**: Every Nth frame is selected for test.

**Note**: Frames identified as "too black" are automatically discarded.

---
## Material Properties
-   **Metallic objects** (`metallic_cube`, `led_tv`): Metallic=0.8, Roughness=0.5
-   **Glass objects** (`window`): Transmission=0.7, IOR=1.52, Alpha noise masking
-   **Standard objects**: Dual-scale procedural noise (Scale 15 & 100) + Bump mapping (Strength 0.2)
-   **Ceiling**: Distinctive bluish-grey.

---
## Resume Capability
If the script is interrupted, simply re-run it. The script will:
-   ✓ Skip rendering frames that already exist on disk.
-   ✓ Accurately collect camera transforms for the JSON manifest.
-   ✓ Re-calculate train/test splits based on the requested strategy.

## Troubleshooting

### GPU Not Detected
If rendering is slow, verify the output terminal:
```
✓ Enabled GPU: NVIDIA RTX [...]
```
If you see "Falling back to CPU rendering", check if Blender can access your GPU (Edit → Preferences → System).

### Black Images
The script automatically discards images that are >3% black pixels (usually means the camera spawned inside a wall). These frames will be missing from the final JSON files.

---

**See also**:
-   [Main README](../README.md) - Project overview
-   [RF Dataset Generation](RF_DATASET.md) - Next step
-   [3DGS Training](TRAINING.md) - Model training
-   [Scene Creation](SCENE_CREATION.md) - Geometry generation
