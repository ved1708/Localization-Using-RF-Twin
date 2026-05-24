# RF Dataset Generation with Sionna

Complete guide for generating Radio-Frequency (RF) propagation datasets using NVIDIA Sionna RT ray-tracing.

## Overview

This process simulates RF signal propagation in the 3D scene and generates heatmap images showing received power at each camera viewpoint. The RF dataset is used to train the radio-frequency component of the RRF model.

---

## Script: generate_rf_dataset_0.py
### Purpose

Simulate 3.5 GHz RF propagation using Sionna RT ray-tracing (Sionna 0.18+) and generate:
- **RF Power Heatmaps**: Grayscale (MPC) or RGB (Delay/AoD/Phase/Beamforming).
- **COLMAP Poses**: Poses generated for 192 physical locations (3 orientations each).
- **Normalization Stats**: phase_norm_stats.txt generated automatically for Phase mode.

### Requirements

- **Sionna** 0.18+ with TensorFlow GPU support
- **TensorFlow** 2.15+
- **Input Files**:
    - **Sionna XML Scene**: `room_with_cube.xml` (Defines geometry & materials. Generated in [SCENE_CREATION.md](SCENE_CREATION.md)).
    - **PLY Meshes**: Located in `meshes_d/` folder (Referenced by the XML).
## Usage:
```bash
# Ideal MPC (Grayscale)
python generate_rf_dataset_0.py --ideal --spectrum-type mpc

# AoD Spectrum (RGB)
python generate_rf_dataset_0.py --ideal --spectrum-type aod

# Propagation Delay Spectrum (RGB)
python generate_rf_dataset_0.py --ideal --spectrum-type delay

# Phase-aware Spectrum (RGB: R=Acos, G=Asin, B=Amp)
python generate_rf_dataset_0.py --ideal --spectrum-type phase

# CBF (Delay-resolved Beamforming)
python generate_rf_dataset_0.py --cbf --mvdr-m 4

# MVDR (Capon) Beamforming - Sharp nulls
python generate_rf_dataset_0.py --mvdr --mvdr-m 4

# Specific Scene
python generate_rf_dataset_0.py --ideal --scene room_with_cube.xml
```

---

## Preparation for Training

After generating the RF dataset, run the preparation script to organize the files and create the train/test splits required for the RF fine-tuning stage of RF-3DGS.

**Usage**:
```bash
cd helper_scripts
python prepare_rf_data.py
```
This script automates the COLMAP structure creation and index generation. **Make sure the root path in prepare_rf_data.py is correctly configured to right dataset path.**

**Required Directory Structure**:
The script expects the dataset at `Project_1/dataset_ideal_...` and will:
1.  Create the `sparse/0/` directory.
2.  Move `cameras.txt` and `images.txt` into the sparse folder.
3.  Generate a dummy `points3D.txt` (required for 3DGS compatibility).
4.  Create `train_index.txt` and `test_index.txt` using an even distribution.

---

## Configuration

Key parameters in `generate_rf_dataset_0.py`:

```python
# === RF PARAMETERS ===
FREQUENCY = 3.5e9             # 3.5 GHz (Sub-6 GHz)
WAVELENGTH = 299792458 / FREQUENCY

# === TRANSMITTER ===
TX_POSITION = (0.01, 2.5, 2.9) # Near wall, high mount
TX_PATTERN = "iso"             # Isotropic

# === RECEIVER (CAMERA) ===
RESOLUTION = 600              # 600×600 pixels
H_FOV = 120                   # 120° Horizontal FOV
# 3 orientations per location (0°, 120°, 240° yaw)

# === RAY-TRACING ===
MAX_DEPTH = 2                 # 2 reflections per path
NUM_SAMPLES = 5e5             # 0.5M rays per viewpoint
SCATTERING = True             # Enabled for MPC realism (coeff=4)
```

---

## Workflow Steps

### 1. Scene Loading & Materials

**Material Physics (3.5 GHz via ITU-R P.2040-2)**:
The script automatically assigns scattering-aware materials to meshes based on keywords:

| Keyword | Material | εr | Conductivity (σ) | Scat. Coeff |
|----------|----------|----|------------------|-------------|
| floor, wall, ceiling, pillar | Concrete | 5.24 | 0.123 S/m | 0.4 |
| furniture, door | Wood | 1.99 | 0.018 S/m | 0.8 |
| window | Glass | 6.27 | 0.019 S/m | 0.1 |
| tv, led, cube | Metal | 1.00 | 1.0e7 S/m | 0.1 |

### 2. Physical Scanning Strategy

The script generates a 3D grid of measurements:
- **X**: 12 steps (0.3m to 6.7m)
- **Y**: 8 steps (0.3m to 4.7m)
- **Z**: 2 heights (1.2m and 2.5m)
- **Orientations**: 3 distinct yaw angles per position to cover 360°.

### 3. Coordinate System

- **Sionna**: Standard right-handed (Z-axis is UP).
- **COLMAP**: Standard vision coordinate system (Y is DOWN).
- **Conversion**: Handled via `euler_to_quaternion` which includes a `posz2posx` rotation to align the Sionna array primary axis with the camera look-at direction.

### 4. Spectrum Projection

The script uses a two-step projection process:
1. **Ray-Tracing Panorama**: Computes paths to generate a 360° Equirectangular heatmap.
2. **Perspective Warp**: Projects the equirectangular map into a perspective view matching the camera FOV (120°) and orientation.

### 5. Normalization

- **MPC/AoD/Delay**: Uses a global dB range (Max/Min) determined by sampling 50 random positions before generation.
- **Phase-aware**: Uses 99th percentile stats for R/G (Phase) and B (Amp) to ensure consistent color mapping across the whole dataset.

---

## Output Structure (Final)

```
dataset_ideal_mpc/
├── spectrum/                # Heatmap Images (00001.png, ...)
├── sparse/
│   └── 0/
│       ├── cameras.txt      # Moved here by prepare_rf_data.py
│       ├── images.txt       # Moved here by prepare_rf_data.py
│       └── points3D.txt     # Created dummy file
├── train_index.txt          # Training set indices
├── test_index.txt           # Test set indices (20 images)
└── phase_norm_stats.txt     # Only for phase-aware mode
```

---

## Performance

- **GPU (RTX 3080/4090)**: ~2-5 seconds per view.
- **Total Generation**: ~20-40 minutes for the full 576-image dataset.
- **Storage**: ~150-300MB per dataset.

---

## Quality Checklist

- [ ] Check `images.txt`: Ensure 576 frames exist.
- [ ] Open a few images: MPC should be grayscale; AoD/Phase should be vibrant RGB.
- [ ] High contrast: Objects (tables/sofa) should cast clear radio shadows.
- [ ] COLMAP check: Positions should match the 12x8x2 grid pattern.

---

## Next Steps

After preparing the RF dataset:
1. **Train visual model**: Use [TRAINING.md](TRAINING.md) to train the base visual 3DGS model.
2. **RF Fine-tuning**: Once the visual model is ready, use the prepared RF dataset to train the Radio-Frequency component.
3. **Evaluation**: Analyze the signal prediction accuracy using [EVALUATION.md](EVALUATION.md).

---

**See also**:
- [Main README](../README.md) - Complete pipeline
- [Visual Dataset Generation](VISUAL_DATASET.md) - Rendering RGB images
- [3DGS Training](TRAINING.md) - Model training