# RF Dataset Generation with Sionna

Complete guide for generating Radio-Frequency (RF) propagation datasets using NVIDIA Sionna RT ray-tracing.

## Overview

This process simulates RF signal propagation in the 3D scene and generates heatmap images showing received power at each camera viewpoint. The RF dataset is used to train the radio-frequency component of the RRF model.

---

## Script: generate_rf_dataset.py

## Usage:
```bash
# Ideal MPC spectrum (default)
python generate_rf_dataset.py

# Ideal AoD spectrum
python generate_rf_dataset.py --ideal --spectrum-type aod

# MVDR beamforming spectrum with default 4×4 array
python generate_rf_dataset.py --mvdr

# MVDR with custom array size and output dir
python generate_rf_dataset.py --mvdr --mvdr-m 8 --output-dir dataset_mvdr_8x8

# Custom scene file
python generate_rf_dataset.py --ideal --scene room_5x3x3.xml
```

### Purpose

Simulate 28 GHz mmWave RF propagation using Sionna RT ray-tracing and generate:
- RF power heatmaps (as grayscale PNG images)
- COLMAP-format camera poses (synchronized with visual dataset)
- Path features (gains, delays, angles) for each receiver position

### Requirements

- **Sionna** 0.18+ with TensorFlow GPU support
- **TensorFlow** 2.15+ with CUDA 11.8+
- **NumPy**, **SciPy**, **Matplotlib**
- **Input**: PLY meshes + Sionna XML scene file
- **Hardware**: 8GB+ GPU VRAM

---

## Configuration

Key parameters in `generate_dataset_ideal_mpc.py`:

```python
# === RF PARAMETERS ===
FREQUENCY = 28e9              # 28 GHz (5G mmWave)
BANDWIDTH = 1e9               # 1 GHz bandwidth
TX_POWER = 20                 # dBm (100 mW)

# === TRANSMITTER ===
TX_POSITION = (6.0, 2.5, 2.5) # Wall-mounted, centered
TX_ORIENTATION = (0, 0, 0)    # Euler angles (yaw, pitch, roll)

# === RECEIVER (CAMERA) ===
NUM_IMAGES = 300              # Match visual dataset
RESOLUTION = 800              # 800×800 pixels
FOCAL_LENGTH = 1164.69        # Calculated from camera_angle_x

# === RAY-TRACING ===
MAX_DEPTH = 5                 # Max reflections/diffractions
NUM_SAMPLES = 1e6             # Rays per viewpoint

# === SCENE ===
SCENE_XML = "room_5x3x3_fixed.xml"
```

---

## Workflow Steps

### 1. Scene Loading

**Load Sionna Scene**:
```python
from sionna.rt import load_scene, RadioMaterial

# Load scene with materials
scene = load_scene(SCENE_XML)

# Materials are assigned in XML:
# - concrete_* → itu_concrete
# - glass_* → itu_glass  
# - wood_* → itu_wood
# - metal_* → itu_metal
```

**Material Properties** (ITU-R P.2040-1):
| Material | Permittivity (εr) | Conductivity (σ) | Loss Tangent |
|----------|-------------------|------------------|--------------|
| Concrete | 5.24 | 0.0462 S/m | 0.0326 |
| Glass | 6.27 | 0.0043 S/m | 0.0020 |
| Wood | 1.99 | 0.0047 S/m | 0.0060 |
| Metal | - | ∞ (perfect conductor) | - |

### 2. Transmitter Setup

**Place Transmitter**:
```python
from sionna.rt import Transmitter, PlanarArray

# Create isotropic antenna
tx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V"
)

# Add to scene
tx = Transmitter(
    name="tx",
    position=TX_POSITION,
    orientation=TX_ORIENTATION
)
scene.add(tx)
tx.array = tx_array
```

**Transmitter Location**: `(6.0, 2.5, 2.5)`
- Mounted on X=6.5m wall (6.0m is 0.5m inward)
- Centered in Y dimension (2.5m from Y=0 and Y=5 walls)
- 2.5m height (mid-height in 3m tall room)

### 3. Camera Pose Synchronization

**Load Visual Dataset Poses**:
```python
import json

# Load transforms_train.json from visual dataset
with open("dataset_visual_v2/transforms_train.json") as f:
    transforms = json.load(f)

# Extract camera poses
camera_angle_x = transforms["camera_angle_x"]
focal_length = 0.5 * RESOLUTION / np.tan(0.5 * camera_angle_x)

poses = []
for frame in transforms["frames"]:
    T = np.array(frame["transform_matrix"])  # 4×4 C2W matrix
    poses.append(T)
```

**Coordinate System Conversion**:
- **Blender/NeRF**: OpenGL convention (Y-up, Z-backward)
- **Sionna**: Right-handed (Z-up, X-forward for antennas)

```python
def euler_to_quaternion(euler):
    # Rotation from COLMAP camera default to Sionna array
    R_posz2posx = Rotation.from_euler('ZYX', [-np.pi/2, 0.0, -np.pi/2])
    
    # Rotation of Sionna array to sampling direction
    yaw, pitch, roll = euler
    R_posx2array = Rotation.from_euler('ZYX', [yaw, pitch, roll])
    
    # Compose rotations
    R_w2c = R_posx2array * R_posz2posx
    R_c2w = R_w2c.inv()
    
    # Convert to quaternion (scalar-first for COLMAP)
    q = R_c2w.as_quat()  # [qx, qy, qz, qw]
    qvec_c2w = [q[3], q[0], q[1], q[2]]  # [qw, qx, qy, qz]
    return R_c2w, qvec_c2w
```

### 4. RF Ray-Tracing

**For Each Camera Pose**:

1. **Place Receiver**:
```python
from sionna.rt import Receiver

# Create receiver array (omnidirectional)
rx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V"
)

# Add receiver at camera position
rx = Receiver(
    name="rx",
    position=camera_position,
    orientation=camera_orientation
)
scene.add(rx)
rx.array = rx_array
```

2. **Compute Ray-Traced Paths**:
```python
# Configure ray-tracing
scene.frequency = FREQUENCY
scene.synthetic_array = True  # Use large virtual array

# Compute propagation paths
paths = scene.compute_paths(
    max_depth=MAX_DEPTH,
    num_samples=NUM_SAMPLES,
    los=True,          # Include line-of-sight
    reflection=True,   # Include reflections
    diffraction=True,  # Include diffractions
    scattering=False   # Disable diffuse scattering (slow)
)

# paths contains:
# - a: Complex path gains [batch, rx, tx, paths]
# - tau: Path delays [batch, rx, tx, paths]
# - theta_r, phi_r: AoA (azimuth, elevation)
# - theta_t, phi_t: AoD (azimuth, elevation)
```

3. **Generate 360° Panorama**:
```python
# Use Sionna camera for panoramic rendering
cam = scene.cameras['rx']
cam.position = camera_position
cam.orientation = camera_orientation

# Render coverage map (equirectangular projection)
cm = scene.coverage_map(
    max_depth=MAX_DEPTH,
    cm_cell_size=(10, 10),  # Angular resolution
    num_samples=NUM_SAMPLES
)

# cm.as_tensor() returns power map in dB
power_map_360 = cm.as_tensor().numpy()  # Shape: [H, W]
```

4. **Project to Perspective View**:
```python
class Equirectangular:
    def GetPerspective(self, FOV, THETA, PHI, height, width):
        """
        FOV: Field of view (degrees)
        THETA: Yaw (degrees)
        PHI: Pitch (degrees)
        height, width: Output image size
        """
        # Convert equirectangular to perspective
        # Uses inverse mapping: pixel → 3D ray → lon/lat → panorama pixel
        ...
        return perspective_image

# Convert panorama to perspective view
equ = Equirectangular(power_map_360)
rf_heatmap = equ.GetPerspective(
    FOV=49.13,  # Match camera_angle_x
    THETA=camera_yaw,
    PHI=camera_pitch,
    height=RESOLUTION,
    width=RESOLUTION
)
```

5. **Save RF Heatmap**:
```python
import cv2

# Normalize to 0-255 range
power_min, power_max = -100, -30  # dBm range
rf_normalized = np.clip((rf_heatmap - power_min) / (power_max - power_min), 0, 1)
rf_image = (rf_normalized * 255).astype(np.uint8)

# Save as grayscale PNG
cv2.imwrite(f"dataset_custom_scene_ideal_mpc/spectrum/frame_{i:04d}.png", rf_image)
```

### 5. COLMAP Export

**Generate cameras.txt**:
```python
# COLMAP camera intrinsics format:
# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
# For PINHOLE model: fx fy cx cy

fx = fy = focal_length
cx = RESOLUTION / 2.0
cy = RESOLUTION / 2.0

with open("cameras.txt", "w") as f:
    f.write(f"1 PINHOLE {RESOLUTION} {RESOLUTION} {fx} {fy} {cx} {cy}\n")
```

**Generate images.txt**:
```python
# COLMAP image format (two lines per image):
# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
# POINTS2D[] (empty line)

with open("images.txt", "w") as f:
    for i, (qvec, tvec) in enumerate(poses):
        img_id = i + 1
        qw, qx, qy, qz = qvec
        tx, ty, tz = tvec
        f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 frame_{i:04d}.png\n")
        f.write("\n")  # Empty line for POINTS2D
```

**Create points3D.txt**:
```python
# Dummy file (required by 3DGS but not used for RF)
with open("sparse/0/points3D.txt", "w") as f:
    f.write("# 3D point list with one line of data per point:\n")
    f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
    f.write("# Number of points: 1\n")
    f.write("1 0.0 0.0 0.0 128 128 128 0.0\n")
```

---

## Output Structure

```
dataset_custom_scene_ideal_mpc/
├── cameras.txt              # COLMAP intrinsics
├── images.txt               # COLMAP extrinsics
├── train_index.txt          # List of training images
├── test_index.txt           # List of test images
├── spectrum/                # RF heatmaps
│   ├── frame_0000.png       # Grayscale power map (800×800)
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ... (300 images)
└── sparse/
    └── 0/
        ├── cameras.txt      # Copy of intrinsics
        ├── images.txt       # Copy of extrinsics
        └── points3D.txt     # Dummy file
```

---

## Usage

### Basic Execution

```bash
# Ensure TensorFlow GPU is available
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Run RF dataset generation
python generate_dataset_ideal_mpc.py
```

### Monitor Progress

```
Loading Sionna scene...
✓ Loaded 7 meshes
Placing transmitter at (6.0, 2.5, 2.5)
Synchronizing with visual dataset...
✓ Loaded 300 camera poses

Generating RF heatmaps:
[001/300] Position: (3.2, 2.1, 1.5) | Paths: 243 | Max Power: -42.3 dBm
[002/300] Position: (4.5, 3.8, 1.8) | Paths: 189 | Max Power: -38.7 dBm
...
[300/300] Position: (2.1, 1.2, 2.2) | Paths: 267 | Max Power: -45.1 dBm

✓ Saved 300 RF heatmaps to spectrum/
✓ Saved COLMAP files to sparse/0/

Done! Total time: 3.2 hours
```

### Performance

- **GPU**: ~30-60 seconds per viewpoint (RTX 3080)
- **CPU**: ~5-10 minutes per viewpoint (not recommended)
- **Total**: 2.5-5 hours for 300 images

---

## Customization

### Change Frequency

```python
# For lower frequency (better penetration, less detail):
FREQUENCY = 2.4e9  # 2.4 GHz WiFi

# For higher frequency (higher resolution, more attenuation):
FREQUENCY = 60e9   # 60 GHz (WiGig)
```

### Increase Ray-Tracing Accuracy

```python
MAX_DEPTH = 10           # More reflections (slower)
NUM_SAMPLES = 5e6        # More rays (slower, smoother heatmaps)

# Enable diffuse scattering (very slow!)
paths = scene.compute_paths(scattering=True)
```

### Multiple Transmitters

```python
# Add second transmitter
tx2 = Transmitter(name="tx2", position=(1.0, 2.5, 2.5))
scene.add(tx2)
tx2.array = tx_array

# Compute combined coverage
cm = scene.coverage_map(...)  # Automatically includes all TXs
```

### Directional Antennas

```python
# Use dipole pattern instead of isotropic
tx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    pattern="dipole",  # or "hw_dipole", "tr38901", etc.
    polarization="V"
)
```

---

## Troubleshooting

### Issue 1: TensorFlow GPU Not Found

**Symptom**: "No GPU devices available"

**Solution**:
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall TensorFlow with GPU support
pip install tensorflow[and-cuda]==2.15.0

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Issue 2: Scene Loading Fails

**Symptom**: `AttributeError: 'Scene' object has no attribute 'mi_scene'`

**Causes**:
1. **Old Sionna version**: Update to 0.18+
2. **Invalid XML**: Check scene file syntax
3. **Missing PLY files**: Ensure all meshes exist

**Debug**:
```python
# Test scene loading
from sionna.rt import load_scene
scene = load_scene("room_5x3x3.xml")
print(scene.objects)  # Should list all meshes
```

### Issue 3: Black/Blank RF Heatmaps

**Symptom**: All RF images are completely black

**Causes**:
1. **No paths found**: TX and RX blocked by walls
2. **Wrong power scale**: dBm range outside normalization limits
3. **Coordinate system mismatch**: Camera looking wrong direction

**Debug**:
```python
# Check if paths exist
paths = scene.compute_paths()
print("Num paths:", paths.a.shape)
print("Path gains (dB):", 20*np.log10(np.abs(paths.a.numpy())))

# Check received power
power_dbm = 10*np.log10(np.sum(np.abs(paths.a.numpy())**2) * 1000)
print(f"Received power: {power_dbm:.2f} dBm")
```

**Fix**: Adjust normalization range
```python
power_min = power_dbm.min() - 10  # 10 dB below minimum
power_max = power_dbm.max() + 10  # 10 dB above maximum
```

### Issue 4: Memory Error

**Symptom**: `ResourceExhaustedError: OOM when allocating tensor`

**Solution**:
```python
# Reduce ray samples
NUM_SAMPLES = 1e5  # 10x fewer rays

# Reduce max depth
MAX_DEPTH = 3

# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
```

### Issue 5: Slow Performance

**Symptom**: >10 minutes per viewpoint

**Optimizations**:
```python
# Disable diffraction (major speedup)
paths = scene.compute_paths(diffraction=False)

# Reduce samples
NUM_SAMPLES = 5e5

# Use synthetic array (faster than full coverage map)
scene.synthetic_array = True
```

---

## Advanced Topics

### Path Feature Extraction

Extract detailed propagation metrics:

```python
def extract_path_features(paths):
    """Extract RF fingerprinting features."""
    # Path gains (linear scale)
    gains = np.abs(paths.a.numpy()).flatten()
    
    # Path delays (seconds)
    delays = paths.tau.numpy().flatten()
    
    # Angles of arrival (radians)
    theta_r = paths.theta_r.numpy().flatten()  # Azimuth
    phi_r = paths.phi_r.numpy().flatten()      # Elevation
    
    # Statistics
    features = {
        'path_gains': gains,
        'path_delays': delays,
        'total_power': np.sum(gains**2),
        'num_paths': len(gains),
        'delay_spread': np.std(delays),
        'mean_delay': np.mean(delays),
        'aoa_azimuth': theta_r,
        'aoa_elevation': phi_r
    }
    return features
```

### RF Localization Dataset

Save path features for localization:

```python
import pickle

dataset = []
for i, pose in enumerate(poses):
    paths = scene.compute_paths()
    features = extract_path_features(paths)
    
    dataset.append({
        'position': pose[:3, 3],        # Camera position
        'orientation': pose[:3, :3],    # Camera rotation
        'features': features,
        'image_id': i
    })

# Save dataset
with open('rf_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
```

Use for kNN localization (see `evaluate_localization.py`).

### Channel Impulse Response (CIR)

Compute time-domain CIR:

```python
from sionna.channel import cir_to_time_channel

# Get CIR
cir = scene.compute_cir()

# Convert to time-domain samples
h_time = cir_to_time_channel(
    cir.tau,
    cir.a,
    bandwidth=BANDWIDTH,
    l_min=0,
    l_max=100
)

# Plot CIR
plt.plot(np.abs(h_time.numpy()))
plt.xlabel('Sample')
plt.ylabel('|h[n]|')
plt.title('Channel Impulse Response')
plt.show()
```

---

## Quality Checklist

Before proceeding to 3DGS training:

- [ ] 300 RF heatmaps generated in `spectrum/`
- [ ] Heatmaps show realistic propagation patterns (not all black/white)
- [ ] Stronger signal near transmitter, weaker in shadowed regions
- [ ] `cameras.txt` and `images.txt` match visual dataset poses
- [ ] `sparse/0/` directory structure correct
- [ ] `train_index.txt` lists all images

**Visual Inspection**:
- Open a few heatmaps: Bright pixels near TX, dark in corners
- Check symmetry: Similar patterns at symmetric positions
- Verify furniture shadows: Tables/sofa block signal (darker regions)

---

## Next Steps

After generating RF dataset:
1. **Prepare data for 3DGS**: `python RF-3DGS/prepare_rf_data.py`
2. **Train visual model**: `python RF-3DGS/train.py -s dataset_visual_v2`
3. **Fine-tune RF model**: `python RF-3DGS/train.py --images spectrum --start_checkpoint ...`

---

**See also**:
- [Main README](../README.md) - Complete pipeline
- [3DGS Training](TRAINING.md) - Next step
- [Evaluation](EVALUATION.md) - Results analysis

---

**Key References**:
- Sionna RT Documentation: https://nvlabs.github.io/sionna/
- ITU-R P.2040-1: Effects of building materials and structures on radiowave propagation
