# Visual Dataset Generation with Blender

Complete guide for generating photorealistic RGB image datasets for 3D Gaussian Splatting training.

## Overview

This process creates a synthetic dataset of RGB images with accurate camera poses using Blender's Cycles renderer. The dataset serves as the foundation for training the visual geometry component of the RRF model.

---

## Script: generate_visual_dataset.py

### Purpose

Generate 300 high-quality RGB images from diverse camera viewpoints inside the 3D room scene, formatted for 3DGS training.

### Requirements

-   **Blender 3.6+** (with GPU support recommended)
-   **Python modules**: `bpy`, `mathutils` (built into Blender)
-   **Input**: PLY meshes from `meshes/` directory
-   **Hardware**: 8GB+ GPU VRAM for fast rendering

---

## Configuration

Key parameters in `generate_visual_dataset.py`:

```python
# === PATHS ===BASE_DIR = "/home/ved/Ved/Project_1"INPUT_MODELS_DIR = os.path.join(BASE_DIR, "meshes")OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "dataset_visual_v2")# === DATASET SIZE ===NUM_IMAGES = 300          # Total images to generateRESOLUTION = 800          # Image resolution (square: 800×800)# === ROOM BOUNDS ===ROOM_MIN = mathutils.Vector((0.5, 0.5, 0.0))  # Minimum (X, Y, Z)ROOM_MAX = mathutils.Vector((6.5, 4.5, 3.0))  # Maximum (X, Y, Z)CENTER = (ROOM_MIN + ROOM_MAX) / 2             # Room center (3.5, 2.5, 1.5)# === MATERIAL COLORS ===MATERIAL_COLORS = {    "walls": (0.85, 0.85, 0.85, 1),      # Light gray    "floor": (0.25, 0.25, 0.25, 1),      # Dark gray    "ceiling": (0.95, 0.95, 0.95, 1),    # White    "door": (0.4, 0.2, 0.1, 1),          # Brown    "window": (0.7, 0.8, 1.0, 1),        # Transparent blue-tint    "furniture": (0.55, 0.35, 0.15, 1),  # Wood brown    "led_tv": (0.05, 0.05, 0.05, 1)      # Black}
```

---

## Workflow Steps

### 1. Scene Setup

**Import Meshes**:

-   Loads all PLY files from `meshes/` directory
-   Assigns materials based on filename prefix:
    -   `concrete_*` → walls/floor/ceiling material
    -   `glass_*` → window material (semi-transparent)
    -   `wood_*` → door/furniture material
    -   `metal_*` → TV material

**Material Creation**:

-   Uses **Principled BSDF** shader node for PBR rendering
-   Special handling for glass:
    
    ```python
    if is_glass:    bsdf.inputs['Transmission Weight'].default_value = 0.65  # Semi-transparent    bsdf.inputs['Roughness'].default_value = 0.3            # Slight roughness    # Add procedural noise for "smudges"    noise = nodes.new(type='ShaderNodeTexNoise')    noise.inputs['Scale'].default_value = 50.0
    ```
    

### 2. Lighting Setup

**HDRI Environment**:

-   Loads `interior_HDRI.exr` for realistic ambient lighting
-   Fallback: Create artificial area lights if HDRI not found

**Area Lights** (fallback):

# Visual Dataset Generation Scripts

## Overview

Two separate scripts for generating photorealistic visual datasets for 3D Gaussian Splatting training:

1.  **`generate_visual_dataset.py`**: Generate 325 training images (static scene, multi-view coverage)
2.  **`generate_single_visual_dynamic.py`**: Generate 1 image from top corner (dynamic scene with metallic cube)

## Features

-   ✅ Realistic RGB colors and material properties (metallic, glass, procedural textures)
-   ✅ Blender 5.0 compatible (PLY import with fallback)
-   ✅ GPU-accelerated rendering (CUDA/OptiX)
-   ✅ Automatic material detection (metallic cube, glass windows, furniture, etc.)
-   ✅ Skip existing frames on re-run (resume capability)

## Usage

### Generate 325 Training Images (Static Scene)
```bash
blender --background --python /home/ved/Ved/Project_1/generate_visual_dataset.py -- \
  --meshes_dir=meshes_d \
  --output_dir=dataset_visual \
  --test_ratio=0.10 \  
  --test_block_size=8
```

**Output**:

-   `dataset_visual_v2/images/` (325 images: frame_0000.png to frame_0324.png)
-   `dataset_visual_v2/transforms_train.json` (293 frames, 90%)
-   `dataset_visual_v2/transforms_test.json` (32 frames, 10%)

**Time**: ~15 minutes (GPU)

### Generate Single Image (Dynamic Scene with Cube)

```bash
blender --background --python generate_single_visual_dynamic.py
```

**Output**:

-   `dynamic_scene_visual/images/dynamic_frame_0000.png`
-   `dynamic_scene_visual/transforms.json`

**Time**: ~3 seconds (GPU)

**Camera Position**: Top corner (6.5, 4.5, 2.8) looking at room center (2.5, 1.5, 1.0) for maximum scene coverage

## Command-Line Arguments

### generate_visual_dataset.py

Argument

Default

Description

`--meshes_dir`

`meshes`

Directory containing PLY mesh files

`--output_dir`

`dataset_visual_v2`

Output directory

### generate_single_visual_dynamic.py

No command-line arguments - uses hardcoded configuration for consistency

## Material System

### RGB Color Mapping

The script automatically assigns realistic colors based on filename keywords:

```python
MATERIAL_COLORS = {    "walls": (0.85, 0.85, 0.85, 1),       # Light gray    "floor": (0.25, 0.25, 0.25, 1),       # Dark gray    "ceiling": (0.95, 0.95, 0.95, 1),     # Almost white    "door": (0.4, 0.2, 0.1, 1),           # Brown wood    "window": (0.7, 0.8, 1.0, 1),         # Light blue glass    "furniture": (0.55, 0.35, 0.15, 1),   # Wood brown    "furniture_center": (0.55, 0.35, 0.15, 1),    "pillar": (0.85, 0.85, 0.85, 1),    "led_tv": (0.05, 0.05, 0.05, 1),      # Black (metallic)    "metallic_cube": (0.7, 0.7, 0.7, 1)   # Silver gray (metallic)}
```

### Material Properties

-   **Metallic objects** (`metallic_cube`, `led_tv`): Metallic=0.9, Roughness=0.1
-   **Glass objects** (`window`, objects with "glass" in name): Transmission=0.65, Roughness=0.3
-   **Standard objects**: Procedural noise textures (2 scales) + bump mapping

## Lighting Setup

Consistent 3-light setup for all renders:

1.  **Background**: Strength=0.6 (ambient fill)
2.  **Sun**: Energy=2.5, Angle=0.5 (key light, 45° elevation, 15° rotation)
3.  **Ceiling Panel**: Energy=250W, Size=4.0m, Warm white (1.0, 0.98, 0.9)

## Render Settings

-   Engine: Cycles (GPU-accelerated)
-   Resolution: 800x800 pixels
-   Samples: 96 (denoised with OpenImageDenoise)
-   Max bounces: 3
-   Camera lens: 20mm (wide angle for room scale)

## Output Format

### Training Dataset (generate_visual_dataset.py)

-   **transforms_train.json** - 293 frames (90% of dataset)
-   **transforms_test.json** - 32 frames (10% of dataset)

Format:

```json
{  "camera_angle_x": 1.5707963,  "frames": [    {      "file_path": "images/frame_0000",      "transform_matrix": [[...], [...], [...], [...]]    }  ]}
```

### Single Image (generate_single_visual_dynamic.py)

-   **transforms.json** - Single frame with time=0

Format:

```json
{  "camera_angle_x": 1.5707963,  "frames": [    {      "file_path": "images/dynamic_frame_0000",      "transform_matrix": [[...], [...], [...], [...]]      "time": 0    }  ]}
```

## Workflow Examples

### Radio Radiance Field (RRF) Training Workflow

1.  **Train static RRF**:
    
    ```bash
    # Generate 325 visual imagesblender --background --python generate_visual_dataset.py --   --meshes_dir=meshes --output_dir=dataset_visual_v2# Generate 200 RF images (separate script)python generate_dataset_ideal_mpc.py# Train 3DGS/RRFpython train.py --source_path dataset_visual_v2 ...
    ```
    
2.  **Generate dynamic scene data for temporal fine-tuning**:
    
    ```bash
    # Single visual image with metallic cubeblender --background --python generate_single_visual_dynamic.py# Single RF image (separate script)python generate_single_rf_dynamic.py
    ```
    
3.  **Temporal fine-tune**:
    
    ```bash
    python train.py --source_path dynamic_scene_visual --temporal_finetune ...
    ```
    

## Resume Capability

If the script is interrupted, simply re-run it. The script will:

-   ✓ Skip rendering frames that already exist
-   ✓ Still capture camera transforms for all frames
-   ✓ Generate JSON files at the end

This makes it safe to restart without losing progress!

## Troubleshooting

### Blender 5.0 Compatibility

The script includes automatic fallback for PLY import:

```python
try:    bpy.ops.wm.ply_import(filepath=filepath)  # Blender 5.0+except AttributeError:    bpy.ops.import_mesh.ply(filepath=filepath)  # Older versions
```

### GPU Not Detected

If rendering is slow, check GPU setup in output:

```
✓ Enabled GPU: NVIDIA RTX A6000 (CUDA)✓ Enabled GPU: NVIDIA RTX A6000 (OPTIX)
```

If you see "Falling back to CPU rendering", ensure CUDA/OptiX is available.

### Inconsistent Colors

Ensure meshes follow naming convention:

-   `room_walls.ply` → Light gray walls material
-   `furniture_center.ply` → Wood brown material
-   `metallic_cube.ply` → Metallic silver material

## Performance

-   **Single image**: ~3 seconds (GPU)
-   **325 images**: ~15 minutes (GPU, first run)
-   **325 images**: ~30 seconds (GPU, re-run with skip existing)
-   **Storage**: ~650KB per image (PNG, 800x800, RGBA)

## Notes

-   The script respects scene boundaries (7m x 5m x 3m room)
-   Camera height in train mode varies: 0.5m (floor) to 2.5m (ceiling views)
-   Single mode ideal for top-corner surveillance-style views
-   Pinhole camera model (20mm lens) works well with 3DGS training

**Solution**:

1.  Check GPU availability:
    
    ```python
    prefs = bpy.context.preferencescycles_prefs = prefs.addons['cycles'].preferencesprint([d.name for d in cycles_prefs.devices])
    ```
    
2.  Enable GPU in Blender UI: Edit → Preferences → System → CUDA/OptiX
3.  Install NVIDIA drivers: `sudo ubuntu-drivers autoinstall`

### Issue 3: Black Images

**Symptom**: Rendered images are completely black

**Possible Causes**:

1.  **No lights**: Ensure HDRI or area lights are enabled
2.  **Camera inside wall**: Check collision detection
3.  **Wrong render engine**: Ensure `scene.render.engine = 'CYCLES'`

**Debug**:

```python
# Manually test render one framebpy.ops.render.render(write_still=True)
```

### Issue 4: Incorrect Camera Poses

**Symptom**: 3DGS training fails with "camera pose error"

**Verification**:

```python
# Check transform matrixT = cam.matrix_worldprint("Camera position:", T.translation)print("Camera rotation:", T.to_euler())
```

**Expected**: Camera should be inside room bounds `(0.5-6.5, 0.5-4.5, 0.0-3.0)`

---

## Advanced Tips

### Add Motion Blur

```python
scene.render.use_motion_blur = Truescene.render.motion_blur_shutter = 0.5
```

### Add Depth-of-Field

```python
cam.data.dof.use_dof = Truecam.data.dof.focus_distance = 3.0cam.data.dof.aperture_fstop = 2.8
```

### Export Depth Maps

```python
# Enable depth outputscene.use_nodes = Truenodes = scene.node_tree.nodesrender_layers = nodes['Render Layers']depth_output = render_layers.outputs['Depth']# Save depth as EXR
```

### Use Real HDRI

Download from [Poly Haven](https://polyhaven.com/hdris):

```python
hdri_path = "path/to/hdri/interior_modern_01_4k.exr"env_texture = nodes.new(type='ShaderNodeTexEnvironment')env_texture.image = bpy.data.images.load(hdri_path)
```

---

## Quality Checklist

Before proceeding to 3DGS training, verify:

-    300 images generated successfully
-    `transforms_train.json` and `transforms_test.json` exist
-    Images are sharp, well-lit, no black frames
-    Camera poses are diverse (not all from same corner)
-    Images have high overlap (neighboring views share features)
-    No missing textures or pink materials in renders
-    Resolution matches configuration (800×800)

---

## Next Steps

After generating the visual dataset:

1.  **Verify data**: Open a few images to check quality
2.  **Generate RF dataset**: `python generate_dataset_ideal_mpc.py`
3.  **Train 3DGS visual model**: `python RF-3DGS/train.py -s dataset_visual_v2`

---

**See also**:

-   [Main README](../README.md) - Complete pipeline
-   [RF Dataset Generation](RF_DATASET.md) - Next step
-   [3DGS Training](TRAINING.md) - Model training

---

**Rendering Tips**:

-   Use **OptiX denoiser** (faster than OpenImageDenoise)
-   Enable **adaptive sampling** for faster convergence
-   Save intermediate checkpoints every 50 frames
-   Render in batches if memory limited