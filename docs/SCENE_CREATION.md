# 3D Scene Creation Scripts

This directory contains scripts for creating custom 3D room model for RF propagation simulation and 3D Gaussian Splatting.

## Scripts Overview

### 1. create_scene.py

**Purpose**: Generates a 3D room model with furniture as separate material-based PLY files.

**Features**:
- Creates 7m × 5m × 3m rectangular room
- Separate meshes for each material type (concrete, glass, wood, metal)
- Realistic furniture placement (tables, chairs, sofa, TV)
- Window and door cutouts in walls
- Combined PLY file for visualization

**Usage**:
```bash
cd helper_scripts
python create_scene.py
```

**Output**:
```
meshes_d/
├── floor.ply             - Floor surface (7m × 5m)
├── ceiling.ply           - Ceiling surface
├── walls.ply             - Walls with cutouts (3m height)
├── window.ply            - Window (2.5m × 1m × 0.02m)
├── door.ply              - Door (0.9m × 2.1m × 0.05m)
├── furniture.ply         - Tables (3x) with chairs (3x)
├── furniture_center.ply  - Coffee table + 2 sofas
├── led_tv.ply            - LED TV (1.2m × 0.7m × 0.05m)
└── metallic_cube.ply     - Metal cube on coffee table
room_combined.ply         - All objects combined with material IDs
```

**Room Layout**:
![alt text](assets/image.png)

**Furnitures**: Tables (3), Chairs (3), Coffee Table, Sofas (2), LED TV, Metallic Cube

**Coordinate System**:
- Origin: (0, 0, 0) at lower-left corner (front-left)
- X-axis: Room length (0m to 7m, left to right)
- Y-axis: Room width (0m to 5m, front to back)
- Z-axis: Height (0m to 3m, floor to ceiling)

**Customization**:
Edit these variables in the script:
```python
# Room dimensions
ROOM_WIDTH = 7.0    # X-dimension
ROOM_DEPTH = 5.0    # Y-dimension
ROOM_HEIGHT = 3.0   # Z-dimension

# Window (on back wall at Y=5)
window_x: (2.5, 5.0)     # X range: 2.5m to 5.0m
window_z: (1.0, 2.0)     # Z range: height 1.0m to 2.0m
window thickness: 0.02m  # ~2cm

# Door (on right wall at X=7)
door_y: (2.05, 2.95)     # Y range: 0.9m wide (centered at 2.5m)
door_z: (0, 2.1)         # Z range: height 0 to 2.1m
door thickness: 0.05m    # ~5cm

# Furniture dimensions
table_length = 1.17      # Along Y-axis
table_width = 0.76       # Along X-axis (depth from wall)
table_height = 0.76

sofa_length = 2.0        # Along X-axis
sofa_depth = 0.7         # Along Y-axis
sofa_seat_height = 0.30
```

---

## PLY File Format

All generated PLY files use **binary little-endian** format:

**Why binary?**
- Smaller file size
- Faster loading in Sionna/Blender/3DGS
- Required by Mitsuba scene loader

---

## Sionna Scene XML

To use generated meshes in Sionna RT, an XML file is generated automatically:
```
room_with_cube.xml   - Mitsuba 2.1.0 format scene with all objects
```

The XML includes:
- Material definitions (diffuse BSDF for concrete, wood, glass, metal)
- Shape references with corresponding material assignments
- PLY file paths (relative to script directory)

---

## Tips

### Mesh Quality
- Ensure **manifold geometry** for clean rendering (no holes, no intersecting faces)
- Use coarser meshes for large flat surfaces (floor, walls, ceiling) to reduce file size
- Each object is in a separate PLY file for material assignment flexibility
- The script auto-generates optimized binary PLY files

### File Organization
- `create_scene.py` is in the `helper_scripts/` directory
- Output directory structure created automatically:
  - `meshes_d/` — individual material-based meshes for Sionna
  - `room_combined.ply` — complete scene for visualization
  - `room_with_cube.xml` — scene configuration for simulations

## Next Steps

After creating the scene:
1. Verify PLY files are generated in `Project_1/meshes_d/`
2. Generate visual dataset → [VISUAL_DATASET.md](VISUAL_DATASET.md) (`generate_visual_dataset.py`)
3. Generate RF dataset → [RF_DATASET.md](RF_DATASET.md) (`generate_rf_dataset.py`)
4. (Optional) Visualize combined mesh in Blender by pulling `meshes_d/` files or in any 3D viewer using `room_combined.ply`

---
---

**See also**: [Main README](../README.md) for complete pipeline.
