# RFDT Construction Pipeline

```
┌─────────────────────┐
│  1. Scene Creation  │
│    create_scene.py  │
└──────────┬──────────┘
           │ PLY meshes
           ▼
┌─────────────────────┐
│ 2. Visual Dataset   │
│ generate_visual_    │
│     dataset.py      │
└──────────┬──────────┘
           │ RGB images + poses
           ▼
┌─────────────────────┐
│  3. RF Dataset      │
│ generate_dataset_   │
│   ideal_mpc.py      │
└──────────┬──────────┘
           │ RF heatmaps + COLMAP
           ▼
┌─────────────────────┐
│ 4a. Visual Training │
│    train.py         │
└──────────┬──────────┘
           │ Visual checkpoint
           ▼
┌─────────────────────┐
│ 4b. RF Fine-tuning  │
│    train.py --rf    │
└──────────┬──────────┘
           │ RRF model
           ▼
┌─────────────────────┐
│ 5. Evaluation       │
│ render.py + metrics │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 6. Visualization    │
│ WebGL Viewer        │
└─────────────────────┘
```

## Step-by-Step Workflow

---

## 1. Scene Creation

### 1.1 Generate 3D Room Model

Create a custom 7m × 5m × 3m room with furniture using parametric mesh generation:

```bash
python create_scene.py
```

**What it does:**
- Generates separate PLY files for each object (walls, floor, ceiling, furniture)
- Creates material-specific meshes for RF simulation
- Generates combined PLY: `room_combined.ply` for visualization

### 1.2 Verify Scene Scale

Ensure proper coordinate system and dimensions:

```bash
python check_scene_scale.py
```

---

## 2. Visual Dataset Generation

### 2.1 Generate RGB Images with Blender

Create photorealistic training images using Cycles renderer:

```bash
blender --background --python generate_visual_dataset.py
```

---

## 3. RF Dataset Generation

### 3.1 Simulate RF Propagation with Sionna

Generate RF heatmaps using ray-tracing simulation:

```bash
python generate_dataset_ideal_mpc.py
```

### 3.2 Prepare RF Data for 3DGS

Organize RF dataset into expected structure:

```bash
cd RF-3DGS
python prepare_rf_data.py
```

---

## 4. 3DGS Training

### 4.1 Train Visual Model (Stage 1)

First, train on visual RGB images to learn scene geometry:

```bash
cd RF-3DGS
conda activate rf-3dgs

python train.py \
  -s /home/ved/Ved/Project_1/dataset_visual_v2 \
  -m output/visual_model \
  --iterations 30000 \
  --save_iterations 7000 15000 30000
```

### 4.2 Train RF Model (Stage 2)

Fine-tune visual model on RF heatmaps:

```bash
python train.py \
  -s /home/ved/Ved/Project_1/dataset_custom_scene_ideal_mpc \
  -m output/rf_model \
  --images spectrum \
  --start_checkpoint output/visual_model/chkpnt30000.pth \
  --iterations 10000 \
  --save_iterations 3000 7000 10000
```

---

## 5. Evaluation

### 5.1 Render Test Views

Generate predictions for test set:

```bash
# Render visual test views
python render.py -m output/visual_model --iteration 30000

# Render RF test views
python render.py -m output/rf_model --iteration 10000
```

### 5.2 Compute Metrics

Evaluate reconstruction quality:

```bash
# Visual metrics
python metrics.py -m output/visual_model

# RF metrics
python metrics.py -m output/rf_model
```

---

## 6. Visualization

### 6.1 Interactive 3D Viewer

View reconstructed RRF in WebGL viewer:

```bash
cd RF-3DGS/SIBR_viewers
./build/bin/SIBR_gaussianViewer_app -m ../output/rf_model --iteration 10000
```

### 6.2 Generate Video

Create flythrough video:

```bash
python make_video.py \
  --input output/rf_model/test/ours_10000/renders \
  --output rf_reconstruction.mp4 \
  --fps 30
```
