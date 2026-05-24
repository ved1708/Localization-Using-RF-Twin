# Evaluation and Localization

Comprehensive guide for evaluating RRF reconstruction quality, machine learning benchmarks (k-NN, MLP), and performing RRF-optimized localization and trajectory tracking.

## Overview
After training, follow these steps to assess quality and performance:
1. **Render test views** to generate predicted heatmaps/images.
2. **Compute quantitative metrics** (PSNR, SSIM, LPIPS).
3. **Evaluate localization accuracy** via regression (ML) or RRF-optimization.
4. **Visualize results** in the interactive 3D viewer.

---

## 1. Rendering Test Views

### Visual Model (Stage 1)
```bash
cd RF-3DGS
python render.py -m output/visual_model --iteration 30000 --skip_train
```

### RF Model (Stage 2)
```bash
python render.py -m output/rf_model --iteration 40000 --skip_train
```

### Parameters
| Parameter | Description |
|-----------|-------------|
| `-m, --model_path` | Path to the trained model directory |
| `--iteration` | Specific checkpoint iteration to load |
| `--skip_train` | Skip rendering training views|

### Output Structure
```text
output/[model]/test/ours_[iter]/
├── renders/    # Predicted output (PNG)
└── gt/         # Ground truth reference (PNG)
```

---

## 2. Quantitative Metrics
```bash
python metrics.py -m output/visual_model  # RGB metrics
python metrics.py -m output/rf_model      # RF metrics
```

| Metric | Target | Formula / Description |
|--------|--------|-----------------------|
| **PSNR** | High | $10 \log_{10} \frac{255^2}{\text{MSE}}$ (Pixel accuracy) |
| **SSIM** | High | Structural similarity index [0, 1] |
| **LPIPS**| Low | Deep perceptual similarity [0, 1] |

---

## 3. Regression Localization
Evaluates accuracy using machine learning models in `Regressive_models/`. Both scripts expect the `rf_dataset.pkl` dataset to be available in the parent directory that is generated using the python script for the same scene used for RFDT Modelling.

### 3.1 k-NN (Fingerprinting)
Evaluates k-Nearest Neighbors localization automatically for `k=3`, `5`, and `10`.
```bash
cd Regressive_models
python evaluate_localization.py
```
**Outputs**: `knn_errors_k{k}.csv` and `localization_results_k{k}.png`.

### 3.2 MLP (Coordinate Regression)
Trains and evaluates a neural network (MLP) to output 3D coordinates.
```bash
cd Regressive_models
python train_nn_localizer.py
```
**Outputs**: `mlp_errors.csv` and `nn_localization_results.png`.

---

## 4. RFDT Pipeline: RRF-Optimization
Core localization method using the trained RFDT model.  
You can directly use the model by downloading rf_model_dealy_3.5ghz foler [here](https://drive.google.com/drive/folders/1YbHeys8ySAJiGE7M5ZK3kAeqkQy0jUS_?usp=sharing) for localization task. Put it in RF-3DGS/output or follow [Training](TRAINING.md) to train the model from scratch.

### 4.1 Frame Generation and Gradient Descent

Run the gradient descent optimizer to estimate its original position:

```bash
python gradient_descent_localization.py \
  --target_image localisation_frames_3.5ghz/delay_0.40_0.40_1.20_0.png \
  --model_path RF-3DGS/output/rf_model_delay_3.5ghz \
  --iteration 40000
```
Before localization, if you need an RF spectrum image representing the target position. Generate these test frames using `generate_csi_dataset.py` by specifying the receiver coordinates(yaw optionally):

```bash
# Generate a single localization frame at position (X=0.4, Y=0.4, Z=1.2)
python generate_csi_dataset.py \
  --ideal --spectrum-type delay \
  --rx-pos 0.4 0.4 1.2 \
  --output-dir localisation_frames_3.5ghz
```
*Note: You can also use `--rx-pos-file locations.txt` to batch-generate multiple test frames. Example of locations.txt content:  
0.4 0.4 1.2  
0.4 0.4 1.8  
...*

**How it works:**
1. **Coarse Estimate:** The script internally invokes grid search logic (`grid_search_localization.py`) to systematically render and compare views, finding the nearest anchor position.
2. **Gradient Descent:** Using the 3D Gaussian Splatting engine, it directly backpropagates the pixel loss (LPIPS + SSIM) into the virtual camera's coordinates (X, Y, Z), walking the camera pose down the loss gradient to the continuous precise location.

### 4.2 Batch Evaluation
To evaluate localization accuracy across all generated test frames:

```bash
./evaluate_all.sh
```
*Note: You can edit `TARGET_DIR` and `MODEL_PATH` directly inside `evaluate_all.sh` to evaluate a different dataset or model.*

**Outputs:**
*   **Live Console Table:** Real-time display of Ground Truth (GT) position, Estimated Position, Yaw, Grid Search error, Gradient Descent error, and execution time per frame.
*   **Stats:** Mean Squared Error (MSE), Average Time, Min Error, and Max Error displayed in the terminal and saved to `evaluation_results.csv`.

---

## 5. Live Trajectory Tracking Demo
*Note: Make sure Flask is installed before running the demo server (`pip install flask`).*
1. **Start Server**: `cd DEMO && python demo_server.py`
2. **Access UI**: Open `http://localhost:5000` in your browser.
3. **Run**: Press **"Start"** to begin real-time trajectory tracking (Solid line = GT, Dashed Line = Pred).

**Note: You can customize the trajectory path by editing `waypoints.txt` within the `DEMO` directory.**

---

## 6. Visualization
*   **Antimatter Viewer**: Drag and drop the `point_cloud.ply` from `output/rf_model_delay_3.5ghz/point_cloud/iteration_40000/`.
*   **SIBR Viewer**: `./SIBR_viewers/bin/SIBR_gaussianViewer_app -m output/rf_model_delay_3.5ghz/point_cloud/iteration_40000/`

---

**See also**: [Main README](../README.md) \| [Training Guide](TRAINING.md)


---

**Evaluation Complete!** 🎉
