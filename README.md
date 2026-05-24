# RF Digital Twin Modeling and Localization for Indoor Wireless Systems

This repository presents an end-to-end architecture for reconstructing **Radio-Frequency Radiance Fields (RRF)** from custom indoor 3D scenes using **Radio Frequency 3D Gaussian Splatting (RF-3DGS)** and leveraging this digital twin for continuous **indoor trajectory tracking and localization**.

By using neural volumetric representations (3DGS), we create digital twins for evaluating wireless characteristics (Path Gains, Delays, AoA/AoD) and performing accurate indoor localization.

---

## 📍 Localization Demo & Results

Our approach enables robust, continuous tracking of users in indoor environments by matching real-time RF measurements against the 3DGS reconstructed radio environment.

### Trajectory Tracking 
> *Tracking Demo*
> ![Trajectory Tracking](DEMO/assets/demo.gif)

This tracking is facilitated through iterative optimization algorithms built directly on top of the differentiable 3DGS renders, alongside neural network-based regressors trained on the RF environment dataset.

---

## 📊 Localization Metrics

We comprehensively evaluate our localization approaches utilizing various RF signatures (Ideal MPC, Delay) and compare traditional optimization against deep learning models:

- **Average Localization Error**: Achieves sub-meter accuracy (~0.5m average error), pushing boundaries in multi-path rich environments.
- **Optimization Methods**: Includes zero-shot optimization evaluations utilizing Grid Search (`grid_search_localization.py`) and Gradient Descent (`gradient_descent_localization.py`).


---

## 📡 RF-3DGS Modeling Results

Before localization can occur, the system generates a surrogate scene. It primarily trains a visual model using RGB datasets and then fine tunes the established geometry using spatially-aware RF heatmaps.

### 1. Visual Geometry Reconstruction
- **Scene Context**: Parametric 7m × 5m × 3m room populated natively with mixed-material furniture (Concrete walls, Glass windows, Wooden tables, Metallic objects/screens).
- **Quality**: Reaches PSNR ~30 dB and SSIM ~0.94 from 800x800 resolution renders.

### 2. Multi-path RF Reconstruction
- **Simulation Settings**: Designed for 3.5 GHz & 28 GHz (mmWave 5G) bands using NVIDIA Sionna RT framework.
- **Capabilities**: Extrapolates distinct RF responses (reflection, diffraction, scattering) seamlessly across continuous unseen spatial locations.
- **Dataset Modalities**: Generated multi-modal configurations encompassing power mappings, delays, and MVDR spatial spectra (e.g., `dataset_ideal_mpc`, `dataset_ideal_delay`).

---

## 🧭 Localization Methodology

The localization pipeline serves as the analytical bridge processing raw spatial RF data through the RRF model:
1. **Data Acquisition**: Extracs multi-path properties (Path Gains, Delays).
2. **Cost formulation Formulation**: Calculates positional gradient differences actively mapping the measured RF fingerprints physically against dynamically generated RF-3DGS test scenes.
3. **Position Solving**:
    - **Optimization Tracking**: Scans grids or steps down calculated gradients relying heavily on differentiable geometry renders.
    - **Regressive Features**: Directly maps features vectors to spatial limits utilizing high-speed regressors for live inference.

---

## 🔗 Documentation Links

For detailed replication instructions, architecture flowcharts, and codebase operation explanations, please select the specified documentation files below:

- **[🛠️ Installation Guide](installation.md)**: Dependencies, Conda environments, Sionna ray-tracing & Blender setup.
- **[🏗️ RFDT Construction Pipeline](rfdt_pipeline.md)**: Sequential structure covering Scene Creation → Visual Generation → RF Processing → RF-3DGS Training.
- **[🎯 Localization Pipeline](localization_pipeline.md)**: Comprehensive guide dissecting the Regressive Neural Networks alongside optimization strategies deployed for tracking.

---

## 👤 Author & Acknowledgments
- **Project & Repository**: [ved1708/Localization-Using-RF-Twin](https://github.com/ved1708/Localization-Using-RF-Twin)
- **Core Frameworks & Tools**: Adapted around [RF-3DGS](https://github.com/Wangmz-1203/RF-3DGS), [NVIDIA Sionna](https://nvlabs.github.io/sionna/), [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) & Blender.

