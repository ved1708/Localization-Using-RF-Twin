# Localization Pipeline

While the RFDT Construction Pipeline focuses on building the Radio-Frequency Digital Twin mapping, this pipeline leverages that digital twin out for physical coordinate mapping (Localization) by matching incoming RF signals (like Channel State Information) to specific indoor spatial positions.

Our framework implements multiple approaches to localization, categorized incrementally from baselines to fine-grained gradient-descent tracking.

## 1. Traditional Fingerprinting Baselines (`Regressive_models/`)

This module bypasses 3DGS entirely to act as a baseline comparison. It learns a direct mapping from RF properties to coordinate positions.

* **`train_nn_localizer.py`**: Trains a Multi-Layer Perceptron (MLP) mapping extracted RF features directly to spatial vectors (x, y, z).
* **`evaluate_localization.py`**: Implements a traditional k-Nearest Neighbors (kNN) strategy. For a new RF signal, it searches the dataset for the closest matching historical RF fingerprints and averages their coordinates.

## 2. Digital Twin Data Generation (`rfdt_based_nn/`)

To expand beyond limited physical captures, we use the trained digital twin to generate vast amounts of simulated data.

* **`generate_rf3dgs_dataset.py`**: Automatically queries bulk viewpoints from the trained RF-3DGS model and/or the Sionna simulator. It generates dense datasets representing RF features in the environment, suitable for training robust Convolutional Neural Networks (CNNs) that can predict location arrays instantly.

## 3. Coarse Localization: Render & Compare (`grid_search_localization.py`)

When an RF observation is fed into the system without prior positional knowledge, we apply Coarse Localization.

1. Defines a constrained 3D grid consisting of multiple $X, Y, Z$, and Yaw coordinates.
2. At each grid point, it renders a synthetic RF signature using the trained 3DGS digital twin model.
3. It structurally compares the predicted images (via SSIM/L1) to the actual "target" RF signal. The grid coordinate with the highest structural similarity is returned as the predicted coarse pose.

## 4. Fine-Grained Localization: Optimization-based (`gradient_descent_localization.py`)

With a coarse position identified, the pipeline uses optimization to dial into the exact location.

1. Takes a starting hypothesis (either from the grid-search step or the last known position) and iteratively refines the physical camera pose $(x,y,z,\text{yaw})$.
2. Uses PyTorch's automatic differentiation to back-propagate the rendering loss (L1/SSIM/LPIPS differences between the rendering and the true RF measurement) directly into the camera pose parameters.
3. Effectively guides the virtual "camera" into the exact location where the target RF signal was recorded.

## 5. Live Tracking Demonstration (`DEMO/`)

The above components are pieced together for continuous positional tracking over time in a moving agent simulation.

* **`demo_pipeline.py`**: Accepts a list of trajectory waypoints. For each point, it triggers the Sionna simulator to create a live CSR reading, runs the gradient descent localization to find the target, and outputs the resulting prediction coordinates iteratively.
* **`demo_server.py` & `demo.html`**: A web server and interactive frontend that poll the live localization results from the pipeline to visualize the agent's movement through the digital twin environment in real-time.
