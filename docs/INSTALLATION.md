# Installation Guide

### 1. Clone this Repository
```bash
git clone https://github.com/ved1708/Localization-Using-RF-Twin.git
cd Localization-Using-RF-Twin
```

### 2. Clone RF-3DGS Framework
```bash
git clone https://github.com/SunLab-UGA/RF-3DGS.git
cd RF-3DGS
```
Follow the RF-3DGS Installation and Training section in [Github Repo](https://github.com/SunLab-UGA/RF-3DGS) to install dependencies.

### 3. Create Conda Environment
```bash
conda create -n rf-3dgs python=3.8
conda activate rf-3dgs
```

### 4. Install Dependencies
```bash
# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt

# Install submodules (diff-surfel-rasterization, simple-knn)
cd submodules
pip install ./diff-surfel-rasterization
pip install ./simple-knn
cd ..
```

### 5. Install Blender (for visual dataset generation)
```bash
# Download Blender 3.6+ from https://www.blender.org/download/
# Or install via snap on Ubuntu:
sudo snap install blender --classic
```
### 6. Install NVIDIA Sionna (for RF simulation)
Generating the RF datasets requires NVIDIA Sionna. For detailed capabilities and system requirements, consult the [official Sionna documentation](https://nvlabs.github.io/sionna/index.html).

**Option A: Native Installation**
```bash
pip install sionna
```

**Option B: Docker Environment (Recommended for GPU/TensorFlow compatibility)**
If you encounter Python or TensorFlow GPU compatibility issues, we recommend look at our [docker sionna](https://github.com/ved1708/docker-sionna.git) repo.

**Follow the repository instructions to build and run the Sionna container.**