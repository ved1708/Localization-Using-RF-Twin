# Installation Guide

### 1. Clone RF-3DGS Framework
```bash
cd Project_1
git clone https://github.com/Wangmz-1203/RF-3DGS.git
cd RF-3DGS
```

### 2. Create Conda Environment
```bash
conda create -n rf-3dgs python=3.8
conda activate rf-3dgs
```

### 3. Install Dependencies
```bash
# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt

# Install Sionna for RF simulation
pip install sionna

# Install submodules (diff-surfel-rasterization, simple-knn)
cd submodules
pip install ./diff-surfel-rasterization
pip install ./simple-knn
cd ..
```

### 4. Install Blender (for visual dataset generation)
```bash
# Download Blender 3.6+ from https://www.blender.org/download/
# Or install via snap on Ubuntu:
sudo snap install blender --classic
```