# Gaussian Garments: Reconstructing Simulation-Ready Clothing with Photo-Realistic Appearance from Multi-View Video
This is the official training and inference repository for the paper [Gaussian Garments: Reconstructing Simulation-Ready Clothing with Photo-Realistic Appearance from Multi-View Video](https://arxiv.org/abs/2409.08189).

## Table of Contents
- [Setup & Installation](#setup--installation)
    - [Prerequisites](#prerequisites)
    - [Environment Setup](#environment-setup)

## Setup & Installation
### Prerequisites
#### System Requirements
This setup assumes you have conda setup, enough disk space for the setup, and CUDA availibility. The version of your CUDA can be checked by
```bash
nvcc --version
```
Make sure you have a version that is compatible with pyTorch.

#### Clone [Gaussian Garments](https://github.com/hlimach/Gaussian-Garments) Repo
To avoid unnecessary complications, please clone the repo within a dedicated directory, as running the code creates directories within the parent directory.
```bash
git clone https://github.com/hlimach/Gaussian-Garments.git  --recursive
```

### Environment Setup
#### Create and activate conda environment
```bash
conda create -n gauss_env python=3.10
conda activate gauss_env
```
#### Install pyTorch that matches with your CUDA version
We used pytorch-cuda=11.8
```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Install bpy
It usually has a higher requirement to python env, so we install it early
```bash
pip install bpy
```

#### Clone [Animatable Gaussians](https://github.com/lizhe00/AnimatableGaussians) repo and install diff_gaussian_rasterization_depth_alpha
```bash
git clone https://github.com/lizhe00/AnimatableGaussians.git --recursive
cd AnimatableGaussians/gaussians/diff_gaussian_rasterization_depth_alpha
python setup.py install
cd ../../network/styleunet/
python setup.py install
cd ../../..
```

#### Clone [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) repo and install simple_knn
```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting/submodules/simple-knn/
python setup.py install
cd ../../..
```
#### Install remaining requirements using pip
```bash
pip3 install -r requirements.txt
```

#### Install COLMAP
Follow their available [installation guide](https://colmap.github.io/install.html) for system specific guidelines.

# 3D Garment Modeling Repository

## Repository Structure

```
3D-Garment-Modeling/
├── assets/                         # Documents (Paper, images, plots, etc.)
├── configs/                        # Configuration files
├── data/                           # Data storage
│   ├── images/                     # subject instance
│   │   ├── images/                 # RGB input frames
│   │   ├── masks/                  # Label and mask images
│   │   ├── cameras.json            # Cameras information
│   └── README.md                   # Detailed description of the data format requirements
├── models/                         # Pretrained or saved models
│   ├── checkpoints/                # Model checkpoints
│   ├── final/                      # Final trained models
│   └── README.md                   # Details on model architectures
├── notebooks/                      # Jupyter notebooks for experiments
├── scripts/                        # Bash scripts for running stages
├── src/                            
│   ├── datasets/                   # Data handling modules
│   ├── losses/                     # Model definitions
│   ├── models/                     # Loss functions
│   ├── stages/                     # Steps of Gaussian Garments pipeline
│   ├── utils/                      # Helper functions
│   └── __init__.py                 
├── README.md                       # Project overview and setup instructions
├── requirements.txt                # Python dependencies
├── .gitignore                      # Files and folders to ignore in Git
```