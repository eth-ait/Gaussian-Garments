#!/bin/bash

# # Initialize Conda (only needed for non-login shells)
# eval "$(conda shell.bash hook)"

# # create the environment
# conda create -n gg python=3.10

# # activate the environment
# conda activate gg
# echo "Conda environment 'gg' is now activated."

# install conda dependencies
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install cuda -c nvidia/label/cuda-11.8.0 -y
conda install gcc=11.4.0 gxx=11.4.0 -y
export CUDA_HOME=$CONDA_PREFIX

# bpy usually has a higher requirement to python env, so we install it early
pip install bpy

# install dependencies
mkdir dependencies
cd dependencies

# Clone Animatable Gaussians repo
git clone https://github.com/lizhe00/AnimatableGaussians.git --recursive
cd AnimatableGaussians/gaussians/diff_gaussian_rasterization_depth_alpha
python setup.py install
cd ../../network/styleunet/
python setup.py install
cd ../../..

# Clone Gaussian Splatting repo
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting/submodules/simple-knn/
python setup.py install
cd ../../../..

# install remaining requirements
pip install -r requirements.txt