# Gaussian Garments: Reconstructing Simulation-Ready Clothing with Photo-Realistic Appearance from Multi-View Video
This is the official training and inference repository for the paper [Gaussian Garments: Reconstructing Simulation-Ready Clothing with Photo-Realistic Appearance from Multi-View Video](https://arxiv.org/abs/2409.08189).

## Table of Contents
- [Setup & Installation](#setup--installation)
    - [Prerequisites](#prerequisites)
    - [Environment Setup](#environment-setup)
- [Using Gaussian Garments](#training--evaluation)
    - [Training](#training)
    - [Evaluation](#evaluation)
- [Repository Structure](#repository-structure)

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
cd Gaussian-Garments
conda create -n gauss_env python=3.10
conda activate gauss_env
```
#### Run setup script
Please ensure to edit the script to match your pyTorch and CUDA versions.
```bash
bash setup.sh
```

#### Install COLMAP
Follow their available [installation guide](https://colmap.github.io/install.html) for system specific guidelines.

## Training & Evaluation
### Training
To train the model on 4D-Dress dataset, we recommend the following steps:
1. Ensure that the data input is prepared as detailed in the [data/README.md](./data/README.md)
2. Run the first stage by:
```bash
python initialisation.py --subject 00190/Inner --sequence Take2 --garment_type Inner
```
<details>
<summary>Parameters</summary>
| Parameter                        | Description                                                                     | Default      |
|----------------------------------|---------------------------------------------------------------------------------|--------------|
| `subject`                        | Subject folder name that contains the sequence folders                          |-             |
| `sequence`                       | Sequence identifier, folder containing cameras.json and camera folders          |-             |
| `garment_type`                   | The garment label to be processed                                               |-             |
| `camera`                         | Camera model used in COLMAP                                                     |PINHOLE       |
| `no_gpu`                         | Whether to use GPU for feature extraction and matching.                         |False         |
| `visualize`                      | Whether to visualize the results.                                               |False         |

### Evaluation

## Repository Structure

```
3D-Garment-Modeling/
├── data/                           # Data storage
│   ├── input/                      # Root to input data
│   ├── outputs/                    # Output root for each run
│   └── README.md                   # Detailed description of the data format requirements and output details
├── dependencies/                   # Dependency submodules cloned in this subdir
├── utils/                          # All of the source code 
│   ├── arg_utils.py                # Builds argument functions for each stage. 
│   ├── defaults.py                 # Stores default params such as the data root.
│   ├── initialisation_utils.py     # Functions for the initialisation stage.
│   ├── parse_scan.py               # 4D-Dress specific data processing.  
│   ├── preprocess_utils.py         # Prepares data for the initialisation stage.     
│   └── etc.py                 
├── setup.sh                        # bash script to handle setup.
├── initialisation.py               # first stage, can be bypassed if custom data is used.
├── train.py                        # training script, assumes first stage completed.
├── eval.py                         # script to visualise final outputs
├── README.md                       # Project overview and setup instructions
├── requirements.txt                # Python dependencies
├── .gitignore                      # Files and folders to ignore in Git
```