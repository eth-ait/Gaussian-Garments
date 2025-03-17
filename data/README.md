# Gaussian garments: Reconstructing simulation-ready clothing with photorealistic appearance from multi-view video

### <img align=center src=./static/icons/project.png width='32'/> [Project](https://ribosome-rbx.github.io/Gaussian-Garments/) &ensp; <img align=center src=./static/icons/paper.png width='24'/> [Paper](https://arxiv.org/abs/2409.08189) &ensp;  

This is a repository for the paper [**"Gaussian garments: Reconstructing simulation-ready clothing with photorealistic appearance from multi-view video"**](https://ribosome-rbx.github.io/Gaussian-Garments/) (3DV2025).

This repository contains the implementation of the three stages described in the paper:
1. Geometry reconstruction (`s1_initiallisation.py`)
2. Garment registration (`s2_registration.py`)
3. Appearance reconstruction (`s3_appearance.py`)

Unfortunately, we can not share the full data used in the paper. Hence, we have prepared the code for running Gaussian Garments algorithm on [ActorsHQ](https://actors-hq.com/) dataset. Notably, we found that the first stage used in the paper does not produce satysfying results on ActorsHQ, so we have prepared an alternative solution.

## Installation
To install the environment with all the required libraries, use `setup.sh` script:
```bash
bash setup.sh
```
In case you encounter errors while running this script, we recommend running the lines from this script manually one by one to isolate the issue.

## Data Preparation


## Creating new Garments

## Inference

# Data Storage
The following serves as a detailed guide of how your data must be setup in order to run the Gaussian Garments pipeline. Note that if custom initialisation is done, the folder in `data/outputs/subject_id/stage1` MUST contain the template.obj and template_uv.obj files for the remaining stages to run.

## Folder Structure
```
data/
├── input/                            # this is where DEFAULTS.data_root points
│   ├── subject_id/                   # (e.g. 00190/Outer)
│   │   ├── Sequence/                 # (e.g. Take1)
│   │   │   ├── cameras.json          # Camera parameters file
│   │   │   ├── Meshes/smplx          # SMPLX data
│   │   │   ├── camera_id/            # (e.g. 0000)
│   │   │   │   ├── capture_images/   # RGB input frames
│   │   │   │   ├── capture_labels/   # Label and segmentation masks
│   │   │   ├── ...
│   │   ├── ...
│   ├── ...
├── outputs/                          # this is where DEFAULTS.output_root points
│   ├── subject_id/                   # (e.g. 00190/Outer)
│   │   ├── stage1/                   # Outputs from initialisation stage
│   │   │   ├── images/               # first frame of each camera
│   │   │   ├── masks/                # binary segmentation masks of garment
│   │   │   ├── txt/                  # COLMAP compatible cameras.json output
│   │   │   │   ├── images.txt        
│   │   │   │   ├── cameras.txt       
│   │   │   │   └── points3D.txt      
│   │   │   ├── sparse/               # sparse recon. output from COLMAP
│   │   │   ├── dense/                # dense recon. output from COLMAP
│   │   │   ├── template.obj          # IMPORTANT: final output from stage 1
│   │   │   ├── template_uv.obj       # IMPORTANT: final output's manual UV mapping.
│   │   ├── ...
│   ├── ...
```