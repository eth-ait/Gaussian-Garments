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