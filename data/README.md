# Data Storage
The following serves as a detailed guide of how your data must be setup in order to run the Gaussian Garments pipeline.

## Folder Structure
```
data/
├── subject1_garment1/                # Subject and garment instance
│   ├── images/                       # RGB input frames
│   ├── masks/                        # Label and mask images
│   └── cameras.json                  # Cameras information 
├── subject1_garment2/                # Subject and garment instance
│   ├── ...
├── subject2_garment3/                # Subject and garment instance
├── ...
```