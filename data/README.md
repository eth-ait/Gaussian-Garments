# Gaussian garments: Reconstructing simulation-ready clothing with photorealistic appearance from multi-view video

### <img align=center src=./static/icons/project.png width='32'/> [Project](https://ribosome-rbx.github.io/Gaussian-Garments/) &ensp; <img align=center src=./static/icons/paper.png width='24'/> [Paper](https://arxiv.org/abs/2409.08189) &ensp;  

This is a repository for the paper [**"Gaussian garments: Reconstructing simulation-ready clothing with photorealistic appearance from multi-view video"**](https://ribosome-rbx.github.io/Gaussian-Garments/) (3DV2025).

This repository contains the implementation of the three stages described in the paper:
1. Geometry reconstruction (`s1_initiallisation.py`)
2. Garment registration (`s2_registration.py`)
3. Appearance reconstruction (`s3_appearance.py`)

This Readme file describes steps to 
* [Install the environment](#installation) 
* [Prepare input data for garment reconstruction](#data-preparation)
* [Create new Gaussian garments](#creating-new-gaussian-garments), and
* [Render the sequences of garment motions](#inference)

Unfortunately, we can not share the full data used in the paper. Hence, we have prepared the code for running Gaussian Garments algorithm on [ActorsHQ](https://actors-hq.com/) dataset. Notably, we found that the first stage used in the paper does not produce satysfying results on ActorsHQ, so we have prepared an alternative solution in the repository [ActorsHQ-for-Gaussian-Garments](https://github.com/hlimach/ActorsHQ-for-Gaussian-Garments).

**Follow instructions in [ActorsHQ-for-Gaussian-Garments](https://github.com/hlimach/ActorsHQ-for-Gaussian-Garments) to convert data from ActorsHQ into our format and reconstruct the garment mesh.** Then, you can continue from the stage 2 in this repository.

## Installation
To install the environment with all the required libraries, use the `setup.sh` script:
```bash
bash setup.sh
```
In case you encounter errors while running this script, we recommend running the lines from the script manually one by one to isolate the issue.

## Data Preparation
This repository assumes three key data locations:
* Inputs folder
* Outputs folder
* Auxilliary data

Your first step is to set paths to these three locations in `utils/defaults.py`:
```python
DEFAULTS['data_root'] = '/path/to/input/folder'
DEFAULTS['output_root'] = '/path/to/output/folder'
DEFAULTS['aux_root'] = f'/path/to/auxilliary/data/folder'
```

Then, refer to [DataPreparation.md](DataPreparation.md) for the instructions on how to set up these folders.

## Creating new Gaussian garments
To reconstruct a new Gaussian garment from multi-view videos, you'll need to run three stages: (1) initialize the geomentry from a single frame, (2) register this geometry to all multi-view videos, and (3) train an appearance model to reconstruct the garment's appearance. Here we'll describe each of these steps.

Before running the reconstruction, make sure you have followed instructions in [DataPreparation.md](DataPreparation.md) and your input and auxilliary folders hava the same structure as described there.

All three steps share the same argument names:
* `-s/--subject`: name of the subject in the input folder (`DEFAULTS.data_root/*subject_id*`)
* `-so/--subject_out`: name of the directory in the output folder where the results will be stored (`DEFAULTS.output_root/*subject_out_id*`). If not set, the same name as in `--subject` is used
* `-q/--sequence`: same of the sequence to use. There may be more than one sequence for one subject. (`DEFAULTS.data_root/*subject_id*/*sequence_id*`)
* `-tf/--template_frame`: the number of the template frame in the sequence. By default `-tf 0` is used.

### Step 1: Geometry initialization
*Note: if you want to run Gaussian Garments for ActorsHQ or another dataset for which the 1st stage described in the paper produces unsatisfying results, refer to [ActorsHQ-for-Gaussian-Garments](https://github.com/hlimach/ActorsHQ-for-Gaussian-Garments) repository. After reconstructing the geometry with that repo, procede directly with Step 2.*

Assuming you have correct structure of the input folder, to reconstruct the garment geometry from a template frame, run:
```bash
python s1_initialisation.py -s *subject_id* -q *sequence_id* -tf *template_frame_id*
```
For the template frame, use the one where the whole surface of the garment is fully visible and where the pose is similar to the first frame of each sequence.

This script will generate the folder `DEFAULTS.output_root/*subject_id*/stage1`, where the reconstructed template will be stored in `template.obj`.

The final action in this step is to create a UV unwrapping for the template mesh. **This have to be done manually.** You can use Blender and follow [this YouTube guide](TODO). **Save the .obj file of the mesh with the UV unwrapping to `DEFAULTS.output_root/*subject_id*/stage1/template_uv.obj`**

### Step 2: Garment registration
Once you have a garment template with uv unwrapping, you can register it to the multi-view videos. This process has two substeps.

First, we need to initialize the appearance of the garment. You can do this with this script:

*(Note that in the paper appearance initialization is described as a part of stage 1)*
```bash
python s2_registration.py -s *subject_id* -q *sequence_id*  -tf *template_frame_id*
```
You'll need to use the same sequence and template frame as in Step 1.

This will optimize the initial appearance and geometry for a single frame and save them into `DEFAULTS.output_root/*subject_id*/stage2/Template/`

Then, register this template to all sequences of this subject:
```bash
python s2_registration.py -s *subject_id* -q *sequence_id* 
python s2_registration.py -s *subject_id* -q *sequence_id2*
... 
```
This will produce registered meshes for each frame and save them into `DEFAULTS.output_root/*subject_id*/stage2/*sequence_id*/meshes/`. To check if it is running correctly, see the renders in `DEFAULTS.output_root/*subject_id*/stage2/*sequence_id*/renders/`.

Registering geometry to each frame takes a few minutes.


### Step 3: 
Finally, optimize the albedo texture and the appearance model using all registered sequences:
```bash
python s3_appearance.py -s *subject_id* 
```
This will store the training checkpoint into `DEFAULTS.output_root/*subject_id*/stage3/ckpt/`

### Inference
TODO
