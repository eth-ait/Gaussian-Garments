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

After the installation, you can activate the environment with
```bash
conda activate gaugar
```

## Data Preparation
This repository assumes three key data locations:
* Input folder
* Output folder
* Auxilliary data folder

Refer to [DataPreparation.md](DataPreparation.md) for the instructions on how to set up these folders.

Then, your first step is to set paths to these three locations in `utils/defaults.py`:
```python
DEFAULTS['data_root'] = '/path/to/input/folder'
DEFAULTS['output_root'] = '/path/to/output/folder'
DEFAULTS['aux_root'] = '/path/to/auxilliary/data/folder'
```



## Creating new Gaussian garments
To reconstruct a new Gaussian garment from multi-view videos, you'll need to run three stages: (1) initialize the geomentry from a single frame, (2) register this geometry to all multi-view videos, and (3) train an appearance model to reconstruct the garment's appearance. Here we'll describe each of these steps.

Before running the reconstruction, make sure you have followed instructions in [DataPreparation.md](DataPreparation.md) and your input and auxilliary folders hava the same structure as described there.

All three steps share the same argument names:
* `-s/--subject`: name of the subject in the input folder (`DEFAULTS.data_root/*subject_id*`)
* `-so/--subject_out`: name of the directory in the output folder where the results will be stored (`DEFAULTS.output_root/*subject_out_id*`). If not set, the same name as in `--subject` is used
* `-q/--sequence`: name of the sequence to use. There may be more than one sequence for one subject. (`DEFAULTS.data_root/*subject_id*/*sequence_id*`)
* `-tf/--template_frame`: the number of the template frame in the sequence. By default `-tf 0` is used.

### Step 1: Geometry initialization
*Note: if you want to run Gaussian Garments for ActorsHQ or another dataset for which the 1st stage described in the paper produces unsatisfying results, refer to [ActorsHQ-for-Gaussian-Garments](https://github.com/hlimach/ActorsHQ-for-Gaussian-Garments) repository. After reconstructing the geometry with that repo, procede directly with Step 2.*

Assuming you have correct structure of the input folder, to reconstruct the garment geometry from a template frame, run:
```bash
python s1_initialisation.py -s *subject_id* -q *sequence_id* -tf *template_frame_id*
```
For the template frame, use the one where the whole surface of the garment is fully visible and where the pose is similar to the first frame of each sequence.

This script will generate the folder `DEFAULTS.output_root/*subject_id*/stage1`, where the reconstructed template will be stored in `template.obj`.

The final action in this step is to create a UV unwrapping for the template mesh. **This have to be done manually.** You can use Blender and follow [this YouTube guide](https://www.youtube.com/watch?v=LqJGD6yjlDE). **Save the .obj file of the mesh with the UV unwrapping to `DEFAULTS.output_root/*subject_id*/stage1/template_uv.obj`**

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

Registering geometry to one frame takes a few minutes.


### Step 3: 
Finally, optimize the albedo texture and the appearance model using all registered sequences:
```bash
python s3_appearance.py -s *subject_id* 
```
This will store the training checkpoint into `DEFAULTS.output_root/*subject_id*/stage3/ckpt/`

### Step 4 and creating trajectories:
In step 4, we optimize the behavior of the garment by finetuning a ContourCraft graph neural network.

To perform this step, please refer to the [ContourCraft](https://github.com/Dolorousrtur/ContourCraft/) repository and specifically to [this notebook](https://github.com/Dolorousrtur/ContourCraft/blob/main/GaussianGarments.ipynb). This notebook also shows how to simulate the garments with ContourCraft in order to create trajectory files needed to run the inference step of Gaussian Garments.

Note that you will need to follow the [installation instructions for ContourCraft](https://github.com/Dolorousrtur/ContourCraft/blob/main/INSTALL.md) before running the finetuning process.


### Inference
To render a dynamic sequence of Gaussian garment geometries, use `inference.py` script:
```bash
python inference.py --traj_path *trajectory_file*.pkl --output_path *directory_to_store_renders*
``` 

Here, `trajectory_file` is a path to a `.pkl` file containing trajectories of the garments and the body meshes. They follow the structure of the trajectory files produced by [ContourCraft](https://github.com/dolorousrtur/ContourCraft) and contain a dictionary with the following elements:

* `pred`: 3D positions of garment vertices in each frame, np.array of shape [N, V, 3], where N is the number of frames and V is the number of the garment vertices.
* `cloth_faces`: faces of the garment meshes, np.array of shape [F, 3]
* `obstacle`: 3D positions of body vertices in each frame, np.array of shape [N, B, 3], where N is the number of frames and B is the number of the body vertices.
* `obstacle_faces`: faces of the body mesh, np.array of shape [Fb, 3]
* `garment_names`: a list of subject names for the garments used in the trajectory. The subject names should correspond to the folders in your `DEFAULTS.output_root` directory. 

A trajectory file may contain a trajectory for a multi-garment outfit comprised of several garments. In this case, 

* `pred` is a concatenation of vertex positions for all garments, 
* `cloth_faces` is the concatenation of their faces with properly renumerated vertex ids, and 
* `garment_names` is the list of subject names in the same order as the garments are concatenated in.

The script `inference.py` will check the  `garment_names` list and load the corresponding checkpoints stored in your `DEFAULTS.output_root` directory. 

See the link to examples of the trajectory files below.

#### Render sequences from the supplementary video
You can download the trajectory files used in the supplementary video [here](https://drive.google.com/file/d/1VoCmCfL-YmWL4BsgcdN8gxcEt3djJp-v/view?usp=drive_link).

To render them, you will need to download checkpoints of the Gaussian Garments used in the paper [here](https://drive.google.com/file/d/1EnmIQQ3BIN9nqykhEAOv18R5mH1eq5fx/view?usp=sharing). Unpack them to your `DEFAULTS.output_root` so that its structure is:

```
DEFAULTS.output_root/
├── 00122_outer/
├── 00134_upper/
└── ...
```

Then, you will be able to render the trajectories with the command above.