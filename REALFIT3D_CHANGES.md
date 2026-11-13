# RealFit3D Changes to Gaussian Garments

[Traditional Chinese](REALFIT3D_CHANGES.zh-TW.md) | [README](README.md)

## Role in the Try-On Pipeline

ContourCraft produces a sequence of body and garment meshes. GG combines those
moving meshes with a trained garment appearance model to render images.
RealFit3D assembles the images into the website's turntable preview and video.

```text
Body and garment trajectories + UV templates + Stage 3 appearance checkpoints
    -> GG rendering
    -> turntable PNGs or animation frames
    -> RealFit3D preview and video
```

We maintain this fork against
[eth-ait/Gaussian-Garments](https://github.com/eth-ait/Gaussian-Garments),
baseline `ceb66fc36598da05b3b3391f4562cdc54d2a2ed2`.
RealFit3D pins it as the `upstream/Gaussian-Garments` submodule.
The changes concern rendering entries, configuration, input validation, and an
additional opacity controller trained after S3. The original S1 reconstruction,
S2 registration, and S3 appearance-training algorithms remain unchanged.

## Registration and Appearance Training I/O

Stage 2 still optimizes a fixed-topology garment mesh against each frame's images.
We convert point-cloud paths to strings for Open3D and export preview images as
clipped unsigned 8-bit RGB arrays for Pillow. These changes affect loading and
diagnostic output, not the registration objective.

Stage 3 consumes the registered mesh sequence, baked AO/normal maps, and calibrated
images. The appearance network and Gaussian parameters are optimized using the
existing training losses. Its entry now honors `--epochs` (default 5) and exposes
`--num_workers` (default 8). At each completed epoch it writes both `epochN/` and
`ckpt/`, so short sequences also produce the canonical checkpoint consumed by
rendering. Diagnostic texture panels are resized to fit the input image height.

## Rendering a Simulated Garment

For each frame, GG takes the simulated garment vertices and the body's position.
It places the garment Gaussians on the deformed mesh and prepares two UV maps
with Blender: an ambient occlusion (AO) map describing how surrounding surfaces
occlude the garment, and a normal map describing surface directions. The trained
AvatarNet uses these maps and the camera to update the garment's appearance.
GG then evaluates visibility, rasterizes the Gaussians, and composites the garment
with the body to produce the final image.

```text
Frame meshes -> UV AO/normal maps -> AvatarNet appearance update
             -> visibility and Gaussian rasterization -> composited image
```

This rendering stage uses an existing Stage 3 checkpoint; it does not retrain
the appearance model for each frame.

### Final-Image Rendering

[render_lean.py](render_lean.py) uses the same mesh preparation and appearance
model as `inference.py`, but renders only the final composited image.
The original entry also renders raw/style comparison panels and combines them
into a diagnostic image. Removing those extra panels reduces rendering and image
assembly work while retaining the image used by the website.

Run from this repository with the GG Python environment and configured data roots:

```bash
python render_lean.py \
  --traj_path /path/to/trajectory.pkl \
  --output_path /path/to/new-render-directory \
  --bake-samples 4
```

The output is a sequence of frame images. RealFit3D's preview worker assembles
these frames into an MP4.

### Turntable Rendering

[render_turntable.py](render_turntable.py) selects one simulated frame and moves
the camera around it. The mesh and UV AO/normal maps stay the same across angles,
so their cached files are reused. The appearance network, visibility calculation,
and rasterization are evaluated for each camera because the visible surface and
view-dependent appearance change with the angle.

```bash
python render_turntable.py \
  --traj_path /path/to/trajectory.pkl \
  --output_path /path/to/new-turntable-directory \
  --frame_idx 4 --n-angles 12 --bake-samples 4
```

The resulting PNGs let the website show several views of the outfit before the
full animation is ready.

### Bake Sampling

Blender baking converts the current 3D surface information into the UV maps used
by AvatarNet. `--bake-samples` sets the Cycles sampling budget for this step.
The new entries default to 4, compared with 16 in the original `inference.py`.
A lower setting reduces the bake sampling budget and can increase AO-map noise.
Both entries retain the existing 512 x 512 map dimensions expected by the model.
Changing this option does not change the simulation or retrain the checkpoint.

## Selecting Data and Models

We added environment-variable overrides to [utils/defaults.py](utils/defaults.py)
so RealFit3D can select the input data, trained garment assets, and auxiliary
models for each deployment without editing GG source files.

| Variable | Contents |
|---|---|
| `REALFIT3D_GG_DATA_ROOT` | GG input data |
| `REALFIT3D_GG_OUTPUT_ROOT` | Reconstructed garment assets and trained appearance checkpoints |
| `REALFIT3D_GG_AUX_ROOT` | Auxiliary models and data |

The renderer loads each named garment's `stage1/template_uv.obj`,
`stage1/cameras.json`, and `stage3/ckpt/net.pt` from the asset root.
Rendered media goes to the command's `--output_path`.

## Checking the Trajectory

The trajectory stores multiple garments in a single vertex sequence.
Its `garment_names` order tells `Simulation` which UV template and appearance
model belong to each portion of that sequence. Before rendering, our checks
require valid names, a nonempty predicted sequence, the expected total vertex
count in every frame, and matching body/garment frame counts.
An incompatible input stops with an error before rendering.

Names are supplied explicitly by the simulation worker rather than inferred
from the trajectory filename. Asset preparation also needs matching face/UV
indices and coordinate conventions. Use trusted pickle files: deserialization
occurs before these checks.

## Opacity Control

All three rendering entries now use [opacity_runtime.py](opacity_runtime.py)
before Gaussian rasterization. `original` preserves the existing rendering.
`direct` multiplies each Gaussian's base opacity by the requested scale.
`optical_depth` applies `1 - (1 - alpha)^s`. In V3, a small network takes the base opacity and normalized
canonical position and produces eight positive, normalized weights. These define
a monotonic piecewise-linear function of the requested scale; its output changes
the Gaussian opacity through `1 - (1 - alpha)^m(s)` before the same rasterization
step. `s=0` hides the garments; `s=1` preserves their original opacity exactly.
One scale applies to the selected garments; each selected garment has its own V3 weights.
`--opacity-garments Actor02_upper` changes only that garment; omitting the flag selects
the whole outfit. Unselected garments retain their original opacity.

After loading Stage 3, `Doll` updates the template face coordinates and captures
a detached copy of its Gaussian positions. V3 normalizes this fixed reference
with the controller checkpoint's center and scale. Simulation updates the rendered
geometry but does not move the controller's reference coordinates. Capturing after
checkpoint loading includes the learned Gaussian offsets.

Training freezes the Stage 3 model and renders its original alpha image A1.
For a sampled control value s, the target is s times A1. The controller changes
the per-Gaussian opacities, and the differentiable rasterizer renders a new alpha
image. Training minimizes its foreground error against the target, updating only
the controller. The saved controller is used alongside the Stage 3 checkpoint
at inference time.

[train_opacity_v3.py](train_opacity_v3.py) reads the subject's Stage 2 meshes,
point clouds and baked AO/normal maps, calibrated camera metadata and image
dimensions, plus its Stage 3 checkpoint. The default protocol uses frames 0-29
for training, 30-39 for validation, 5,000 Adam steps, and up to eight cameras.
The loss is foreground-weighted alpha L1 plus `0.001 * mean((p - 1/8)^2)`.
It saves `controller.pt`, the configuration, training log, and validation metrics.
Existing output directories are rejected.

The camera reader combines a full calibration dictionary with the Stage 2
view list. Calibration supplies the focal lengths, principal point,
world-to-camera transform, and image shape; the list limits training to the
registered views. It checks each listed view's dimensions, focal lengths and
pose against that calibration before selecting up to eight training cameras.
The resulting cameras render the alpha targets and controller predictions.
RGB photos are not training targets for this step.

Calibration is read from `<data_root>/<subject>/<sequence>/cameras.json`, then
a full Stage 2 calibration dictionary, then `<output_root>/<subject>/stage1/cameras.json`.
With calibration `shape: [height, width]` and a Stage 2 view list, no input image
directory is needed. Legacy inputs without `shape` use a PNG/JPEG for dimensions;
without a Stage 2 list, calibrated image directories define the available views.
A rendering-camera list alone cannot recover the principal point. Training
configuration records the calibration and selection paths and SHA256 hashes.
The configured data and auxiliary root directories must still exist.

Check cameras on CPU, without importing Torch or loading checkpoints:

```bash
python opacity_training_cameras.py \
  --data-dir /data/Actor02_upper/Sequence1 \
  --stage1-dir /assets/Actor02_upper/stage1 \
  --stage2-dir /assets/Actor02_upper/stage2/Sequence1
```

```bash
python train_opacity_v3.py --subject Actor02_upper --sequence Sequence1
python render_turntable.py \
  --traj_path /path/to/trajectory.pkl --output_path /path/to/new-preview \
  --frame_idx 0 --n-angles 12 --opacity-mode v3 --opacity-scale 0.5 \
  --v3-controller Actor02_upper=/assets/Actor02_upper/opacity_controller_v3/controller.pt \
  --v3-controller Actor02_lower=/assets/Actor02_lower/opacity_controller_v3_valid_cameras/controller.pt
```

The same opacity arguments work with `render_lean.py` and `inference.py`.
Without explicit mappings, V3 looks for exactly one controller in either
`opacity_controller_v3/controller.pt` or
`opacity_controller_v3_valid_cameras/controller.pt` inside each garment directory.
Missing, ambiguous, mismatched-subject, or invalid-normalization checkpoints stop
the run. Checkpoints are installed separately, not bundled in Git.

The controller retains the team's V3 `4 -> 32 -> 32 -> 8` network and checkpoint
keys. We replaced the Actor02-only training paths with subject/sequence arguments,
added output protection, and connected the renderer used by RealFit3D.
This changes rendered transparency, not cloth mechanics or material refraction.
Source file hashes and GPU test results are recorded in the
[integration record](https://github.com/Realfit3D/realfit3d/blob/main/docs/OPACITY_INTEGRATION_20261020.md).

## Version and Validation

The rendering entries, path overrides, and input checks were introduced in
`8a52dc6`. Controller and training-helper tests run with
`python -m unittest discover -s tests -v` in a Torch environment.
`tests/smoke_opacity_render.py` checks real checkpoints through both rendering
entries, including original/V3 endpoints and a half-opacity image.
Full body fitting and cloth simulation are separate acceptance tests.
Retraining the generalized V3 entry requires the complete Stage 2 training assets.

For development and version updates, see
[RealFit3D's development guide](https://github.com/Realfit3D/realfit3d/blob/main/docs/DEVELOPMENT.md).
