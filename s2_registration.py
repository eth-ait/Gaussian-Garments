#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import os
from pathlib import Path
from scene.cross_scene import crossScene
import torch
import numpy as np
from PIL import Image
from random import randint
from utils.defaults import DEFAULTS
from utils.loss_utils import l1_loss, ssim
from sklearn import neighbors
from gaussian_renderer import render
import sys
from scene import Scene, Dataloader, MeshGaussianModel
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, GaussianClothParams
import torch.nn.functional as F
from aitviewer.scene.camera import PinholeCamera
from aitviewer.renderables.meshes import Meshes
from aitviewer.headless import HeadlessRenderer

def prepare_output_and_logger(args):    
    if not args.subject_out:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.subject_out = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder

    stage2_path = Path(args.subject_out) / DEFAULTS.stage2 / args.sequence
    stage2_path.mkdir(parents=True, exist_ok=True)
    print("Output folder: {}".format(stage2_path))
    os.makedirs(args.subject_out, exist_ok = True)
    with open(stage2_path / "cfg_args", 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def ait_render(viewer, v, f, cam, vb=None, fb=None):
    v = v.detach().cpu().numpy()
    f = f.cpu().numpy()

    mesh = Meshes(v, f, draw_edges=True)
    mesh.backface_culling = False
    target = mesh.current_center
    position = cam.T + (target - cam.T) * 0.2
    w, h = cam.image_width, cam.image_height

    viewer.scene.nodes = viewer.scene.nodes[:4]
    viewer.scene.add(mesh)

    if vb is not None and fb is not None:
        body = Meshes(vb, fb, color=(173/255, 0/255, 211/255, 1.0), draw_edges=True)
        body.backface_culling = False
        viewer.scene.add(body)

    camera = PinholeCamera(position, target, w, h, viewer=viewer)
    viewer.set_temp_camera(camera)
    image = np.array(viewer.get_frame()).transpose(2, 0, 1)

    return torch.tensor(image)


# TODO: rewrite this function
def logger(loss, iteration, max_iter):
    # TODO: what's acc and why catch exception?
    global acc
    try:
        acc = {f'AVG_{k}': (acc[f'AVG_{k}']*(iteration-1)+v) / iteration for k, v in loss.items()}
    except:
        acc = {f'AVG_{k}': v for k, v in loss.items()}

    # Progress bar
    total_loss_for_log = np.array([v.item() for v in loss.values()]).sum()
    if iteration % 10 == 0:
        progress_bar.set_postfix({"Loss": f"{total_loss_for_log:.{7}f}"})
        progress_bar.update(10)
    if iteration == max_iter:
        progress_bar.close()

def saver(viewer, gaussians, scene, args, bg):
    current_frame = scene.current_frame

    print('current_frame', current_frame)

    if current_frame == 0:
        scene.save(current_frame, args.is_template)

    stage2_path = Path(args.subject_out) / DEFAULTS.stage2 
    if args.is_template:
        mesh_path = stage2_path / "Template" / "template.obj"
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        gaussians.save_mesh(mesh_path)
    else:
        sequence_path = Path(args.subject_out) / DEFAULTS.stage2 / args.sequence
        # save mesh
        mesh_path = sequence_path / "meshes"
        os.makedirs(mesh_path, exist_ok=True)
        gaussians.save_mesh(os.path.join(mesh_path, f"frame_{current_frame:05d}.obj"))

    # store rendered imgs
    render_cam = scene.getTrainCameras()[0]
    gt_img = render_cam.original_image.detach().cpu()
    img = render(render_cam, gaussians, args, bg)["render"].detach().cpu()

    if viewer is None:
        h, w = gt_img.shape[-2:]
        viewer = HeadlessRenderer(size=(2*w, 2*h))

    penalize = render_cam.gt_alpha_mask
    penalize = torch.cat([penalize,penalize,penalize]).detach().cpu()
    diff = torch.abs(img - gt_img)
    _ait = ait_render(viewer, gaussians.mesh.v, gaussians.mesh.f, render_cam, gaussians.mesh.body.vertices, gaussians.mesh.body.faces) / 255
    col1 = torch.cat([gt_img, penalize], axis=1)
    col2 = torch.cat([img, diff], axis=1)
    container = torch.cat([col1, col2, _ait], axis=-1)

    container = Image.fromarray(np.array(container*255.0, dtype=np.byte).transpose((1, 2, 0)), "RGB")

    if args.is_template:
        render_path = stage2_path / "Template" / "template_render.png"
    else:
        render_path = sequence_path / "renders" / f"{current_frame:05d}.png"
    render_path.parent.mkdir(parents=True, exist_ok=True)
    container.save(render_path)

    return viewer
        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")


    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser) # used in the renderer


    parser.add_argument('-s', '--subject', type=str, required=True, default='')
    parser.add_argument('-so', '--subject_out', type=str, default='')
    parser.add_argument('-t', '--template_seq', type=str, default='')
    parser.add_argument('-q', '--sequence', type=str, required=True, default='')

    parser.add_argument('-tf', '--template_frame', type=int, default=None)

    parser.add_argument("--first_frame_iterations", type=int, default=10000)
    parser.add_argument("--first_frame_iterations_cross", type=int, default=15000)
    parser.add_argument("--other_frame_iterations", type=int, default=5000)
    parser.add_argument("--collision_iteration", type=int, default=2000)
    parser.add_argument("--ff_collision_iteration", type=int, default=2000)
    parser.add_argument('--start_from', type=int, default=-1)

    parser.add_argument("--camera", default="PINHOLE", type=str) # only used for cross scene
    args = parser.parse_args(sys.argv[1:])

    

    args.sh_degree = 0 
    args.debug = False
    if len(args.subject_out) == 0:
        args.subject_out = args.subject
    args.subject_out = Path(DEFAULTS.output_root) / args.subject_out


    args.is_template = args.template_frame is not None
    args.is_template_seq = args.is_template
    
    if not args.is_template_seq:
        args.first_frame_iterations = args.first_frame_iterations_cross
        args.cross_from = Path(DEFAULTS.output_root) / args.subject_out / DEFAULTS.stage2 / args.template_seq

    prepare_output_and_logger(args)
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    
    # build components
    dataloader = Dataloader(args)
    gaussians = MeshGaussianModel(args)


    if args.is_template:
        scene = Scene(args, dataloader, gaussians)
    else:
        scene = crossScene(args, dataloader, gaussians)

    # w, h = 940, 1280
    # viewer = HeadlessRenderer(size=(2*w, 2*h))
    viewer = None

    stage2_path = Path(args.subject_out) / DEFAULTS.stage2 / args.sequence

    # reconstruct and optimize
    frames_iterator = range(dataloader._len) if not args.is_template else [args.template_frame]
    for t in frames_iterator:
        is_first_frame = (t==0) or args.is_template
        collision_iteration = args.ff_collision_iteration if is_first_frame else args.collision_iteration
        iterations = args.first_frame_iterations + collision_iteration if is_first_frame else args.other_frame_iterations

        # TODO: useful for testing, remove just before publising
        ############ DEBUG ############ 
        # skip first frame if it already exists 
        if is_first_frame and (stage2_path / "point_cloud").exists(): 
            continue 
        # skip if we are starting from a specific frame
        if t < args.start_from:
            continue
        ############ DEBUG ############

        
        scene.prepare_frame(t, is_first_frame)
        gaussians.training_setup(opt, is_first_frame and args.is_template_seq)
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        viewpoint_stack = None

        desc = "{} frame{} --> {}/{}".format("Reconstruct" if is_first_frame else "Optimize", scene.current_frame, t+1, dataloader._len)
        progress_bar = tqdm(range(iterations), desc=desc)

        iterations = 10
        for iter in range(1, iterations+1):
            use_body = iter > iterations - collision_iteration
            # first frame remove collision

            if args.is_template_seq:
                if is_first_frame and use_body: 
                    gaussians._xyz.requires_grad = False
                    gaussians._features_dc.requires_grad = False
                    gaussians._features_rest.requires_grad = False
                    gaussians._scaling.requires_grad = False
                    gaussians._rotation.requires_grad = False
                    gaussians._opacity.requires_grad = False
                    gaussians.mesh.v.requires_grad = True
                if is_first_frame:
                    gaussians.update_learning_rate(iter)
                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iter % 1000 == 0:
                    gaussians.oneupSHdegree()
            else:
                if is_first_frame and iter == iterations-collision_iteration+1:
                    face_center = gaussians.mesh.body.vertices[gaussians.mesh.body.faces].mean(-2)
                    _, nn_list = neighbors.KDTree(face_center).query(gaussians.mesh.v.detach().cpu().numpy())
                    gaussians.mesh.collision_faces_ids = nn_list
                    gaussians.mesh.init_body(gaussians.mesh.body)
            iter_start.record()
            gaussians.update_face_coor()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # set background
            bg = torch.rand((3), device="cuda") if opt.random_background else background



            # Render
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            mask = viewpoint_cam.gt_alpha_mask.cuda() if opt.only_foreground_loss else None
    
            loss_dict = {}
            loss_dict['img'] = l1_loss(image, gt_image, mask) * (1.0 - opt.lambda_dssim)
            loss_dict['ssim'] = 1.0 - ssim(image, gt_image, mask) * opt.lambda_dssim
            if is_first_frame and args.is_template_seq:
                loss_dict['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz
                loss_dict['scale']  = F.relu(gaussians.scaling_activation(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                if gaussians.mesh.v.requires_grad:
                    loss_dict.update(gaussians.mesh.get_energy_loss(args, use_body=True))
            else:
                loss_dict.update(gaussians.mesh.get_energy_loss(args, use_body))

            # for k, v in loss_dict.items():
            #     print(f"{k}: {v.item():.5f}")

            # print('visibility_filter', visibility_filter.shape, visibility_filter.sum())

            # assert False
            # ########### DEBUG ############ 
            if iter % 100 == 0:
                gt_image_np = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
                image_np = image.detach().cpu().numpy().transpose(1, 2, 0)
                mask_np = mask.detach().cpu().numpy().transpose(1, 2, 0) 
                mask_np = np.concatenate([mask_np, mask_np, mask_np], axis=-1)

                error_map = np.abs(gt_image_np - image_np)

                temp_folder = Path(DEFAULTS.temp_folder)
                gt_image_path = temp_folder / 'gt' / f'{iter:05d}.png'
                image_path = temp_folder / 'img' / f'{iter:05d}.png'
                mask_path = temp_folder / 'mask' / f'{iter:05d}.png'
                error_map_path = temp_folder / 'error' / f'{iter:05d}.png'

                gt_image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                error_map_path.parent.mkdir(parents=True, exist_ok=True)

                # print('gt_image_np', gt_image_np.shape, gt_image_np.min(), gt_image_np.max())
                # print('image_np', image_np.shape, image_np.min(), image_np.max())
                # print('mask_np', mask_np.shape, mask_np.min(), mask_np.max())
                # print('error_map', error_map.shape, error_map.min(), error_map.max())

                Image.fromarray((gt_image_np * 255).astype(np.uint8)).save(gt_image_path)
                Image.fromarray((image_np * 255).astype(np.uint8)).save(image_path)
                Image.fromarray((mask_np * 255).astype(np.uint8)).save(mask_path)
                Image.fromarray((error_map * 255).astype(np.uint8)).save(error_map_path)

                # assert False
            # ########### DEBUG ############ 

            loss = 0
            for k, v in loss_dict.items():
                loss = loss + v
            loss.backward()
            iter_end.record()

            with torch.no_grad():
                # prune and densify

                if is_first_frame and not use_body and args.is_template_seq:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iter > opt.densify_from_iter and iter % opt.densification_interval == 0:
                        size_threshold = 20 if iter > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iter % opt.opacity_reset_interval == 0 or (dataset.white_background and iter == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iter <= iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad()

                # log and save
                logger(loss_dict, iter, iterations)

                if iter == iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iter))
                    viewer = saver(viewer, gaussians, scene, args, bg)


        progress_bar.close()
        scene.post_each_frame(is_first_frame)

    # All done
    print("\nTraining complete.")
