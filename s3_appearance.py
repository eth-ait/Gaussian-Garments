import os
from pathlib import Path
import torch
import socket
import numpy as np
from PIL import Image
import random
from random import randint
from utils.defaults import DEFAULTS
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene.cameras import Camera
from scene import AvatarNet, AvatarGaussianModel
from scene.dataloader import AvatarDataloader
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.sh_utils import eval_sh
from utils.sh_utils import SH2RGB
from argparse import ArgumentParser
from arguments import PipelineParams, OptimizationParams
import torch.nn.functional as F
from aitviewer.headless import HeadlessRenderer
from aitviewer.scene.camera import PinholeCamera
from aitviewer.utils import path
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.lines import Lines
from aitviewer.renderables.spheres import Spheres

def rm_dimension(data: dict):
    for k, v in data.items():
        if type(v) == dict:
            rm_dimension(data[k])
        else:
            data[k] = v[0]

def logger(loss, bar, iteration):
    # TODO: what's acc and wry catch exception?
    global acc
    try:
        acc = {f'AVG_{k}': (acc[f'AVG_{k}']*(iteration-1)+v) / iteration for k, v in loss.items()}
    except:
        acc = {f'AVG_{k}': v for k, v in loss.items()}

    total_loss_for_log = np.array([v.item() for v in loss.values()]).sum()
    bar.set_postfix({"Loss": f"{total_loss_for_log:.{7}f}"})

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('-s', '--subject', type=str, required=True, default='')
    parser.add_argument('-so', '--subject_out', type=str, default='')
    
    # parser.add_argument('-g', '--garment_type', type=str, required=True) # garment label
    parser.add_argument('--ckpt_path', type=str, default='') 
    # mesh gaussian config
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--texture_size', type=int, default=512)
    parser.add_argument('--texture_margin', type=int, default=5)
    parser.add_argument('--white_background', action='store_true', default=False)
    # training config
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--save_iterations', type=int, default=3000)
    parser.add_argument('--pre_train_iterations', type=int, default=3000)
    parser.add_argument('--lr_init', type=float, default=5e-4)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--random_bg", action="store_true")
    parser.add_argument("--blur_mask", action="store_true")
    parser.add_argument("--erode_mask", action="store_true") # TODO: always false?
    parser.add_argument('--no_xyz', action='store_true') # TODO: is it for ablation study?
    parser.add_argument('--llffhold', type=int, default=12)
    args = parser.parse_args(sys.argv[1:])

    if len(args.subject_out) == 0:
        args.subject_out = args.subject
    args.subject_out = Path(DEFAULTS.output_root) / args.subject_out

    # store_path = DEFAULTS.registration_root
    # args.model_path = os.path.join(DEFAULTS.output_root, args.model_path)
    # args.ckpt_path = os.path.join(DEFAULTS.output_root, args.ckpt_path)

    stage3_path = DEFAULTS.output_root / args.subject_out / DEFAULTS.stage3


    args.debug = False
    args.epochs = 5
    args.eval = True
    args.shuffle = True
    args.random_bg = True
    args.blur_mask = True

    # # Initialize system state (RNG)
    torch.manual_seed(31359)
    np.random.seed(31359)
    ############ DEBUG ############
    w, h = 940, 1280
    global viewer, test_id
    viewer = HeadlessRenderer(size=(2*w, 2*h))
    test_id = 0
    ############ DEBUG ############

    # build components
    dataloader = torch.utils.data.DataLoader(AvatarDataloader(args), batch_size=1, 
                                             shuffle=args.shuffle, num_workers=8)
    gaussians = AvatarGaussianModel(args)
    avatar_net = AvatarNet(args, gaussians).to('cuda')
    start_epoch = avatar_net.training_setup(args)

    for epoch in range(start_epoch, args.epochs+1):
        progress_bar = tqdm(dataloader, desc="Epoch{}".format(epoch))
        for iter, frame_data in enumerate(progress_bar):
            rm_dimension(frame_data)

            if epoch==1 and iter < args.pre_train_iterations:
                gaussians.requires_grad(False)
            else:
                gaussians.requires_grad(True)

            avatar_net.gaussians.mesh.v = frame_data['mesh_v'].cuda()
            avatar_net.gaussians.update_face_coor()
            bg = frame_data['bg'].to(torch.float32).cuda()
            viewpoint_cam = Camera(**frame_data['camera']).to('cuda')

            # predict appearance
            shadow_shs, vis_mask = avatar_net(frame_data['ambient'].cuda(), frame_data['normal'].cuda(), viewpoint_cam)

            # Render
            render_pkg = render(viewpoint_cam, gaussians, args, bg, vis_mask=vis_mask)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            mask = viewpoint_cam.gt_alpha_mask.cuda() if args.only_foreground_loss else None
    
            loss_dict = {}
            loss_dict['img'] = l1_loss(image, gt_image, mask) * (1.0 - args.lambda_dssim)
            loss_dict['ssim'] = 1.0 - ssim(image, gt_image, mask) * args.lambda_dssim
            loss_dict['xyz'] = F.relu(gaussians.local_xyz.norm(dim=1) - args.threshold_xyz).mean() * args.lambda_xyz
            loss_dict['scale']  = F.relu(gaussians.scaling_activation(gaussians._scaling) - args.threshold_scale).norm(dim=1).mean() * args.lambda_scale
            _opacity = gaussians.get_opacity
            loss_dict['opacity'] = F.relu(args.threshold_opacity - _opacity).mean() * args.lambda_opacity
            loss = 0
            for k, v in loss_dict.items():
                loss = loss + v
            loss.backward()


            with torch.no_grad():
                # Optimizer step
                avatar_net.optimizer.step()
                avatar_net.optimizer.zero_grad()
                # log and save
                logger(loss_dict, progress_bar, iter)
            
            if iter > 0 and iter % args.save_iterations == 0:
                print("\n[Epoch {} Frame {}] Saving Checkpoint".format(epoch, iter))
                avatar_net.save_ckpt(stage3_path, epoch)

            ############ SAVE RENDERS ############
            with torch.no_grad():
                if iter % int(args.save_iterations / 5) == 0:
                    # pred gaussian map
                    color_map = torch.zeros([args.texture_size, args.texture_size, 3])
                    shadow_map = torch.zeros([args.texture_size, args.texture_size, 3])
                    dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.num_gs, 1))
                    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

                    ## color --> rgb
                    # color_shs_view = color_shs.view(-1, 3, (gaussians.max_sh_degree+1)**2)
                    # pred_color = torch.clamp_min(eval_sh(gaussians.active_sh_degree, color_shs_view, dir_pp_normalized) + 0.5, 0.0)
                    # color_map[gaussians.gs_u, gaussians.gs_v] = torch.nn.functional.normalize(pred_color, p=2.0, dim = 1).detach().cpu()
                    ## gaussians.shs --> rgb
                    color_shs_view = avatar_net.gaussians.get_features.view(-1, 3, (gaussians.max_sh_degree+1)**2)
                    pred_color = torch.clamp_min(eval_sh(gaussians.active_sh_degree, color_shs_view, dir_pp_normalized) + 0.5, 0.0)
                    color_map[gaussians.gs_u, gaussians.gs_v] = torch.nn.functional.normalize(pred_color, p=2.0, dim = 1).detach().cpu()
                    ## shadow --> rgb
                    shadow_shs_view = shadow_shs.view(-1, 3, (gaussians.max_sh_degree+1)**2)
                    pred_shadow = torch.clamp_min(eval_sh(gaussians.active_sh_degree, shadow_shs_view, dir_pp_normalized) + 0.5, 0.0)
                    shadow_map[gaussians.gs_u, gaussians.gs_v] = torch.nn.functional.normalize(pred_shadow, p=2.0, dim = 1).detach().cpu()
                    # store textures
                    col1 = torch.cat([shadow_map, color_map], axis=0).permute(2,0,1)
                    _a = frame_data['ambient'].cpu()
                    _a = torch.cat([_a, _a, _a], axis=0)
                    col2 = torch.cat([_a, frame_data['normal'].cpu()], axis=1)
                    _t = torch.cat([col1, col2], axis=-1)
                    # _t = torch.tensor(np.array(Image.fromarray(np.array(_t.permute(1,2,0), dtype=np.uint8)).resize([512, 512]))).permute(2,0,1)
                    texture = torch.zeros([3, viewpoint_cam.original_image.shape[1], _t.shape[-1]])
                    texture[:,78:78+_t.shape[1]] = _t

                    # store rendering
                    _test_cam = viewpoint_cam
                    test_gt_img = _test_cam.original_image.detach().cpu()
                    test_img = render(_test_cam, gaussians, args, bg)["render"].detach().cpu()
                    diff = torch.abs(test_img - test_gt_img).cpu()
                    panelize = _test_cam.gt_alpha_mask
                    panelize = torch.cat([panelize,panelize,panelize]).detach().cpu()
                    container = torch.cat([panelize, texture, test_gt_img, test_img, diff], axis=-1)
                    container = container.permute(1,2,0)*255

                    _render = stage3_path / "debug_renders" 
                    container = Image.fromarray(np.array(container, dtype=np.byte), "RGB")
                    os.makedirs(_render, exist_ok=True)
                    render_path = _render / f"ep{epoch:03d}_iter{iter:06d}_{frame_data['current_seq']}_frame{frame_data['current_frame']:04d}.png"
                    container.save(render_path)
                    test_id += 1
            ############ SAVE RENDERS ############

        progress_bar.close()
        avatar_net.save_ckpt(stage3_path, epoch, name=f"epoch{epoch}")


    # All done
    print("\nTraining complete.")

        

