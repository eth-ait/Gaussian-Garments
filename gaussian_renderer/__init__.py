#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from typing import Union
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_depth_alpha import GaussianRasterizationSettings, GaussianRasterizer
from scene.mesh_gaussian_model import MeshGaussianModel
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : Union[GaussianModel, MeshGaussianModel], pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, mode=None, vis_mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc._xyz, dtype=pc._xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # debug
    means3D = pc.get_xyz if not hasattr(pc, 'local_xyz') or pc.local_xyz is None else pc.get_final_xyz
    try:
        means3D.retain_grad()
    except:
        pass
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.shs if hasattr(pc, 'shs') else pc.get_features
    else:
        colors_precomp = override_color

    # print('scales', scales.shape)

    # only render visible gaussians
    if vis_mask is not None:
        means3D = means3D[vis_mask] if means3D is not None else means3D
        means2D = means2D[vis_mask] if means2D is not None else means2D
        shs = shs[vis_mask] if shs is not None else shs
        colors_precomp = colors_precomp[vis_mask] if colors_precomp is not None else colors_precomp
        opacity = opacity[vis_mask] if opacity is not None else opacity
        scales = scales[vis_mask] if scales is not None else scales
        rotations = rotations[vis_mask] if rotations is not None else rotations
        cov3D_precomp = cov3D_precomp[vis_mask] if cov3D_precomp is not None else cov3D_precomp

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            '3dposition': means3D,
            "depth": depth,
            "alpha": alpha,
            }

def doll_render(viewpoint_camera, pc : Union[GaussianModel, MeshGaussianModel], pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, override_shs=None, vis_mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.xyz, dtype=pc.xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # debug
    means3D = pc.xyz
    try:
        means3D.retain_grad()
    except:
        pass
    means2D = screenspace_points
    opacity = pc.opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.covariance(scaling_modifier)
    else:
        scales = pc.scaling
        rotations = pc.rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.xyz - viewpoint_camera.camera_center.repeat(pc.features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        elif override_shs is None:
            shs = pc.features
        else:
            shs = override_shs
    else:
        colors_precomp = override_color

    # only render visible gaussians
    if vis_mask is not None:
        means3D = means3D[vis_mask] if means3D is not None else means3D
        means2D = means2D[vis_mask] if means2D is not None else means2D
        shs = shs[vis_mask] if shs is not None else shs
        colors_precomp = colors_precomp[vis_mask] if colors_precomp is not None else colors_precomp
        opacity = opacity[vis_mask] if opacity is not None else opacity
        scales = scales[vis_mask] if scales is not None else scales
        rotations = rotations[vis_mask] if rotations is not None else rotations
        cov3D_precomp = cov3D_precomp[vis_mask] if cov3D_precomp is not None else cov3D_precomp

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return rendered_image, depth, alpha
