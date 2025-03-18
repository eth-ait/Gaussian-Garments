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

import os
import glob
import copy
import json
from pathlib import Path
import trimesh
import smplx
import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from arguments import ModelParams
import pickle
import open3d as o3d
from sklearn import neighbors
from lbs import prepare_lbs
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw, quat_product, unitquat_to_rotmat
from argparse import ArgumentParser
from scene.mesh_model import MeshModel
from scene.gaussian_model import GaussianModel
from utils.defaults import DEFAULTS
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p
from utils.graphics_utils import BasicPointCloud
from utils.graphics_utils import compute_face_orientation
from utils.io_utils import read_obj, fetchPly, storePly, write_obj
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation

def check_if_template(tdir):
    obj_files = list(tdir.glob('*_uv.obj'))
    ply_files = list(tdir.glob('*.ply'))

    return len(obj_files) > 0 and len(ply_files) > 0

class MeshGaussianModel(GaussianModel):
    def __init__(self, args : ModelParams):
        super().__init__(args.sh_degree)


        source_path = Path(DEFAULTS.output_root) / args.subject_out 
        _template = source_path / DEFAULTS.stage1 / 'template_uv.obj'
        tem_dict = read_obj(_template)

        pc_path = source_path / DEFAULTS.stage1 / 'point_cloud.ply'
        dense_pc = o3d.io.read_point_cloud(pc_path)

        # Convert mesh to point cloud
        xyz = tem_dict['vertices'][tem_dict['faces']].mean(1)
        pc_xyz = np.array(dense_pc.points)
        pc_rgb = np.array(dense_pc.colors)*255

        _, nb_idx = neighbors.KDTree(pc_xyz).query(xyz)
        rgb = pc_rgb[nb_idx.reshape(-1)]

        stage2_path = Path(args.subject_out) / DEFAULTS.stage2 / args.sequence

        # store PC as mesh.ply
        os.makedirs(args.subject_out, exist_ok=True)
        ply_path = stage2_path / "input.ply"
        storePly(ply_path, xyz, rgb)

        # init mesh
        self.mesh = MeshModel(tem_dict['vertices'], tem_dict['faces'])
        _hand = Path(DEFAULTS.aux_root) / "smplx" / "smplx_vert_segmentation.json"
        hand_verts = json.load(open(_hand, 'r'))
        self.hand_list = np.array([v for k, verts in hand_verts.items() for v in verts if 'hand' in k.lower()])

       
        # init params
        self.binding = torch.arange(len(self.mesh.f)).cuda()
        self.binding_counter = torch.ones(len(self.mesh.f), dtype=torch.int32).cuda()
        self.prev_xyz, self.prev_rot = None, None
        self.prev_offset, self.prev_col = None, None
        self.neighbor_indices, self.neighbor_weight, self.neighbor_dist = None, None, None
        self.prev_gv_offset, self.gv_neighbor_weight = None, None
        self.template = tem_dict

    def update_face_coor(self):
        # position
        self.face_center = self.mesh.v[self.mesh.f].mean(1)
        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(self.mesh.v, self.mesh.f, return_scale=True)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))


    def remember_scaling(self):
        scaling = self.scaling_activation(self._scaling)
        scaling = scaling * self.face_scaling[self.binding]
        self.remembered_scaling = scaling

        self.face_scaling_remembered = self.face_scaling

    @property
    def get_scaling(self):
        if hasattr(self, "face_scaling_remembered") and self.face_scaling_remembered is not None:
            scaling = self.scaling_activation(self._scaling)
            scaling = scaling * self.face_scaling_remembered[self.binding]
            return scaling
        
        scaling = self.scaling_activation(self._scaling)
        scaling = scaling * self.face_scaling[self.binding]

        return scaling
    
    @property
    def get_rotation(self):
        rot = self.rotation_activation(self._rotation)
        face_orien_quat = self.rotation_activation(self.face_orien_quat[self.binding])
        world_quat = quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(rot)))
        return self.rotation_activation(world_quat)
    
    @property
    def get_xyz(self):
        # print('self.face_orien_mat', self.face_orien_mat.device)
        # print('self._xyz', self._xyz.device)
        # print('self.binding', self.binding.device)
        xyz = torch.bmm(self.face_orien_mat[self.binding], self._xyz[..., None]).squeeze(-1)
        return xyz * self.face_scaling[self.binding] + self.face_center[self.binding]

    def prune_points(self, mask):
        if hasattr(self, "binding"):
            # make sure each face is bound to at least one point after pruning
            binding_to_prune = self.binding[mask]
            counter_prune = torch.zeros_like(self.binding_counter)
            counter_prune.scatter_add_(0, binding_to_prune, torch.ones_like(binding_to_prune, dtype=torch.int32, device="cuda"))
            mask_redundant = (self.binding_counter - counter_prune) > 0
            mask[mask.clone()] = mask_redundant[binding_to_prune]

        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        if hasattr(self, "binding"):
            self.binding_counter.scatter_add_(0, self.binding[mask], -torch.ones_like(self.binding[mask], dtype=torch.int32, device="cuda"))
            self.binding = self.binding[valid_points_mask]

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        selected_scaling = self.get_scaling[selected_pts_mask]
        face_scaling = self.face_scaling[self.binding[selected_pts_mask]]
        new_scaling = self.scaling_inverse_activation((selected_scaling / face_scaling).repeat(N,1) / (0.8*N))

        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        if hasattr(self, "binding"):
            new_binding = self.binding[selected_pts_mask].repeat(N)
            self.binding = torch.cat((self.binding, new_binding))
            self.binding_counter.scatter_add_(0, new_binding, torch.ones_like(new_binding, dtype=torch.int32, device="cuda"))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        if hasattr(self, "binding"):
            new_binding = self.binding[selected_pts_mask]
            self.binding = torch.cat((self.binding, new_binding))
            self.binding_counter.scatter_add_(0, new_binding, torch.ones_like(new_binding, dtype=torch.int32, device="cuda"))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def local_to_world(self):
        assert hasattr(self, "binding"), "Have no binding Information"
        assert self.face_center is not None, "Please update mesh properties"
        assert self.face_orien_mat is not None, "Please update mesh properties"
        assert self.face_scaling is not None, "Please update mesh properties"
        assert self.face_orien_quat is not None, "Please update mesh properties"

        self._xyz = self.get_xyz
        self._scaling = self.scaling_inverse_activation(self.get_scaling)
        self._rotation = self.get_rotation


    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        # init gaussians in the local triangle frame
        fused_point_cloud = torch.zeros(np.asarray(pcd.points).shape).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.mesh.v = nn.Parameter(self.mesh.v.clone().detach()).requires_grad_(False)



    def save_ply(self, path, save_local = False):
        mkdir_p(os.path.dirname(path))
        mask = self.find_valid_gaussians()

        if save_local:
            xyz = self._xyz.detach().cpu().numpy()[mask]
            scale = self._scaling.detach().cpu().numpy()[mask]
            rotation = self._rotation.detach().cpu().numpy()[mask]
        else:
            xyz = self.get_xyz.detach().cpu().numpy()[mask]
            scale = self.scaling_inverse_activation(self.get_scaling).detach().cpu().numpy()[mask]
            rotation = self.get_rotation.detach().cpu().numpy()[mask]
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[mask]
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[mask]
        opacities = self._opacity.detach().cpu().numpy()[mask]
        # binding = self.binding[...,None].detach().cpu().numpy()[mask]

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, binding), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        if save_local:
            # save bindings
            binding_path = os.path.join(os.path.dirname(path), "binding.pkl")
            binding = self.binding[mask]
            with open(binding_path, 'wb') as f:
                pickle.dump(binding, f)

    def find_valid_gaussians(self,):
        if self.mesh.valid_faces:
            # filter gaussian
            return [i for i, bind_face_id in enumerate(self.binding) if bind_face_id in self.mesh.valid_faces]
        else:
            return range(len(self.binding))

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # fixed when optimizing
        self._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False)
        self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False)
        self._features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False)
        self._features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False)
        self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False)
        self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False)

        self.active_sh_degree = self.max_sh_degree

        binding_path = os.path.join(os.path.dirname(path), "binding.pkl")
        with open(binding_path, 'rb') as f:
            self.binding = pickle.load(f)

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.mesh.v = nn.Parameter(self.mesh.v.clone().detach().requires_grad_(True))


        # print('gaussians._features_dc.requires_grad', self._features_dc.requires_grad)
        # print('gaussians.mesh.v.requires_grad', self.mesh.v.requires_grad)
        # assert False
        

    def training_setup(self, training_args, is_ff):

        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        if is_ff:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': [self.mesh.v], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "vertex"},
            ]
        else:
            print("Optimizing mesh")
            l = [
                {'params': [self.mesh.v], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "vertex"},
                # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]

        

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def GG_ARAP_loss(self,):
        prev_rotmat = nn.functional.normalize(unitquat_to_rotmat(quat_wxyz_to_xyzw(self.prev_rot)))
        curr_rotmat = nn.functional.normalize(unitquat_to_rotmat(quat_wxyz_to_xyzw(self.get_rotation)))
        curr_offset = self.get_xyz[self.neighbor_indices] - self.get_xyz[:,None]

        rel_rot = prev_rotmat @ curr_rotmat.transpose(2,1)
        curr_offset_in_prev_coor = (rel_rot[:, None]  @ curr_offset[..., None]).squeeze(-1)
        loss = torch.sqrt(((curr_offset_in_prev_coor - self.prev_offset) ** 2).sum(-1) * self.neighbor_weight + 1e-20).mean()
        return loss
    
    def GV_ARAP_loss(self,):
        prev_rotmat = nn.functional.normalize(unitquat_to_rotmat(quat_wxyz_to_xyzw(self.prev_rot)))
        curr_rotmat = nn.functional.normalize(unitquat_to_rotmat(quat_wxyz_to_xyzw(self.get_rotation)))
        curr_gv_offset = self.mesh.v[self.mesh.f[self.binding]] - self.get_xyz[:,None]

        rel_rot = prev_rotmat @ curr_rotmat.transpose(2,1)
        curr_gv_offset_in_prev_coor = (rel_rot[:, None]  @ curr_gv_offset[..., None]).squeeze(-1)
        loss = torch.sqrt(((curr_gv_offset_in_prev_coor - self.prev_gv_offset) ** 2).sum(-1) * self.gv_neighbor_weight + 1e-20).mean()
        return loss

    def dynamic3DLoss(self, _lambda):
        loss_dict = {}
        prev_rotmat = nn.functional.normalize(unitquat_to_rotmat(quat_wxyz_to_xyzw(self.prev_rot)))
        curr_rotmat = nn.functional.normalize(unitquat_to_rotmat(quat_wxyz_to_xyzw(self.get_rotation)))
        curr_offset = self.get_xyz[self.neighbor_indices] - self.get_xyz[:,None]

        rel_rot = prev_rotmat @ curr_rotmat.transpose(2,1)
        curr_offset_in_prev_coor = (rel_rot[:, None]  @ curr_offset[..., None]).squeeze(-1)
        loss_dict['rigid'] = torch.sqrt(((curr_offset_in_prev_coor - self.prev_offset) ** 2).sum(-1) * self.neighbor_weight + 1e-20).mean() * _lambda

        rel_quat = rotmat_to_unitquat(rel_rot)
        loss_dict['rot'] = torch.sqrt(((rel_quat[self.neighbor_indices] - rel_quat[:, None]) ** 2).sum(-1) * self.neighbor_weight + 1e-20).mean() * _lambda

        # curr_neighbor_dist = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        # loss_dict['dist'] = torch.sqrt(((curr_neighbor_dist - self.neighbor_dist) ** 2) * self.neighbor_weight + 1e-20).mean() * _lambda
        
        return loss_dict

    def lbs_frame(self, idx):
        assert idx > 0, "lbs start after first frame"
        prev_pose = self.pose_list[idx-1]
        # update LBS cano state
        prev_vertices = self.mesh.v.detach().cpu().numpy()-prev_pose['transl']
        cano_vertices, blend_weights, nb_idx = prepare_lbs(self.smplx_model, prev_pose, prev_vertices, unpose=True)
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points =  o3d.utility.Vector3dVector(self.mesh.v.detach().cpu().numpy())
        # pcd.points =  o3d.utility.Vector3dVector(self.get_xyz.detach().cpu().numpy())
        # pcd.points =  o3d.utility.Vector3dVector(cano_vertices)
        # pcd.points =  o3d.utility.Vector3dVector(posed_vertices)
        # o3d.visualization.draw_geometries([pcd])
        pose_param = self.pose_list[idx]
        posed_vertices, _, _ = prepare_lbs(self.smplx_model, pose_param, cano_vertices, blend_weights=blend_weights, nn_ids=nb_idx)
        posed_vertices += pose_param['transl']
        self.mesh.v = torch.tensor(posed_vertices, dtype=torch.float32).cuda()
        self.update_face_coor()

    def save_mesh(self, path):
        output = copy.deepcopy(self.template)
        output['vertices'] = self.mesh.v.detach().cpu().numpy()
        write_obj(output, path)