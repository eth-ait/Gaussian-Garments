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
from pathlib import Path
import pickle
import torch
from torch import nn
import numpy as np
from plyfile import PlyData
import open3d as o3d
from sklearn import neighbors
from argparse import ArgumentParser

from scene.mesh_model import MeshModel
from scene.mesh_gaussian_model import MeshGaussianModel
from utils.defaults import DEFAULTS
from utils.io_utils import read_obj
from utils.geometry_utils import barycentric_2D


class AvatarGaussianModel(MeshGaussianModel):
    def __init__(self, args: ArgumentParser):
        super(MeshGaussianModel, self).__init__(args.sh_degree)

        # locate template
        output_root = Path(DEFAULTS.output_root) / args.subject_out
        _template = output_root / DEFAULTS.stage1 / 'template_uv.obj'
        tem_dict = read_obj(_template)

        stage2_path = output_root / DEFAULTS.stage2
        sequence_dir = sorted([d for d in stage2_path.iterdir() if d.is_dir()])[0]
        ply_glob = sequence_dir / 'point_cloud' / 'frame_*'
        _ply = glob.glob(str(ply_glob))[0]

        # init mesh
        self.mesh = MeshModel(tem_dict['vertices'], tem_dict['faces'])

        # Compute pixel-wise bound face indices
        bind_map, img = self.get_texture_binding(tem_dict['uvs'], tem_dict['texture_faces'], res=args.texture_size)

        # init uv bindings
        self.gaussian_mask = bind_map > -1 # uv map with the binded face ids for each pixel
        self.binding = torch.tensor(bind_map[self.gaussian_mask], dtype=int).cuda()
        # others
        self.num_gs = len(self.binding)
        
        self.gs_u, self.gs_v = np.where(self.gaussian_mask)
        # compute barycentric coor from uv
        uv_triangles = tem_dict['uvs'][tem_dict['texture_faces']][self.binding.cpu()]*args.texture_size
        uv_gs = np.array([self.gs_v, self.gs_u]).T + 0.5
        bc = barycentric_2D(torch.tensor(uv_triangles), torch.tensor(uv_gs))
        self.gs_bc = [torch.tensor(i, dtype=torch.float32).cuda() for i in bc]

        # gaussians
        self.init_gaussians(_ply)
        self.shs = None # store combined shs feature
        self.local_xyz = None

    def init_gaussians(self, _path):
        print("Number of points at initialisation : ", self.num_gs)
        # load template ply in world frame
        world_pc = PlyData.read(os.path.join(_path, "point_cloud.ply"))
        world_xyz = np.stack((np.asarray(world_pc.elements[0]["x"]),
                    np.asarray(world_pc.elements[0]["y"]),
                    np.asarray(world_pc.elements[0]["z"])),  axis=1)

        ####### convert UV to 3D points and get local self._xyz #######
        self.update_face_coor()
        gs_3d = self.get_barycentric_3d()
        local_offset = self.get_local_offset(gs_3d)

        # find nearest neighbor in loaded ply
        _, nb_idx = neighbors.KDTree(world_xyz).query(gs_3d.cpu().numpy())
        nb_idx = nb_idx.reshape(-1)

        # load template ply in local frame
        local_pc = PlyData.read(os.path.join(_path, "local_point_cloud.ply"))
        xyz = np.stack((np.asarray(local_pc.elements[0]["x"]),
                        np.asarray(local_pc.elements[0]["y"]),
                        np.asarray(local_pc.elements[0]["z"])),  axis=1)
        opacities = np.asarray(local_pc.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(local_pc.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(local_pc.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(local_pc.elements[0]["f_dc_2"])

        # why catch exception?
        try:
            extra_f_names = [p.name for p in local_pc.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(local_pc.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        except:
            features_extra = np.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in local_pc.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(local_pc.elements[0][attr_name])

        rot_names = [p.name for p in local_pc.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(local_pc.elements[0][attr_name])

        # Initialize Gaussian parameters
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.num_gs), device="cuda")

        # self._xyz = nn.Parameter(local_offset.requires_grad_(True))
        self._xyz = nn.Parameter(torch.zeros_like(local_offset).requires_grad_(True))
        # self._xyz = None
        ## local = o3d.geometry.PointCloud()
        ## local.points =  o3d.utility.Vector3dVector(self.get_xyz.detach().cpu().numpy())
        ## o3d.visualization.draw_geometries([local])
        self.prev_feature = torch.tensor(np.concatenate([features_dc,features_extra], axis=-1)[nb_idx], dtype=torch.float, device="cuda").transpose(1, 2) # (N, 3, 16)
        self._features_dc = nn.Parameter(torch.tensor(features_dc[nb_idx], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra[nb_idx], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[nb_idx], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots[nb_idx], dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities[nb_idx], dtype=torch.float, device="cuda").requires_grad_(True))

    @property
    def get_xyz(self):
        xyz = torch.bmm(self.face_orien_mat[self.binding], self._xyz[..., None]).squeeze(-1)
        return xyz * self.face_scaling[self.binding] + self.get_barycentric_3d()
    
    @property
    def get_final_xyz(self):
        xyz = torch.bmm(self.face_orien_mat[self.binding], self.local_xyz[..., None]).squeeze(-1)
        return xyz * self.face_scaling[self.binding] + self.get_barycentric_3d()


    def get_barycentric_3d(self,):
        triangles = self.mesh.v[self.mesh.f][self.binding]
        a, b, c = self.gs_bc
        gs_3d = a[:,None] * triangles[:,0] + b[:,None] * triangles[:,1] + c[:,None] * triangles[:,2] 
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points =  o3d.utility.Vector3dVector(gs_3d.detach().cpu().numpy())
        # o3d.visualization.draw_geometries([pcd])
        return gs_3d

    def get_local_offset(self, gs_3d):
        '''
        compute the offset from face center to gaussian position, in local coordinate
        '''
        face_centers = self.mesh.v[self.mesh.f].mean(1)[self.binding]
        global_offset = (gs_3d - face_centers) / self.face_scaling[self.binding]
        local_offset = torch.bmm(self.face_orien_mat[self.binding].permute(0,2,1), global_offset[..., None]).squeeze(-1)
        return local_offset

    def get_texture_binding(self, uvs, faces, res=512):
        '''
        params:
        @uvs: array(V_e, 2), uv coor. of extended vertices (seam vertices are duplicated)
        @faces: array(F, 3), unwarped faces
        @res: int, texture map resolution
        
        Output:
        @bind_map: array(t_size, t_size), bound face id for each pixel on texture map; non-bound pixels is set to -1
        '''
        # init bind map
        bind_map = np.zeros([res,res]) - 1
        img = np.zeros([res,res,3])
        for i, t in enumerate(faces):
            pixels = self.rasterize(uvs[t], res)
            if len(pixels) == 0: continue
            bind_map[pixels[:,1], pixels[:,0]] = i
            img[pixels[:,1], pixels[:,0]] = np.random.rand(3)*255
        return bind_map, img

    def rasterize(self, triangle, scale):
        pixels = []
        def itp(end1, end2, target, id=0):
            return end1 + (end2-end1) * (target - end1[id]) / (end2[id] - end1[id])
        # find bounding box
        triangle = np.array(triangle) * scale
        miny, maxy = np.floor(triangle[:,1].min()), np.floor(triangle[:,1].max())
        # sort face by y
        v1, v2, v3 = triangle[triangle[:, 1].argsort()]
        v4 = itp(v1, v3, v2[1], 1)
        """
        0 --> 1
        |           o v1
        1

            v2 o       o v4



                           o v3
        """
        # rasterization
        for _y in range(int(miny), int(maxy)+1):
            y = _y + 0.5
            if y < v1[1] or y >= v3[1] : continue
            elif y < v2[1]:
                fmin, fmax = sorted([itp(v1, v2, y, 1)[0], itp(v1, v4, y, 1)[0]])
                minx, maxx = np.floor(fmin), np.floor(fmax)
            elif y < v3[1]:
                fmin, fmax = sorted([itp(v3, v2, y, 1)[0], itp(v3, v4, y, 1)[0]])
                minx, maxx = np.floor(fmin), np.floor(fmax)
            for _x in range(int(minx), int(maxx)+1):
                x = _x + 0.5
                if x < fmin or x >= fmax: continue
                pixels.append([_x,_y])
        return np.array(pixels)
    
    def get_visible_mask(self, camera, dot_product_t=-0.7):
        '''
        compute the mask, for filtering gaussians that are visible
        params:
            @camera: torch.tensor(3,)
        output:
            @vis_mask: torch.tensor(num_gs)
        '''
        device = camera.device
        v = self.mesh.v.cpu().numpy()
        f = self.mesh.f.cpu().numpy()
        # init scene
        scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(v, o3d.core.float32)
        mesh.triangle.indices = o3d.core.Tensor(f, o3d.core.int32)
        scene.add_triangles(mesh)

        # compute UV projection in 3D
        gs_3d = self.get_barycentric_3d()
        # gs_3d = self.get_xyz.detach()

        # cast ray from camera to GS
        ray_o = camera[None].repeat(self.num_gs,1)
        _d = gs_3d - ray_o
        norm_d = _d.norm(dim=1, keepdim=True)
        ray_d = _d / norm_d
        rays = o3d.core.Tensor(torch.cat([ray_o, ray_d], axis=-1).detach().cpu().numpy())

        # Compute the ray intersections.
        EPSILON = 1e-3 # max distance between intersection point and gaussians
        ans = scene.cast_rays(rays)

        f_id = torch.tensor(np.array(ans['primitive_ids'].numpy(), dtype=np.int32))
        vis_mask = self.binding.cpu() == f_id

        return vis_mask.to(device)
        
    def requires_grad(self, value):
        self._xyz.requires_grad = value
        self._features_dc.requires_grad = value
        self._features_rest.requires_grad = value
        self._scaling.requires_grad = value
        self._rotation.requires_grad = value
        self._opacity.requires_grad = value



class AvatarSimulationModel(AvatarGaussianModel):
    def __init__(self, tmp_path: str, sh_degree=3, texture_size=512):
        super(MeshGaussianModel, self).__init__(sh_degree)
        self.texture_size = texture_size

        tmp_path = Path(tmp_path)

        # locate template
        _tmp = tmp_path / DEFAULTS.stage1 / 'template_uv.obj'
        tem_dict = read_obj(_tmp)
        # init mesh
        self.mesh = MeshModel(tem_dict['vertices'], tem_dict['faces'])

        # Compute pixel-wise bound face indices
        bind_map, _ = self.get_texture_binding(tem_dict['uvs'], tem_dict['texture_faces'], res=texture_size)

        # init uv bindings
        self.gaussian_mask = bind_map > -1 # uv map with the binded face ids for each pixel
        self.binding = torch.tensor(bind_map[self.gaussian_mask], dtype=int).cuda()
        # others
        self.num_gs = len(self.binding)
        self.gs_u, self.gs_v = np.where(self.gaussian_mask)
        # compute barycentric coor from uv
        uv_triangles = tem_dict['uvs'][tem_dict['texture_faces']][self.binding.cpu()]*texture_size
        uv_gs = np.array([self.gs_v, self.gs_u]).T + 0.5
        bc = barycentric_2D(torch.tensor(uv_triangles), torch.tensor(uv_gs))
        self.gs_bc = [i.to(torch.float32).cuda() for i in bc]

        # gaussians
        self.init_empty_gaussians()

    def init_empty_gaussians(self,):
        # Initialize Gaussian parameters
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.num_gs), device="cuda")

        self._xyz = nn.Parameter(torch.tensor(np.zeros([self.num_gs, 3]), dtype=torch.float, device="cuda"))
        # self._xyz = None
        ## local = o3d.geometry.PointCloud()
        ## local.points =  o3d.utility.Vector3dVector(self.get_xyz.detach().cpu().numpy())
        ## o3d.visualization.draw_geometries([local])
        self.shs = torch.tensor(np.zeros([self.num_gs, 3, (self.max_sh_degree+1)**2]), dtype=torch.float, device="cuda").transpose(1, 2) # (N, 3, 16)
        self._features_dc = nn.Parameter(torch.tensor(np.zeros([self.num_gs, 3, 1]), dtype=torch.float, device="cuda").transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(torch.tensor(np.zeros([self.num_gs, 3, (self.max_sh_degree+1)**2-1]), dtype=torch.float, device="cuda").transpose(1, 2).contiguous())
        self._scaling = nn.Parameter(torch.tensor(np.zeros([self.num_gs, 3]), dtype=torch.float, device="cuda"))
        self._rotation = nn.Parameter(torch.tensor(np.zeros([self.num_gs, 4]), dtype=torch.float, device="cuda"))
        self._opacity = nn.Parameter(torch.tensor(np.zeros([self.num_gs, 1]), dtype=torch.float, device="cuda"))

    def get_gaussian_map(self,):
        map = {}
        # init zero map
        map['features_dc'] = torch.zeros([self.texture_size, self.texture_size, *self._features_dc.shape[1:]])
        map['features_rest'] = torch.zeros([self.texture_size, self.texture_size, *self._features_rest.shape[1:]])
        map['scaling'] = torch.zeros([self.texture_size, self.texture_size, *self._scaling.shape[1:]])
        map['rotation'] = torch.zeros([self.texture_size, self.texture_size, *self._rotation.shape[1:]])
        map['opacity'] = torch.zeros([self.texture_size, self.texture_size, *self._opacity.shape[1:]])
        # fill in gaussian values
        map['features_dc'][self.gaussian_mask] = self._features_dc.detach().cpu()
        map['features_rest'][self.gaussian_mask] = self._features_rest.detach().cpu()
        map['scaling'][self.gaussian_mask] = self._scaling.detach().cpu()
        map['rotation'][self.gaussian_mask] = self._rotation.detach().cpu()
        map['opacity'][self.gaussian_mask] = self._opacity.detach().cpu()
        return map
    
    def load_texture(self, path):
        map = pickle.load(open(path, "rb"))
        mask = map['mask']
        self._features_dc = map['features_dc'][mask].cuda()
        self._features_rest = map['features_rest'][mask].cuda()
        self._scaling = map['scaling'][mask].cuda()
        self._rotation = map['rotation'][mask].cuda()
        self._opacity = map['opacity'][mask].cuda()
        self._xyz = map['xyz'][mask].cuda()

