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
from pathlib import Path
import random
import json
import socket
import time
import trimesh
import copy
import shutil
import torch
import glob
import numpy as np
import open3d as o3d
from sklearn import neighbors
from typing import Union, List
from utils.defaults import DEFAULTS
from utils.initialisation_utils import COLMAP_recon
from utils.preprocess_utils import PrepareDataset
from utils.system_utils import searchForMaxIteration, searchForMinFrame
from scene.scene import Scene, getNerfppNorm, store_cam
from scene.mesh_gaussian_model import MeshGaussianModel
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import Dataloader
from arguments import ModelParams
# from convert4d import prepare_4ddress, colmap
from scene.colmap_loader import read_points3D_binary
from utils.camera_utils import cameraList_from_camInfo, camera_to_JSON
from utils.graphics_utils import getWorld2View2
from utils.io_utils import fetchPly, read_obj
from utils.general_utils import o3d_knn


class crossScene(Scene):

    gaussians : MeshGaussianModel

    def __init__(self, args : ModelParams, dataloader : Dataloader, gaussians :  MeshGaussianModel):
        """
        :param path: Path to colmap scene main folder.
        :param mode: trainning mode (recon/opt)
        """
        super(crossScene, self).__init__(args, dataloader, gaussians)

    def prepare_frame(self, t, is_ff, shuffle=True, resolution_scales=[1.0]):
        """
        @param t: int, frame index, starts from 0
        @param is_ff: bool, is first frame
        """

        self.current_frame = t

        # if not start optimizing from first frame
        if not is_ff and self.gaussians.prev_xyz is None:
                self.prep_start_from_frame(self.current_frame)

        ply_path = Path(self.subject_out) / DEFAULTS.stage2 / "Template" / "local_point_cloud.ply"


        self.gaussians.load_ply(ply_path)

        # image, mask and camera
        self.dataloader.load_frame(t)

        # split train/test sets
        self.train_cameras = {}
        self.test_cameras = {}

        if self.args.eval:
            train_cam_inf = [c for idx, c in enumerate(self.dataloader.cam_info) if idx % self.args.llffhold != 0]
            test_cam_info = [c for idx, c in enumerate(self.dataloader.cam_info) if idx % self.args.llffhold == 0]
        else:
            train_cam_inf = self.dataloader.cam_info
            test_cam_info = []

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfo(train_cam_inf, resolution_scale, self.args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfo(test_cam_info, resolution_scale, self.args)

        nerf_normalization = getNerfppNorm(train_cam_inf)
        self.cameras_extent = nerf_normalization["radius"]
        if shuffle:
            random.shuffle(train_cam_inf)  # Multi-res consistent random shuffling
            random.shuffle(test_cam_info)  # Multi-res consistent random shuffling

        stage2_path = Path(self.subject_out) / DEFAULTS.stage2 / self.args.sequence



        if is_ff:
            store_cam(self.dataloader.cam_info, stage2_path)
            print(f"Cross from model {self.args.cross_from}")
            # first frame ICP init
            if self.args.use_icp:
                self.gaussians.mesh.v = self.sparse_icp()
                self.gaussians.mesh.v.requires_grad = True

            # body
            body = o3d.io.read_triangle_mesh(self.dataloader.smplx_list[t])
            body.remove_vertices_by_index(self.gaussians.hand_list)
            face_center = np.array(body.vertices)[np.array(body.triangles)].mean(-2)
            _, nn_list = neighbors.KDTree(face_center).query(self.gaussians.mesh.v.detach().cpu().numpy())
            self.gaussians.mesh.collision_faces_ids = nn_list
            self.gaussians.mesh.init_body(body)
        else:
            # body
            body = o3d.io.read_triangle_mesh(self.dataloader.smplx_list[t-1])
            body.remove_vertices_by_index(self.gaussians.hand_list)
            face_center = np.array(body.vertices)[np.array(body.triangles)].mean(-2)
            _, nn_list = neighbors.KDTree(face_center).query(self.gaussians.mesh.v.detach().cpu().numpy())
            self.gaussians.mesh.collision_faces_ids = nn_list
            body = o3d.io.read_triangle_mesh(self.dataloader.smplx_list[t])
            body.remove_vertices_by_index(self.gaussians.hand_list)
            self.gaussians.mesh.init_body(body)

            print(f"Loading Mesh at frame {self.current_frame-1:05d}")
            previous_path = stage2_path / 'meshes' / f'frame_{self.current_frame-2:05d}.obj'
            if not previous_path.exists():
                previous_path = stage2_path / 'meshes' / f'frame_{self.current_frame-1:05d}.obj'
            previous = read_obj(previous_path)
            current = read_obj(stage2_path / 'meshes' / f'frame_{self.current_frame-1:05d}.obj')
            self.gaussians.mesh.momentum_update(current['vertices'], current['faces'])

            self.gaussians.mesh.tar_v = torch.tensor(current['vertices'] + (current['vertices']-previous['vertices'])).cuda()

            print("Loading Gaussian at frame_00000")
            ply_path = stage2_path / "point_cloud" / "frame_00000" / "local_point_cloud.ply"
            self.gaussians.load_ply(ply_path)
            

        self.gaussians.spatial_lr_scale = self.cameras_extent

    def sparse_icp(self,):
        stage1_path = Path(self.subject_out) / DEFAULTS.stage1
        stage2_path = Path(self.subject_out) / DEFAULTS.stage2
        stage2_sequence_path = stage2_path / self.args.sequence
        points3D_path = stage1_path / "sparse" / "points3D.bin"
        source = self.binary_to_o3d(points3D_path)

        source_root = Path(DEFAULTS.data_root) / self.args.subject / self.args.sequence
        target_root = stage2_sequence_path / 'colmap'
        remove_folder = not os.path.exists(target_root)

        PrepareDataset(source_root, target_root, self.args.camera)
        COLMAP_recon(target_root, skip_dense=True)

        points3D_path_out = stage2_sequence_path / "colmap" / "sparse" / "points3D.bin"


        target = self.binary_to_o3d(points3D_path_out)
        # run icp
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, 10.)

        # glob_str = os.path.join(self.args.cross_from, 'meshes/frame_*.obj')
        # _src_mesh = sorted(glob.glob(glob_str), key=lambda x: int(x[:-4].split('_')[-1]))[0]

        _src_mesh = stage2_path / 'Template' / 'template.obj'

        # _src_mesh = 
        source.points =  o3d.utility.Vector3dVector(read_obj(_src_mesh)['vertices'])
        source.transform(reg_p2p.transformation)
        if remove_folder: shutil.rmtree(target_root) # delete point cloud files
        return torch.tensor(np.array(source.points), dtype=torch.float32).cuda()

    def binary_to_o3d(self, path, nb=5):
        xyz, rgb, _ = read_points3D_binary(path)
        # init pc
        pcd = o3d.geometry.PointCloud()
        pcd.points =  o3d.utility.Vector3dVector(xyz)
        # remove outliers & ICP register
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.005)
        # _, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=3.5)
        _, ind = voxel_down_pcd.remove_radius_outlier(nb_points=nb, radius=0.05)
        out = voxel_down_pcd.select_by_index(ind)
        return out


    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])