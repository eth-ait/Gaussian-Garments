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
from register_meta import check_if_template
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
from utils.system_utils import searchForMaxIteration, searchForMinFrame
from scene.scene import Scene, getNerfppNorm, store_cam
from scene.mesh_gaussian_model import MeshGaussianModel
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import Dataloader
from arguments import ModelParams
# from convert4d import prepare_4ddress, colmap
from scene.colmap_loader import read_points3D_binary
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
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

        self.current_frame = self.dataloader.start_frame + t

        # if not start optimizing from first frame
        if not is_ff and self.gaussians.prev_xyz is None:
            self.prep_start_from_frame(self.current_frame)

        # image, mask and camera
        self.dataloader.load_frame(t)

        # split train/test sets
        self.train_cameras = {}
        self.test_cameras = {}

        if self.args.eval:
            train_cam_infos = [c for idx, c in enumerate(self.dataloader.cam_infos) if idx % self.args.llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(self.dataloader.cam_infos) if idx % self.args.llffhold == 0]
        else:
            train_cam_infos = self.dataloader.cam_infos
            test_cam_infos = []

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(train_cam_infos, resolution_scale, self.args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(test_cam_infos, resolution_scale, self.args)

        nerf_normalization = getNerfppNorm(train_cam_infos)
        self.cameras_extent = nerf_normalization["radius"]
        if shuffle:
            random.shuffle(train_cam_infos)  # Multi-res consistent random shuffling
            random.shuffle(test_cam_infos)  # Multi-res consistent random shuffling


        if is_ff:
            store_cam(self.dataloader.cam_infos, self.subject_out)
            print(f"Cross from model {self.args.cross_from}")
            # first frame ICP init
            self.gaussians.mesh.v = self.sparse_icp()

            cross_from_glob = os.path.join(self.args.cross_from, 'point_cloud/frame_*')

            print('cross_from_glob', cross_from_glob)
            _ply_path = sorted(glob.glob(cross_from_glob))[0]
            print("Loading Gaussian at frame {}".format(_ply_path.split('_')[-1]))
            ply_path = os.path.join(_ply_path, "local_point_cloud.ply")
            self.gaussians.load_ply(ply_path)

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

            print("Loading Mesh at frame {}".format(self.current_frame-1))
            try:
                previous = read_obj(os.path.join(self.subject_out, 'meshes', f'frame_{self.current_frame-2}.obj'))
            except:
                previous = read_obj(os.path.join(self.subject_out, 'meshes', f'frame_{self.current_frame-1}.obj'))
            current = read_obj(os.path.join(self.subject_out, 'meshes', f'frame_{self.current_frame-1}.obj'))
            self.gaussians.mesh.momentum_update(current['vertices'], current['faces'])

            self.gaussians.mesh.tar_v = torch.tensor(current['vertices'] + (current['vertices']-previous['vertices'])).cuda()
            self.gaussians.mesh.v = torch.tensor(current['vertices'] + (current['vertices']-previous['vertices'])).cuda()
            # self.gaussians.mesh.v = torch.tensor(current['vertices']).cuda()

            # if use_body:
            #     self.gaussians.lbs_frame(t)


            _ply_idx = self.dataloader.start_frame
            print("Loading Gaussian at frame {}".format(_ply_idx))
            ply_path = os.path.join(self.subject_out,
                                    "point_cloud",
                                    "frame_" + str(_ply_idx),
                                    "local_point_cloud.ply")
            self.gaussians.load_ply(ply_path)           

        self.gaussians.spatial_lr_scale = self.cameras_extent

    def sparse_icp(self,):
        source_path = self.args.subject

        ttype = Path(self.args.subject_out).parts[-3]

        if ttype in DEFAULTS.ttypes:
            ttype_param = f'-t {ttype}'
        else:
            ttype_param = ''
            ttype = ''

        print('ttype', ttype)
        print('ttype_param', ttype_param)
        output_path = "../datas/"+f"/{ttype}/"+"_".join(self.args.subject.split('/'))

        template_path_globstr = "../datas/"+f"/"+"_".join(self.args.subject.split('/')[:-1])+"*"
        template_path_globstr = Path(template_path_globstr)


        print('output_path', output_path)
        print('template_path_globstr', template_path_globstr)

        # get source body clouds
        template_path_candidates = glob.glob(str(template_path_globstr))
        for template_path_candidate in template_path_candidates:
            if 'Old' in template_path_candidate:
                continue
            template_path_candidate = Path(template_path_candidate)
            if check_if_template(template_path_candidate):
                template_path = template_path_candidate
                break



        # print('template_path', template_path)
        # assert False

        points3D_path = os.path.join(template_path,"sparse/points3D.bin")
        print('points3D_path', points3D_path)

        source = self.binary_to_o3d(points3D_path)

        # get target body clouds
        # output_path = "../datas/"+"_".join(self.args.subject.split('/'))
        output_path = "../datas/"+f"/{ttype}/"+"_".join(self.args.subject.split('/'))

        # assert False
        remove_folder = not os.path.exists(output_path)

        # if not os.path.exists(output_path):
        hostname = socket.gethostname()

        if 'ait-server' in hostname or hostname == 'ps075' or hostname.startswith('g'):
            sparse_recon_cmd = f"conda run -n colmap-env python ./server_sparse.py -s {source_path} -g full_body {ttype_param}"
            # sparse_recon_cmd = sparse_recon_cmd.split(' ')
            r = os.system(sparse_recon_cmd)
            
            # subprocess.run(f"conda run -n colmap-env python ./server_sparse.py -s {source_path} -g {self.args.garment_type}", shell=True)
        # else:
        #     prepare_4ddress(source_path, output_path, fg_label=self.args.garment_type, use_mask=True)
        #     colmap(output_path, skip_dense=True)

#             print('colmap')

#         _bin = os.path.join(output_path,"sparse/points3D.bin")
#         xyz, rgb, _ = read_points3D_binary(_bin)
        
        target = self.binary_to_o3d(os.path.join(output_path,"sparse/points3D.bin"))

        # run icp
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, 10.)
        # visualize
        # self.draw_registration_result(source, target, reg_p2p.transformation)
        if self.args.no_icp: 
            aligned_points = np.array(source.points) + (np.array(target.points).mean(0) - np.array(source.points).mean(0))
            source.points = o3d.utility.Vector3dVector(aligned_points)
        else:
            # replace body source mesh with garment
            _src_mesh = sorted(glob.glob(os.path.join(self.args.cross_from, 'meshes/frame_*.obj')), key=lambda x: int(x[:-4].split('_')[-1]))[0]
            source.points =  o3d.utility.Vector3dVector(read_obj(_src_mesh)['vertices'])
            source.transform(reg_p2p.transformation)
        if remove_folder: shutil.rmtree(output_path) # delete point cloud files
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