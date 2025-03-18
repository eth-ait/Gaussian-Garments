import os
from pathlib import Path
import random
import json
import torch
import numpy as np
import open3d as o3d
from sklearn import neighbors
from scene.mesh_gaussian_model import MeshGaussianModel
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import Dataloader
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfo, camera_to_JSON
from utils.defaults import DEFAULTS
from utils.graphics_utils import getWorld2View2
from utils.io_utils import fetchPly, read_obj
from utils.general_utils import o3d_knn

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def store_cam(camlist, _out):
    json_cams = []
    for id, cam in enumerate(camlist):
        json_cams.append(camera_to_JSON(id, cam))
    with open(os.path.join(_out, "cameras.json"), 'w') as file:
        json.dump(json_cams, file)


class Scene:

    gaussians : MeshGaussianModel

    def __init__(self, args : ModelParams, dataloader : Dataloader, gaussians :  MeshGaussianModel):
        """
        :param path: Path to colmap scene main folder.
        :param mode: trainning mode (recon/opt)
        """
        self.args = args
        self.subject_out = args.subject_out
        self.dataloader = dataloader
        self.gaussians = gaussians

    def prepare_frame(self, t, is_ff, shuffle=True, resolution_scales=[1.0]):
        """
        @param t: int, frame index, starts from 0
        @param is_ff: bool, is first frame
        """
        self.current_frame = t
        stage2_path = Path(self.subject_out) / DEFAULTS.stage2 / self.args.sequence

        # if not start optimizing from first frame
        if not is_ff:
            print("Loading Gaussian at frame frame_00000")
            ply_path = stage2_path / "point_cloud" / "frame_00000" / "local_point_cloud.ply"
            self.gaussians.load_ply(ply_path)
        

            if self.gaussians.prev_xyz is None:
                self.prep_start_from_frame(self.current_frame)
        # image, mask and camera
        self.dataloader.load_frame(t)

        # split train/test sets
        self.train_cameras = {}
        self.test_cameras = {}

        if self.args.eval:
            train_cam_info = [c for idx, c in enumerate(self.dataloader.cam_info) if idx % self.args.llffhold != 0]
            test_cam_info = [c for idx, c in enumerate(self.dataloader.cam_info) if idx % self.args.llffhold == 0]
        else:
            train_cam_info = self.dataloader.cam_info
            test_cam_info = []

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfo(train_cam_info, resolution_scale, self.args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfo(test_cam_info, resolution_scale, self.args)

        nerf_normalization = getNerfppNorm(train_cam_info)
        self.cameras_extent = nerf_normalization["radius"]
        if shuffle:
            random.shuffle(train_cam_info)  # Multi-res consistent random shuffling
            random.shuffle(test_cam_info)  # Multi-res consistent random shuffling


        if is_ff:
            store_cam(self.dataloader.cam_info, stage2_path)
            ply_path = stage2_path  / "input.ply"
            self.gaussians.create_from_pcd(fetchPly(ply_path), self.cameras_extent)

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
            self.gaussians.mesh.v = torch.tensor(current['vertices'] + (current['vertices']-previous['vertices'])).cuda()


        self.gaussians.spatial_lr_scale = self.cameras_extent

    def prep_start_from_frame(self, curr_t):
        '''
        This func is used only when you don't want to train from first frame 
        @param curr_t: the frame name taht you want to start
        '''
        print(f"Preparing to start from frame {curr_t}")

        stage2_path = Path(self.subject_out) / DEFAULTS.stage2 / self.args.sequence

        _mesh = stage2_path / 'meshes' / f'frame_{curr_t-1:05d}.obj'
        self.gaussians.mesh.v = torch.tensor(read_obj(_mesh)['vertices'], device='cuda')
        self.gaussians.update_face_coor()
        self.post_each_frame(True)

    def post_each_frame(self, is_ff, num_knn=20):

        self.gaussians.prev_xyz = self.gaussians.get_xyz.detach()
        self.gaussians.prev_rot = self.gaussians.get_rotation.detach()
        if is_ff:
            # gaussian 2 gaussian ARAP
            neighbor_sq_dist, neighbor_indices = o3d_knn(self.gaussians.prev_xyz.cpu().numpy(), num_knn)
            neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
            neighbor_dist = np.sqrt(neighbor_sq_dist)
            self.gaussians.neighbor_indices = torch.tensor(neighbor_indices).cuda().long().contiguous()
            self.gaussians.neighbor_weight = torch.tensor(neighbor_weight).cuda().float().contiguous()
            self.gaussians.neighbor_dist = torch.tensor(neighbor_dist).cuda().float().contiguous()

            # gaussian 2 vertices ARAP
            neighbor_vt = self.gaussians.mesh.vt[self.gaussians.mesh.f]
            init_gv_offset = neighbor_vt[self.gaussians.binding] - self.gaussians.prev_xyz[:,None]
            gv_neighbor_weight = np.exp(-2000 * init_gv_offset.norm(dim=-1).cpu().numpy())
            self.gaussians.gv_neighbor_weight = torch.tensor(gv_neighbor_weight).cuda().float().contiguous()
            self.gaussians.prev_gv_offset = init_gv_offset
            
        self.gaussians.prev_offset = self.gaussians.prev_xyz[self.gaussians.neighbor_indices] - self.gaussians.prev_xyz[:,None]

    def save(self, frame, template=False):
        frame = int(frame)
        if template:
            point_cloud_path = Path(self.subject_out) / DEFAULTS.stage2 / "Template" 
        else:
            stage2_path = Path(self.subject_out) / DEFAULTS.stage2 / self.args.sequence
            point_cloud_path = stage2_path / "point_cloud"/ f"frame_{frame:05d}"

        self.gaussians.save_ply(point_cloud_path/ "local_point_cloud.ply", save_local=True)
        self.gaussians.save_ply(point_cloud_path / "point_cloud.ply")


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]