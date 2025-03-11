import os
import sys
import glob
import torch
import socket
import pickle
import trimesh
import bpy
from PIL import Image, ImageFilter
from typing import NamedTuple
from utils.defaults import DEFAULTS
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.dataset_readers import CameraInfo, Dataloader
from sklearn import neighbors
from arguments import ModelParams
from utils.io_utils import load_masked_image, read_obj, write_obj
from utils.camera_utils import cameraList_from_camInfos

class AvatarDataloader(Dataloader):
    def __init__(self, args) -> None:
        self.device = 'cuda'
        self.subject = args.subject
        self.subject_out = args.subject_out 
        # self.colmap_path = "../datas"
        # self.fg_label = args.garment_type
        # self.panelize_labels = ['background', args.garment_type]
        self.texture_size = args.texture_size
        self.texture_margin = args.texture_margin

        # 4DDress dataset pre-defined labels
        self.SURFACE_LABEL = ['full_body', 'skin', 'upper', 'lower', 'hair', 'glove', 'shoe', 'outer', 'background']
        GRAY_VALUE = np.array([255, 128, 98, 158, 188, 218, 38, 68, 255])
        self.MaskLabel = dict(zip(self.SURFACE_LABEL, GRAY_VALUE))
    
        # locate multiple sequence
        _seq = f"{DEFAULTS.dataset_root}/{args.subject}"
        _regs = glob.glob(os.path.join(args.subject_out, "*/meshes"))
        seq_names = [s.split('/')[-2] for s in _regs]
        if '*' in _seq:
            _seqs = [_seq.replace("*",n.split('take')[-1]) for n in seq_names]
        else:
            _seqs = [_seq]
            seq_names = [_seq.split('/')[-1]]

        self.dataset_infos = {}
        self.frame_collection = []
        for name, seq_path in zip(seq_names, _seqs):
            info = {}
            # locate camera
            info['cam_paths'] = sorted([os.path.join(seq_path, fn) for fn in os.listdir(seq_path) if '00' in fn])
            info['camera_params'] = json.load(open(os.path.join(seq_path, 'cameras.json'), 'r'))
            info['cam_num'] = len(info['cam_paths'])
            # frame info
            _imgs = sorted(glob.glob(os.path.join(info['cam_paths'][0], "capture_images/*.png")))
            info['start_frame'] = int(_imgs[0].split('/')[-1].split(".png")[0])
            info['frame_num'] = len(_imgs)

            # collect info
            self.dataset_infos[name] = info
            self.frame_collection += [(name, i) for i in range(info['frame_num'])]


    def get_seq_info(self, name):
        info = self.dataset_infos[name]
        # locate camera
        self.cam_paths = info['cam_paths']
        self.camera_params = info['camera_params']
        self.cam_num = info['cam_num']
        # frame info
        self.start_frame = info['start_frame']
        self.frame_num = info['frame_num']

    def get_cam_info(self, idx, is_ff=False):
        frame_idx = self.start_frame + idx # frame filename
        camera_infos = []

        # process all cameras
        for idx, _cam in enumerate(self.cam_paths):
            print(f"[4DDress] Reading frame {frame_idx} camera {idx+1}/{self.cam_num} ")

            _img = os.path.join(_cam,"capture_images",f"{frame_idx:05d}.png")
            # if os.path.exists(os.path.join(_cam,"group_labels",f"{frame_idx:05d}.png")):
            #     _lab = os.path.join(_cam,"group_labels",f"{frame_idx:05d}.png")
            # else:
            _lab = os.path.join(_cam,"capture_labels",f"{frame_idx:05d}.png")

            bg = np.array([0, 0, 0])
            image_dict = load_masked_image(_img, _lab, bg)
            masked_img = image_dict['masked_img'] 
            mask = image_dict['mask']


            cam_name = _cam.split('/')[-1]

            # image = Image.open(_img)
            # width, height = image.size
            width, height = masked_img.shape[1], masked_img.shape[0]

            # get camera intrinsic and extrinsic matrices
            intrinsic = np.asarray(self.camera_params[cam_name]["intrinsics"])
            extrinsic = np.asarray(self.camera_params[cam_name]["extrinsics"])

            R, T = np.transpose(extrinsic[:, :3]), extrinsic[:, 3]
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[:2, 2]
            FovY, FovX = focal2fov(fy, height), focal2fov(fx, width)

            # label = np.array(Image.open(_lab))[...,None]
            # mask = label == self.MaskLabel[self.fg_label]
            # if self.fg_label == 'full_body': mask = ~mask

            # masked_img =  np.array(image) * mask + 255 * bg * ~mask
            image = Image.fromarray(np.array(masked_img, dtype=np.byte), "RGB")

            # get panelize mask # TODO: What's that?
            # if len(self.panelize_labels) > 1: 
            #     panelize = np.zeros_like(mask)
            #     for key in self.panelize_labels:
            #         mask = label == self.MaskLabel[key]
            #         if key == 'full_body': mask = ~mask
            #         panelize += mask
            # else:
            #     panelize = mask
                
            panelize = mask

            # append camera_info        
            camera_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,  fx=fx, fy=fy, cx=cx, cy=cy, image=image, mask=panelize,
                                    image_path=_img, image_name=cam_name, width=width, height=height)
            camera_infos.append(camera_info)

        sorted_cam_info = sorted(camera_infos.copy(), key = lambda x : x.image_name)
        return sorted_cam_info

    def bake_texture(self, mesh_path, w=512, h=512, show=False, save=False):
        '''
        Bake normal map and ambient occlusion map for given mesh
        Output:
            ambient: np.array(w, h)
            normal: np.array(w, h, 3)
        '''
        # Remove all elements
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # load mesh
        bpy.ops.wm.obj_import(filepath=mesh_path)
        mesh = bpy.data.objects[-1]

        # init ambient occlusion material && texture
        ao_mat = bpy.data.materials.new(name="AmbientOcclusion")
        ao_mat.use_nodes = True
        ImgTextnode = ao_mat.node_tree.nodes.new('ShaderNodeTexImage')
        ao_img = bpy.data.images.new("AmbientOcclusion", width=w, height=h, alpha=False)
        ImgTextnode.image = ao_img
        # assign material to mesh
        mesh.data.materials.clear()
        mesh.data.materials.append(ao_mat)     
        # Set up context for baking
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.object.mode_set(mode='OBJECT')
        # Set render engine and device
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        # Bake ambient occlusion texture
        bpy.ops.object.bake(type='AO', margin=self.texture_margin)


        # init normal material && texture
        nm_mat = bpy.data.materials.new(name="Normal")
        nm_mat.use_nodes = True
        ImgTextnode = nm_mat.node_tree.nodes.new('ShaderNodeTexImage')
        nm_img = bpy.data.images.new("Normal", width=w, height=h, alpha=False)
        ImgTextnode.image = nm_img
        # assign material to mesh
        mesh.data.materials.clear()
        mesh.data.materials.append(nm_mat)     
        # Set up context for baking
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.mode_set(mode='OBJECT')
        # Set render engine and device
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        # Bake ambient occlusion texture
        bpy.ops.object.bake(type='NORMAL', normal_space='OBJECT', margin=self.texture_margin)

        # output
        ambient = np.array(ao_img.pixels).reshape(w,h,-1)[...,0]
        normal = np.array(nm_img.pixels).reshape(w,h,-1)[...,:3]

        # Convert the NumPy array to PIL image, viualize/save
        if show:
            Image.fromarray(np.uint8(ambient * 255)).show()
            Image.fromarray(np.uint8(normal * 255)).show()
        if save:
            _ambient = mesh_path.replace("meshes/","texture/ambient/")
            _normal = mesh_path.replace("meshes/","texture/normal/")
            os.makedirs(os.path.dirname(_ambient), exist_ok=True)
            os.makedirs(os.path.dirname(_normal), exist_ok=True)

            Image.fromarray(np.uint8(ambient * 255)).save(_ambient.replace(".obj",".png"))
            Image.fromarray(np.uint8(normal * 255)).save(_normal.replace(".obj",".png"))

        return ambient, normal


    def get_maps(self, idx):
        # TODO: Is this used anywhere?
        frame_idx = self.start_frame + idx # frame filename
        # locate template
        _tmp = os.path.join(self.colmap_path, "_".join(self.subject.split('/')[:-1])+"_Take*", f"{self.fg_label}_uv.obj")
        _tmp = glob.glob(_tmp)[0]
        _mesh = os.path.join(self.subject_out, self.current_seq, "meshes", f"frame_{frame_idx}.obj")
        tmp, mesh = read_obj(_tmp), read_obj(_mesh)
        self.vertices = mesh['vertices']
        tmp['vertices'] = mesh['vertices']
        write_obj(tmp, _mesh)
        # locate mesh (.obj)
        _ambient = _mesh.replace("meshes/","texture/ambient/").replace(".obj",".png")
        _normal = _mesh.replace("meshes/","texture/normal/").replace(".obj",".png")
        if os.path.exists(_ambient) and os.path.exists(_normal): 
            ambient = np.array(Image.open(_ambient)) / 255.
            normal = np.array(Image.open(_normal)) / 255.
        else:
            ambient, normal = self.bake_texture(_mesh, w=self.texture_size, h=self.texture_size, save=True)
        ambient = torch.tensor(ambient, dtype=torch.float32).unsqueeze(0).cuda()
        normal = torch.tensor(normal, dtype=torch.float32).permute(2,0,1).cuda()
        return ambient, normal

    def load_frame(self, args, name, idx, eval=False, llffhold=8):
        self.get_seq_info(name)
        self.current_seq = name
        self.current_frame = self.start_frame + idx
        # load cam & GT images
        self.cam_infos = self.get_cam_info(idx)
        # load mesh --> maps
        self.ambient, self.normal = self.get_maps(idx)

        # split train & eval set
        if eval:
            train_cam_infos = [c for idx, c in enumerate(self.cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(self.cam_infos) if idx % llffhold == 0]
        else:
            train_cam_infos = self.cam_infos
            test_cam_infos = []

        args.resolution, args.data_device = -1, "cuda"
        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(train_cam_infos, 1., args)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(test_cam_infos, 1., args)
