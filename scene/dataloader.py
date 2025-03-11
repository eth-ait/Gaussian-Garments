import os
import sys
import cv2
import glob
import random
import torch
import socket
import pickle
import trimesh
import copy
import bpy
from PIL import Image, ImageFilter
from typing import NamedTuple
from utils.defaults import DEFAULTS
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.dataset_readers import CameraInfo, Dataloader
from scene.cameras import Camera
from sklearn import neighbors
from arguments import ModelParams
from torch.utils.data import Dataset
from utils.io_utils import load_masked_image, read_obj, write_obj
from utils.camera_utils import cameraList_from_camInfos

class AvatarDataloader(Dataset):
    def __init__(self, args):
        super(AvatarDataloader, self).__init__()
        '''
        Input Params:
        args.hoost_root: path to the server04 Datasets
        args.subject: path to the sequences ended with *, e.g.: 00190/Inner/Take*
        args.subject_out: path to the registered outputs, e.g.: 00190_upper
        # args.garment_type: one of ['outer'/'upper'/'lower']
        '''
        # 4DDress dataset pre-defined labels
        self.SURFACE_LABEL = ['full_body', 'skin', 'upper', 'lower', 'hair', 'glove', 'shoe', 'outer', 'background']
        GRAY_VALUE = np.array([255, 128, 98, 158, 188, 218, 38, 68, 255])
        self.MaskLabel = dict(zip(self.SURFACE_LABEL, GRAY_VALUE))
        self.valid_cameras = []

        # locate path in servers
        # _seq = Path(DEFAULTS.data_root) / args.subject
        output_root = Path(DEFAULTS.output_root) / args.subject_out

        self.device = 'cuda'
        self.data_dir = Path(DEFAULTS.data_root) / args.subject
        self.output_dir = args.subject_out 
        # self.fg_label = args.garment_type
        self.bg = np.array([1,1,1]) if args.white_background else np.array([0, 0, 0])
        self.random_bg = args.random_bg
        self.blur_mask = args.blur_mask
        self.erode_mask = args.erode_mask
        # self.panelize_labels = ['background', args.garment_type]
        self.texture_size = args.texture_size
        self.texture_margin = args.texture_margin


        _template = output_root / DEFAULTS.stage1 / 'template_uv.obj'

        self.template = read_obj(_template)
        sequence_dirs = [seq_dir for seq_dir in self.data_dir.iterdir() if seq_dir.is_dir()]
        

        # collect info across whole dataset
        self.dataset_infos = {}
        self.frame_collection = []
        for seq_path in sequence_dirs:
            seq_name = seq_path.name

            print(f"[Locating] {seq_path}")
            info = {}
            # camera info
            # cam_folders = sorted([seq_path / fn for fn in os.listdir(seq_path) if '00' in fn])
            cam_folders = sorted(list(seq_path.glob('00*')))
            # print(cam_folders)

            if args.eval:
                info['cam_names'] = [n.name for idx, n in enumerate(cam_folders) if idx % args.llffhold != 0]
            else:
                info['cam_names'] = [n.name for n in cam_folders]
            info['json_path'] = os.path.join(seq_path, 'cameras.json')
            # frame info

            images_dir = cam_folders[0] / "capture_images"
            _imgs = sorted(glob.glob(os.path.join(images_dir, "*.png")))
            _imgs = [Path(img) for img in _imgs]

            # print('_imgs[0]', _imgs[0].stem)
            # assert False

            info['start_frame'] = int(_imgs[0].stem)
            info['frame_num'] = len(_imgs)
            # collect info
            self.dataset_infos[seq_name] = info
            self.frame_collection += [(seq_name, f, c) for f in range(info['frame_num']) for c in info['cam_names']]

        if args.shuffle: random.shuffle(self.frame_collection)
        
    def __len__(self):
        return len(self.frame_collection)
    
    def __getitem__(self, index):
        return self.load_frame(*self.frame_collection[index])

    def load_frame(self, name, frame, cam):
        info = self.dataset_infos[name]
        data = {}

        data['current_seq'] = name
        data['current_frame'] = info['start_frame'] + frame
        data['bg'] = np.random.rand(3) if self.random_bg else self.bg

        # load GT image & mask
        _folder = os.path.join(os.path.dirname(info['json_path']),cam)
        _img = os.path.join(_folder, "capture_images", f"{data['current_frame']:05d}.png")
        _lab = os.path.join(_folder, "capture_labels",f"{data['current_frame']:05d}.png")
        image_dict = load_masked_image(_img, _lab, data['bg'])
        masked_img = image_dict['masked_img'] 
        mask = image_dict['mask']
        # image = np.array(Image.open(_img))
        # label = np.array(Image.open(_lab))[...,None]
        # mask = label == self.MaskLabel[self.fg_label]
        # if self.fg_label == 'full_body': mask = ~mask
        # if self.erode_mask:
        #     kernel = np.ones([5,5])
        #     mask = cv2.erode(mask.astype('uint8'), kernel)[...,None].astype(bool)
            # Image.fromarray(np.array(mask[...,0]*255, dtype=np.uint8)).save('mask.png')
            # Image.fromarray(np.array(image * mask + 255 * data['bg'] * ~mask, dtype=np.uint8)).save('gt.png')

        # masked_img =  image * mask + 255 * data['bg'] * ~mask
        # get panelize mask
        # if len(self.panelize_labels) > 1: 
        #     panelize = np.zeros_like(mask)
        #     for key in self.panelize_labels:
        #         mask = label == self.MaskLabel[key]
        #         if key == 'full_body': mask = ~mask
        #         panelize += mask
        #     if self.blur_mask:
        #         # gaussian blur panelization mask
        #         _img = panelize * 255
        #         _img = Image.fromarray(np.concatenate([_img,_img,_img], axis=-1, dtype=np.byte), "RGB")
        #         _img = np.array(_img.filter(ImageFilter.GaussianBlur(radius=15)))[...,:1] / 255
        #         panelize = panelize + _img * ~panelize
        #     if self.erode_mask:
        #         panelize = cv2.erode(panelize.astype('uint8'), kernel)[...,None].astype(bool)
        # else:
        #     panelize = mask
        panelize = mask
        masked_img = torch.tensor(masked_img, dtype=torch.float32) / 255.
        panelize = torch.tensor(panelize)

        # load cam
        cam_params = json.load(open(info['json_path'], 'r'))[cam]
        data['camera'] = self.get_cam_info(cam_params, masked_img, panelize)

        # load current frame mesh and bake textures
        _mesh = self.output_dir / DEFAULTS.stage2 /  data['current_seq'] / "meshes" / f"frame_{data['current_frame']}.obj"
        _body = self.data_dir / data['current_seq'] / "Meshes" / "smplx" / f"{data['current_frame']:05d}.ply"

        # _mesh = os.path.join(self.output_dir, data['current_seq'], "meshes", f"frame_{data['current_frame']}.obj")
        # _body = os.path.join(self.data_dir.replace('*', data['current_seq'].split('take')[-1]), f"Meshes/smplx/{data['current_frame']:05d}.ply")
        data['ambient'], data['normal'], data['mesh_v'] = self.get_maps(_mesh, _body)

        return data


    def get_cam_info(self, params, image, mask):
        # get camera intrinsic and extrinsic matrices
        intrinsic = np.asarray(params["intrinsics"])
        extrinsic = np.asarray(params["extrinsics"])
        h, w, _ = image.shape

        R, T = np.transpose(extrinsic[:, :3]), extrinsic[:, 3]
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[:2, 2]
        FovY, FovX = focal2fov(fy, h), focal2fov(fx, w)

        # return Camera(R=R, T=T, FoVx=FovX, FoVy=FovY, fx=fx, fy=fy, cx=cx, cy=cy,
        #               image=image.permute(2,0,1), gt_alpha_mask=mask.permute(2,0,1), data_device='cpu')
        return  {"R":R, "T":T, "FoVx":FovX, "FoVy":FovY, "fx":fx, "fy":fy, "cx":cx, "cy":cy, 'colmap_id':np.nan, 'image_name':np.nan, 'uid':params['ids'],
                "image":image.permute(2,0,1), "gt_alpha_mask":mask.permute(2,0,1), "data_device":'cpu'} 
    
    def get_maps(self, _mesh: str, _body: str = None):
        # load current mesh
        mesh = read_obj(_mesh)
        # try:
        #     mesh = read_obj(_mesh)
        #     assert len(mesh['vertices']) == len(self.template['vertices']), f"Num of Vertices mismatch, {_mesh}"
        #     assert not False in (mesh['faces'] == self.template['faces']), f"Num of Face mismatch, {_mesh}"
        #     mesh['vertices'][mesh['faces']]
        #     return np.nan, np.nan, torch.tensor(mesh['vertices'], dtype=torch.float32)
        # except:
        #     print(_mesh)
        #     return np.nan, np.nan, torch.tensor(mesh['vertices'], dtype=torch.float32)

        # locate texture path
        _ambient = _mesh.parents[1] / "texture" / "ambient" / f"{_mesh.stem}.png"
        _normal = _mesh.parents[1] / "texture" / "normal" / f"{_mesh.stem}.png"


        # _ambient = _mesh.replace("meshes/","texture/ambient/").replace(".obj",".png")
        # _normal = _mesh.replace("meshes/","texture/normal/").replace(".obj",".png")
        if os.path.exists(_ambient) and os.path.exists(_normal): 
            ambient = np.array(Image.open(_ambient)) / 255.
            normal = np.array(Image.open(_normal)) / 255.
        else:
            ambient, normal = self.bake_texture(_mesh, body_path=_body, w=self.texture_size, h=self.texture_size, save=True)
            # TODO: why try???
            # try:
            #     ambient, normal = self.bake_texture(_mesh, body_path=_body, w=self.texture_size, h=self.texture_size, save=True)
            # except:
            #     # copy uv info from template
            #     output = copy.deepcopy(self.template)
            #     assert len(output['vertices']) == len(mesh['vertices']), f"Num of Vertices mismatch, {_mesh}"
            #     output['vertices'] = mesh['vertices']
            #     write_obj(output, _mesh)
            #     ambient, normal = self.bake_texture(_mesh, body_path=_body, w=self.texture_size, h=self.texture_size, save=True)
        ambient = torch.tensor(ambient, dtype=torch.float32).unsqueeze(0)
        normal = torch.tensor(normal, dtype=torch.float32).permute(2,0,1)
        mesh_v = torch.tensor(mesh['vertices'], dtype=torch.float32)
        return ambient, normal, mesh_v

    def bake_texture(self, mesh_path, body_path=None, w=512, h=512, show=False, save=False):
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
        if os.path.exists(body_path):
            # TODO: are there faulty body meshes?
            # bpy.ops.wm.ply_import(filepath=str(body_path))
            try:
                bpy.ops.wm.ply_import(filepath=str(body_path))
            except:
                print(body_path)
        bpy.ops.wm.obj_import(filepath=str(mesh_path))
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
            _ambient = mesh_path.parents[1] / "texture" / "ambient" / f"{mesh_path.stem}.png"
            _normal = mesh_path.parents[1] / "texture" / "normal" / f"{mesh_path.stem}.png"
            os.makedirs(os.path.dirname(_ambient), exist_ok=True)
            os.makedirs(os.path.dirname(_normal), exist_ok=True)

            Image.fromarray(np.uint8(ambient * 255)).save(_ambient)
            Image.fromarray(np.uint8(normal * 255)).save(_normal)

        return ambient, normal

    def get_pos_map(self, vert, smplx):
        rot_mat = R.from_rotvec(smplx['global_orient']).as_matrix()
        new_vert = torch.mm(torch.tensor(rot_mat.T, dtype=torch.float32), vert.T).T - smplx['transl']
        return new_vert

