import os
import sys
import glob
import random
import torch
import socket
import pickle
import trimesh
import bpy
import smplx
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
from scene.cameras import Camera
from sklearn import neighbors
from arguments import ModelParams
from torch.utils.data import Dataset
from lbs import load_4DDress_smplx, prepare_lbs
from utils.io_utils import read_obj, write_obj
from utils.camera_utils import cameraList_from_camInfos

class BodyDataloader(Dataset):
    def __init__(self, args):
        super(BodyDataloader, self).__init__()

        self.device = 'cuda'
        self.subject = args.subject
        self.subject_out = args.subject_out 
        colmap_path = "../datas"
        self.fg_label = ['skin', 'hair', 'glove', 'shoe']
        self.panelize_labels = ['background'] + self.fg_label

        self.bg = np.array([1,1,1]) if args.white_background else np.array([0, 0, 0])

        # 4DDress dataset pre-defined labels
        self.SURFACE_LABEL = ['full_body', 'skin', 'upper', 'lower', 'hair', 'glove', 'shoe', 'outer', 'background']
        GRAY_VALUE = np.array([255, 128, 98, 158, 188, 218, 38, 68, 255])
        self.MaskLabel = dict(zip(self.SURFACE_LABEL, GRAY_VALUE))
    
        # locate multiple sequence
        host_path = DEFAULTS.dataset_root
        _seq = f'{host_path}{args.subject}'
        seq_names = [s.split('/')[-1] for s in glob.glob(_seq)]
        _seqs = [_seq.replace("*",n.split('Take')[-1]) for n in seq_names]

        # collect info across whole dataset
        self.dataset_infos = {}
        self.frame_collection = []
        for name, seq_path in zip(seq_names, _seqs):
            info = {}
            # camera info
            cam_folders = sorted([os.path.join(seq_path, fn) for fn in os.listdir(seq_path) if '00' in fn])
            info['cam_names'] = [n.split('/')[-1] for n in cam_folders]
            info['json_path'] = os.path.join(seq_path, 'cameras.json')
            # frame info
            _imgs = sorted(glob.glob(os.path.join(cam_folders[0], "capture_images/*.png")))
            info['start_frame'] = int(_imgs[0].split('/')[-1].split(".png")[0])
            info['frame_num'] = len(_imgs)
            # load 4ddress data and body mesh
            info['smplx_poses'], tmp_pose = load_4DDress_smplx(seq_path)
            # collect info
            self.dataset_infos[name] = info
            self.frame_collection += [(name, f, c) for f in range(info['frame_num']) for c in info['cam_names']]

        if args.shuffle: random.shuffle(self.frame_collection)

        ## Init Body info
        # fullbody mesh
        self.gs_body = trimesh.load(os.path.join(args.subject_out, "body.obj"))
        # load smplx
        _npz = os.path.join(host_path, f"smplx/SMPLX_{args.gender.upper()}.npz")
        self.smplx_model = smplx.create(model_path=_npz, model_type='smplx', gender=args.gender.lower(), num_betas=10, use_pca=True, num_pca_comps=12)
        # get canonical template (at origin)
        self.cano_vertices, self.blend_weights, self.nb_idx = prepare_lbs(self.smplx_model, tmp_pose, self.gs_body.vertices-tmp_pose['transl'], unpose=True)

        
    def __len__(self):
        return len(self.frame_collection)
    
    def __getitem__(self, index):
        return self.load_frame(*self.frame_collection[index])

    def load_frame(self, name, frame, cam):
        info = self.dataset_infos[name]
        data = {}

        data['current_seq'] = name
        data['current_frame'] = info['start_frame'] + frame

        # load image & mask
        _folder = os.path.join(os.path.dirname(info['json_path']),cam)
        _img = os.path.join(_folder, "capture_images", f"{data['current_frame']:05d}.png")
        _lab = os.path.join(_folder, "capture_labels",f"{data['current_frame']:05d}.png")
        image = np.array(Image.open(_img))
        label = np.array(Image.open(_lab))[...,None]
        if isinstance(self.fg_label, list):
            mask = np.zeros_like(label)
            for key in self.fg_label:
                assert not key == "full_body", "The fg_label shouldn't be 'fullbody' when training Body guassians"
                mask += label == self.MaskLabel[key]
        else:
            mask = label == self.MaskLabel[self.fg_label]
            if self.fg_label == 'full_body': mask = ~mask

        masked_img =  image * mask + 255 * self.bg * ~mask
        # get panelize mask
        if len(self.panelize_labels) > 1: 
            panelize = np.zeros_like(mask)
            for key in self.panelize_labels:
                mask = label == self.MaskLabel[key]
                if key == 'full_body': mask = ~mask
                panelize += mask
        else:
            panelize = mask
        masked_img = torch.tensor(masked_img, dtype=torch.float32) / 255.
        panelize = torch.tensor(panelize)

        # load cam & GT images
        cam_params = json.load(open(info['json_path'], 'r'))[cam]
        data['camera'] = self.get_cam_info(cam_params, masked_img, panelize)

        # LBS body
        pose_param = info['smplx_poses'][frame]
        posed_vertices, _, _ = prepare_lbs(self.smplx_model, pose_param, self.cano_vertices, blend_weights=self.blend_weights, nn_ids=self.nb_idx)
        posed_vertices += pose_param['transl']
        data['body'] = posed_vertices
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

        return Camera(R=R, T=T, FoVx=FovX, FoVy=FovY, fx=fx, fy=fy, cx=cx, cy=cy, colmap_id=None, image_name=None, uid=None,
                      image=image.permute(2,0,1), gt_alpha_mask=mask.permute(2,0,1), data_device='cpu')
        # return  {"R":R, "T":T, "FoVx":FovX, "FoVy":FovY, "fx":fx, "fy":fy, "cx":cx, "cy":cy,
        #         "image":image.permute(2,0,1), "gt_alpha_mask":mask.permute(2,0,1), "data_device":'cpu'} 
