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
import sys
import glob
import socket
import pickle
import trimesh
from PIL import Image, ImageFilter
from typing import NamedTuple
from utils.defaults import DEFAULTS
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.io_utils import load_masked_image
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from sklearn import neighbors
from arguments import ModelParams
from tqdm import tqdm

class CameraInfo():
    def __init__(self, uid, R, T, FovY, FovX, fx, fy, cx, cy,
                 image=None, image_path=None, image_name=None,
                 width=None, height=None, mask=None):
        self.uid = uid
        self.R = R
        self.T = T
        self.FovY = FovY
        self.FovX = FovX
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.image = image
        self.image_path = image_path
        self.image_name = image_name
        self.width = width
        self.height = height
        self.mask = mask

class Dataloader():
    def __init__(self, args: ModelParams) -> None:
        self.subject_out = args.subject_out
        self.white_background = args.white_background

        # locate sequence
        seq_path = Path(DEFAULTS.data_root) / args.subject / args.sequence

        # locate camera
        self.cam_paths = sorted([path for path in seq_path.iterdir() if path.is_dir() and path.name != 'smplx'])
        self.camera_params = json.load(open(os.path.join(seq_path, 'cameras.json'), 'r'))
        self.cam_num = len(self.cam_paths)
        # frame info

        self._img_names = {}
        self._gm_names = {}
        self._fg_names = {}

        print('Reading frame info...')
        # Need to collect filenames for each camera separately in case they are not the same as in ActorsHQ
        cam_to_copy_from = None
        for i, cam_path in tqdm(enumerate(self.cam_paths)):
            cam_name = cam_path.name

            if cam_to_copy_from is not None:
                self._img_names[cam_name] = self._img_names[cam_to_copy_from]
                self._gm_names[cam_name] = self._gm_names[cam_to_copy_from]
                self._fg_names[cam_name] = self._fg_names[cam_to_copy_from]            

            img_files = sorted((cam_path/DEFAULTS.rgb_images).glob("*.png"))
            if len(img_files) == 0:
                img_files = sorted((cam_path/DEFAULTS.rgb_images).glob("*.jpg"))

            gm_files = sorted((cam_path/DEFAULTS.garment_masks).glob("*.png"))
            if len(gm_files) == 0:
                gm_files = sorted((cam_path/DEFAULTS.garment_masks).glob("*.jpg"))

            fg_files = sorted((cam_path/DEFAULTS.foreground_masks).glob("*.png"))

            self._img_names[cam_name] = [img.name for img in img_files]
            self._gm_names[cam_name] = [gm.name for gm in gm_files]
            self._fg_names[cam_name] = [fg.name for fg in fg_files]

            if i == 1:
                first_cam = self.cam_paths[0].name
                if self._img_names[cam_name][0] == self._img_names[first_cam][0]:
                    cam_to_copy_from = first_cam

        print('Reading frame info done')

        # self._img_names = [img.name for img in img_files]
        # self._gm_names = [gm.name for gm in gm_files]
        # self._fg_names = [fg.name for fg in fg_files]
        self._len = len(self._img_names[self.cam_paths[0].name])
        # smplx info
        self.smplx_list = sorted(glob.glob(os.path.join(seq_path, "smplx/*.ply")))

    def __len__(self,):
        return self._len
    

    def load_frame(self, frame):
        camera_info_list = []

        # process all cameras
        for c_idx, _cam in enumerate(self.cam_paths):
            print(f"Reading frame #{c_idx} camera {c_idx+1}/{self.cam_num} ")

            camera_name = _cam.name

            _img_name = self._img_names[camera_name][frame]
            _gmask_name = self._gm_names[camera_name][frame]
            _fgmask_name = self._fg_names[camera_name][frame]

            _img = _cam / DEFAULTS.rgb_images / _img_name
            _gmask = _cam / DEFAULTS.garment_masks / _gmask_name
            _fgmask = _cam / DEFAULTS.foreground_masks / _fgmask_name

            bg = np.array([1,1,1]) if self.white_background else np.array([0, 0, 0])
            image_dict = load_masked_image(_img, _gmask, _fgmask, bg)
            masked_img = image_dict['masked_img'] 
            penalized_mask = image_dict['penalized_mask']
            width, height = masked_img.shape[1], masked_img.shape[0]

            cam_name = _cam.name

            # get camera intrinsic and extrinsic matrices
            intrinsic = np.asarray(self.camera_params[cam_name]["intrinsics"])
            extrinsic = np.asarray(self.camera_params[cam_name]["extrinsics"])

            R, T = np.transpose(extrinsic[:, :3]), extrinsic[:, 3]
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[:2, 2]
            FovY, FovX = focal2fov(fy, height), focal2fov(fx, width)
            image = Image.fromarray(np.array(masked_img, dtype=np.byte), "RGB")
                

            # append camera_info        
            camera_info = CameraInfo(uid=c_idx, R=R, T=T, FovY=FovY, FovX=FovX,  fx=fx, fy=fy, cx=cx, cy=cy, image=image, mask=penalized_mask,
                                    image_path=_img, image_name=cam_name, width=width, height=height)
            camera_info_list.append(camera_info)

        self.cam_info = sorted(camera_info_list.copy(), key = lambda x : x.image_name)
    
    


