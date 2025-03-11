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
        # self.fg_label = ['skin', 'hair', 'glove', 'shoe'] if body else args.garment_type
        # self.panelize_labels = ['background', 'skin', 'hair', 'glove', 'shoe'] if body else ['background', args.garment_type]
        self.white_background = args.white_background

        # locate sequence
        seq_path = f'{DEFAULTS.data_root}/{args.subject}/{args.sequence}'


        # locate camera
        self.cam_paths = sorted([os.path.join(seq_path, fn) for fn in os.listdir(seq_path) if '00' in fn])
        self.camera_params = json.load(open(os.path.join(seq_path, 'cameras.json'), 'r'))
        self.cam_num = len(self.cam_paths)
        # frame info

        glob_str = os.path.join(self.cam_paths[0], "capture_images/*.png")

        _imgs = sorted(glob.glob(glob_str))

        min_frame = int(_imgs[0].split('/')[-1].split(".png")[0])
        self.start_frame = min_frame # if args.template_frame is None else int(args.template_frame)
        self.frame_num = len(_imgs)
        # smplx info
        self.smplx_list = sorted(glob.glob(os.path.join(seq_path, "smplx/*.ply")))

    def __len__(self,):
        return self.frame_num
    

    def load_frame(self, idx):
        frame_idx = self.start_frame + idx # frame filename
        camera_infos = []

        # process all cameras
        for idx, _cam in enumerate(self.cam_paths):
            print(f"Reading frame {frame_idx} camera {idx+1}/{self.cam_num} ")

            _img = os.path.join(_cam,"capture_images",f"{frame_idx:05d}.png")
            _lab = os.path.join(_cam,"capture_labels",f"{frame_idx:05d}.png")
            bg = np.array([1,1,1]) if self.white_background else np.array([0, 0, 0])
            image_dict = load_masked_image(_img, _lab, bg)
            masked_img = image_dict['masked_img'] 
            mask = image_dict['mask']
            width, height = masked_img.shape[1], masked_img.shape[0]

            cam_name = _cam.split('/')[-1]

            # get camera intrinsic and extrinsic matrices
            intrinsic = np.asarray(self.camera_params[cam_name]["intrinsics"])
            extrinsic = np.asarray(self.camera_params[cam_name]["extrinsics"])

            R, T = np.transpose(extrinsic[:, :3]), extrinsic[:, 3]
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[:2, 2]
            FovY, FovX = focal2fov(fy, height), focal2fov(fx, width)

            # label = np.array(Image.open(_lab))[...,None]
            # if isinstance(self.fg_label, list):
            #     mask = np.zeros_like(label)
            #     for key in self.fg_label:
            #         assert not key == "full_body", "The fg_label shouldn't be 'fullbody' when training Body guassians"
            #         mask += label == self.MaskLabel[key]
            # else:
            #     mask = label == self.MaskLabel[self.fg_label]
            #     if self.fg_label == 'full_body': mask = ~mask


            # masked_img =  np.array(image) * mask + 255 * bg * ~mask
            image = Image.fromarray(np.array(masked_img, dtype=np.byte), "RGB")

            # get panelize mask
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

        self.cam_infos = sorted(camera_infos.copy(), key = lambda x : x.image_name)
    
    


