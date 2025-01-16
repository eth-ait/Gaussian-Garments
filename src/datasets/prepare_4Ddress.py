import os
import json
import glob
import numpy as np
from PIL import Image

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))


from .prepare_data import PrepareDataset
from src.common.defaults import DEFAULTS

class Prepare4DDress(PrepareDataset):
    def __init__(self, data_folder):
        super().__init__(data_folder)

        SURFACE_LABELS = ['full_body', 'skin', 'upper', 'lower', 'hair', 'glove', 'shoe', 'outer', 'background']
        GRAY_VALUES = np.array([255, 128, 98, 158, 188, 218, 38, 68, 255])

        self.data_path = DEFAULTS.host_root + self.data_folder  
        self.mask_label = dict(zip(SURFACE_LABELS, GRAY_VALUES))
        self.fg_label = SURFACE_LABELS[0]
        self.load_cam_params()
        self.populate_directory()
    
    def load_cam_params(self):
        camera_params = json.load(open(os.path.join(self.data_path, 'cameras.json'), 'r'))
        json.dump(camera_params, open(os.path.join(self.output_root, 'cameras.json'), 'w'))
    
    def populate_directory(self):
        cam_paths = sorted([os.path.join(self.data_path, fn) for fn in os.listdir(self.data_path) if '00' in fn])

        _imgs = sorted(glob.glob(os.path.join(cam_paths[0], "capture_images/*.png")))
        start_frame = int(_imgs[0].split('/')[-1].split(".png")[0])
        cam_num = len(cam_paths)

        for idx, _cam in enumerate(cam_paths):
            print(f"Reading frame {start_frame} camera {idx+1}/{cam_num} ")
            _img = os.path.join(_cam,"capture_images",f"{start_frame:05d}.png")
            _lab = os.path.join(_cam,"capture_labels",f"{start_frame:05d}.png")
            cam_name = _cam.split('/')[-1]

            image = np.array(Image.open(_img))
            label = np.array(Image.open(_lab))
            mask = label == self.mask_label[self.fg_label]
            if self.fg_label == 'full_body': mask = ~mask

            # use green background
            masked_img = image * mask[...,None] + np.array([0,255,0]) * ~mask[...,None]
            image = Image.fromarray(np.array(masked_img, dtype=np.byte), "RGB")
            image.save(os.path.join(self._img_out, cam_name+'.png'))
            label = Image.fromarray(label)
            label.save(os.path.join(self._mask_out, cam_name+'.png')) 
            mask = Image.fromarray(mask)
            mask.save(os.path.join(self._mask_out, cam_name+'.png.png'))