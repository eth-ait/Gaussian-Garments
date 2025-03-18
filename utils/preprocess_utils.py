import os
import sys
import shutil
import json
import glob
import numpy as np
from PIL import Image
from pathlib import Path

from utils.io_utils import load_masked_image
from .defaults import DEFAULTS

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class PrepareDataset:
    def __init__(self, source_root, target_root, cam_type, fg_label="full_body"):
        """
        Initializes the preprocessing utility for a given subject, sequence, and garment type.
        This method reads data from the input folder and writes the processed data to the output folder subdirectory 'stage1'. 
        It creates the necessary subdirectories within the 'data' directory, including directories for images, masks, and text files. 
        The images directory is populated with the first frame from each camera, and the text files are prepared for COLMAP format input.
        Args:
            subject (str): The subject identifier e.g. '00190/Inner'
            sequence (str): The sequence identifier e.g. 'Take2'
        """
        self.fg_label = fg_label
        self.cam_type = cam_type

        self.target_root = target_root
        self.source_root = source_root
        
        already_populated = self.prepare_output_dir()
        if already_populated:
            return
        self.populate_imgs_dir()
        self.export_colmap()
        
    def prepare_output_dir(self):
        if os.path.exists(self.target_root): 
            delete = input("Output path already exists. Please grant permission to remove current folder. (Y/N) ")
            if delete.lower() == 'y':
                shutil.rmtree(self.target_root)
            else:
                print(f"Using existing '{self.target_root}'.")
                return True

        os.makedirs(self.target_root, exist_ok=True)
        print(f"-------\nOutput folder created at {self.target_root}\n-------")
        self._img_out = os.path.join(self.target_root, "images")
        self._mask_out = os.path.join(self.target_root, "masks")
        self._txt_out = os.path.join(self.target_root, "txt")
        os.makedirs(self._img_out, exist_ok=True)
        os.makedirs(self._mask_out, exist_ok=True)
        os.makedirs(self._txt_out, exist_ok=True)

        return False
    
    def populate_imgs_dir(self):
        """
        Read the images, labels, and masks and writes them to the output directory within respective folders. 
        """

        cam_paths = sorted([path for path in self.source_root.iterdir() if path.is_dir() and path.name != 'smplx'])
        
        # _imgs = sorted((cam_paths[0]/DEFAULTS.rgb_images).glob("*.png"))
        # start_img_name = _imgs[0].name

        # _gmasks = sorted((cam_paths[0]/DEFAULTS.garment_masks).glob("*.png"))
        # start_gmask_name = _gmasks[0].name

        # _fgmasks = sorted((cam_paths[0]/DEFAULTS.foreground_masks).glob("*.png"))
        # start_fgmask_name = _fgmasks[0].name

        cam_num = len(cam_paths)

        for idx, _cam in enumerate(cam_paths):
            print(f"Reading first frame for camera {idx+1}/{cam_num} ")
            cam_name = _cam.name

            _imgs = sorted((_cam/DEFAULTS.rgb_images).glob("*.png"))
            if len(_imgs) == 0:
                _imgs = sorted((_cam/DEFAULTS.rgb_images).glob("*.jpg"))

            _gmasks = sorted((_cam/DEFAULTS.garment_masks).glob("*.png"))
            if len(_gmasks) == 0:
                _gmasks = sorted((_cam/DEFAULTS.garment_masks).glob("*.jpg"))

            _fgmasks = sorted((_cam/DEFAULTS.foreground_masks).glob("*.png"))

            _img = _imgs[0]
            _gmask = _gmasks[0]
            _fgmask = _fgmasks[0]

            # _img = _cam / DEFAULTS.rgb_images / start_img_name
            # _gmask = _cam / DEFAULTS.garment_masks / start_gmask_name
            # _fgmask = _cam / DEFAULTS.foreground_masks / start_fgmask_name
            image_dict = load_masked_image(_img, _gmask, _fgmask, np.array([0,1,0]))
            mask = image_dict['mask']
            masked_img = image_dict['masked_img']

        
            
            cam_name = _cam.name

            image = Image.fromarray(np.array(masked_img, dtype=np.byte), "RGB")
            image.save(os.path.join(self._img_out, cam_name+'.png'))
            mask = (mask * 255).astype(np.uint8)[..., 0]
            mask = Image.fromarray(mask)
            mask.save(os.path.join(self._mask_out, cam_name+'.png.png'))
    
    def export_colmap(self):
        """
        Exports the camera parameters to a COLMAP compatible format in txt subdirectory.
        """

        # locate camera_list within image_path
        image_fns = sorted(os.listdir(self._img_out))
        camera_list = [image_fn.split('.')[0] for image_fn in image_fns]
        camera_params = json.load(open(os.path.join(self.source_root, 'cameras.json'), 'r'))
        json.dump(camera_params, open(os.path.join(self.target_root, 'cameras.json'), 'w'))

        for ID, camera_id in enumerate(camera_list):
            m = "w" if ID == 0 else "a"

            _path = os.path.join(self._img_out, camera_id + '.png')
            image = Image.open(_path) if os.path.exists(_path) else None
            width, height = image.size

            # get camera intrinsic and extrinsic matrices
            intrinsic = np.asarray(camera_params[camera_id]["intrinsics"])
            extrinsic = np.asarray(camera_params[camera_id]["extrinsics"])

            R, T = extrinsic[:, :3], extrinsic[:, 3]
            fx, fy = intrinsic[0,0], intrinsic[1,1]
            px, py = intrinsic[:2,2]

            # write cameras.txt
            content = f"{ID} {self.cam_type} {width} {height} {fx} {fy} {px} {py}"
            with open(os.path.join(self._txt_out,"cameras.txt"), m) as file:
                file.write(content+"\n")

            qvec = rotmat2qvec(R)
            tvec = T

            # write images.txt
            content = f"{ID} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} {ID} {camera_id + '.png'}"
            with open(os.path.join(self._txt_out, "images.txt"), m) as file:
                file.write(content+"\n"+"\n")
            
        # new empty points3D.txt
        with open(os.path.join(self._txt_out, "points3D.txt"), 'w'):
            pass