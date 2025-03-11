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
                # print(f"Error: Permission denied. Unable to proceed because the folder '{self.target_root}' already exists.")
                # sys.exit(1)  # Terminate the program with an error status

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
        
        # SURFACE_LABELS = ['full_body', 'skin', 'upper', 'lower', 'hair', 'glove', 'shoe', 'outer', 'background']
        # GRAY_VALUES = np.array([255, 128, 98, 158, 188, 218, 38, 68, 255])
        # mask_label = dict(zip(SURFACE_LABELS, GRAY_VALUES))

        cam_paths = sorted([os.path.join(self.source_root, fn) for fn in os.listdir(self.source_root) if '00' in fn])
        _imgs = sorted(glob.glob(os.path.join(cam_paths[0], "capture_images/*.png")))
        start_frame = int(_imgs[0].split('/')[-1].split(".png")[0])
        cam_num = len(cam_paths)

        for idx, _cam in enumerate(cam_paths):
            print(f"Reading frame {start_frame} camera {idx+1}/{cam_num} ")
            _img = os.path.join(_cam,"capture_images",f"{start_frame:05d}.png")
            _lab = os.path.join(_cam,"capture_labels",f"{start_frame:05d}.png")
            image_dict = load_masked_image(_img, _lab, np.array([0,1,0]))
            mask = image_dict['mask']
            masked_img = image_dict['masked_img']


            
            cam_name = _cam.split('/')[-1]

            # image = np.array(Image.open(_img)) / 255
            # mask = np.array(Image.open(_lab)) / 255
            # mask = label == mask_label[self.fg_label]
            # if self.fg_label == 'full_body': mask = ~mask

            # print('mask', mask.max(), mask.min())
            # assert False

            # use green background
            # masked_img = image * mask[...,None] + np.array([0,1,0]) * (1 - mask[...,None])
            # masked_img = (masked_img * 255).astype(np.uint8)

            image = Image.fromarray(np.array(masked_img, dtype=np.byte), "RGB")
            image.save(os.path.join(self._img_out, cam_name+'.png'))
            # label = Image.fromarray(label)
            # label.save(os.path.join(self._mask_out, cam_name+'.png')) 
            mask = (mask * 255).astype(np.uint8)
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