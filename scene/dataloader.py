import os
import glob
import random
import torch
import bpy
from PIL import Image
from utils.defaults import DEFAULTS
from utils.graphics_utils import focal2fov
import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from utils.io_utils import load_masked_image, read_obj
from tqdm import tqdm

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
        self.valid_cameras = []

        # locate path in servers
        output_root = Path(DEFAULTS.output_root) / args.subject_out

        self.device = 'cuda'
        self.data_dir = Path(DEFAULTS.data_root) / args.subject
        self.output_dir = args.subject_out 
        self.bg = np.array([1,1,1]) if args.white_background else np.array([0, 0, 0])
        self.random_bg = args.random_bg
        self.blur_mask = args.blur_mask
        self.texture_size = args.texture_size
        self.texture_margin = args.texture_margin


        _template = output_root / DEFAULTS.stage1 / 'template_uv.obj'

        self.template = read_obj(_template)
        sequence_dirs = [seq_dir for seq_dir in self.data_dir.iterdir() if seq_dir.is_dir()]
        

        # collect info across whole dataset
        self.dataset_info = {}
        self.frame_collection = []
        for seq_path in sequence_dirs:
            seq_name = seq_path.name

            print(f"[Locating] {seq_path}")
            info = {}
            # camera info
            # cam_folders = sorted(list(seq_path.glob('00*')))
            cam_folders = sorted([path for path in seq_path.iterdir() if path.is_dir() and path.name != 'smplx'])

            if args.eval:
                info['cam_names'] = [n.name for idx, n in enumerate(cam_folders) if idx % args.llffhold != 0]
            else:
                info['cam_names'] = [n.name for n in cam_folders]
            info['json_path'] = seq_path / 'cameras.json'


            img_names = {}
            gm_names = {}
            fg_names = {}

            print(f'Reading frame info for {seq_name}...')
            # Need to collect filenames for each camera separately in case they are not the same as in ActorsHQ
            cam_to_copy_from = None
            for i, cam_path in tqdm(enumerate(cam_folders)):
                cam_name = cam_path.name

                if cam_to_copy_from is not None:
                    img_files = img_names[cam_to_copy_from]
                    gm_files = gm_names[cam_to_copy_from]
                    fg_files = fg_names[cam_to_copy_from]
                    continue

                img_files = sorted((cam_path/DEFAULTS.rgb_images).glob("*.png"))
                if len(img_files) == 0:
                    img_files = sorted((cam_path/DEFAULTS.rgb_images).glob("*.jpg"))

                gm_files = sorted((cam_path/DEFAULTS.garment_masks).glob("*.png"))
                if len(gm_files) == 0:
                    gm_files = sorted((cam_path/DEFAULTS.garment_masks).glob("*.jpg"))

                fg_files = sorted((cam_path/DEFAULTS.foreground_masks).glob("*.png"))

                img_names[cam_name] = [img.name for img in img_files]
                gm_names[cam_name] = [gm.name for gm in gm_files]
                fg_names[cam_name] = [fg.name for fg in fg_files]

                if i == 1:
                    first_cam = cam_folders[0].name
                    if img_names[cam_name][0] == img_names[first_cam][0]:
                        cam_to_copy_from = first_cam
                

            # frame info
            # img_files = sorted((cam_folders[0] / DEFAULTS.rgb_images).glob("*.png"))
            # gm_files = sorted((cam_folders[0] / DEFAULTS.garment_masks).glob("*.png"))
            # fg_files = sorted((cam_folders[0] / DEFAULTS.foreground_masks).glob("*.png"))

            # info['img_names'] = [img.name for img in img_files]
            # info['gm_names'] = [gm.name for gm in gm_files]
            # info['fg_names'] = [fg.name for fg in fg_files]

            info['img_names'] = img_names
            info['gm_names'] = gm_names
            info['fg_names'] = fg_names


            info['frame_num'] = len(info['img_names'][cam_folders[0].name])
            # collect info
            self.dataset_info[seq_name] = info
            self.frame_collection += [(seq_name, f, c) for f in range(info['frame_num']) for c in info['cam_names']]

        if args.shuffle: random.shuffle(self.frame_collection)
        
    def __len__(self):
        return len(self.frame_collection)
    
    def __getitem__(self, index):
        return self.load_frame(*self.frame_collection[index])

    def load_frame(self, name, frame, cam):
        info = self.dataset_info[name]
        data = {}

        data['current_seq'] = name
        data['current_frame'] = frame
        data['bg'] = np.random.rand(3) if self.random_bg else self.bg

        img_name = info['img_names'][cam][frame]
        gmask_name = info['gm_names'][cam][frame]
        fgmask_name = info['fg_names'][cam][frame]

        # load GT image & mask
        _folder = info['json_path'].parent / cam
        _img = _folder / DEFAULTS.rgb_images / img_name
        _gmask = _folder / DEFAULTS.garment_masks / gmask_name
        _fgmask = _folder / DEFAULTS.foreground_masks / fgmask_name


        image_dict = load_masked_image(_img, _gmask, _fgmask, data['bg'])
        masked_img = image_dict['masked_img'] 
        masked_img = torch.tensor(masked_img, dtype=torch.float32) / 255.
        penalized_mask = image_dict['penalized_mask']
        penalized_mask = torch.tensor(penalized_mask)

        # load cam
        cam_params = json.load(open(info['json_path'], 'r'))[cam]
        data['camera'] = self.get_cam_info(cam_params, masked_img, penalized_mask)

        # load current frame mesh and bake textures
        _mesh = self.output_dir / DEFAULTS.stage2 /  data['current_seq'] / "meshes" / f"frame_{data['current_frame']:05d}.obj"
        _body = self.data_dir / data['current_seq'] / "smplx" / f"{data['current_frame']:05d}.ply"

        data['ambient'], data['normal'], data['mesh_v'] = self.get_maps(_mesh, _body)

        return data


    def get_cam_info(self, params, image, mask):
        # get camera intrinsic and extrinsic matrices
        intrinsic = np.asarray(params["intrinsics"])
        extrinsic = np.asarray(params["extrinsics"])
        h, w, _ = image.shape

        R, T = np.transpose(extrinsic[:, :3]), extrinsic[:, 3]
        fx, fy = intrinsic[0, 0], intrinsic[1, 1] # focal length
        cx, cy = intrinsic[:2, 2] # principal point
        FovY, FovX = focal2fov(fy, h), focal2fov(fx, w)

        return  {"R":R, "T":T, "FoVx":FovX, "FoVy":FovY, "fx":fx, "fy":fy, "cx":cx, "cy":cy, 'colmap_id':np.nan, 'image_name':np.nan, 'uid':params['ids'],
                "image":image.permute(2,0,1), "gt_alpha_mask":mask.unsqueeze(0), "data_device":'cpu'} 
    
    def get_maps(self, _mesh: str, _body: str = None):
        # load current mesh
        mesh = read_obj(_mesh)

        # locate texture path
        _ambient = _mesh.parents[1] / "texture" / "ambient" / f"{_mesh.stem}.png"
        _normal = _mesh.parents[1] / "texture" / "normal" / f"{_mesh.stem}.png"

        if os.path.exists(_ambient) and os.path.exists(_normal): 
            ambient = np.array(Image.open(_ambient)) / 255.
            normal = np.array(Image.open(_normal)) / 255.
        else:
            ambient, normal = self.bake_texture(_mesh, body_path=_body, w=self.texture_size, h=self.texture_size, save=True)

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
            try:
                bpy.ops.wm.ply_import(filepath=str(body_path))
            except:
                print("BODY_MESH", body_path, "NOT LOADED")
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

