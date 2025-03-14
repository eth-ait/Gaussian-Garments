from pathlib import Path
import sys
import os
import bpy
import copy
import json
import glob
import pickle
import torch
from torch.utils.data import Dataset

import trimesh
import open3d as o3d
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from gaussian_renderer import doll_render
from matplotlib import pyplot as plt
from utils.defaults import DEFAULTS
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from arguments import ParamGroup
from scene.cameras import Camera, get_cam_info
from scene import MeshGaussianModel
from scene.avatar_gaussian_model import AvatarSimulationModel
from scene.avatar_net import AvatarNet
from utils.graphics_utils import focal2fov
from utils.io_utils import read_obj, write_obj
from aitviewer.headless import HeadlessRenderer
from aitviewer.scene.camera import PinholeCamera
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.meshes import VariableTopologyMeshes as VTMeshes
from aitviewer.scene.camera import OpenCVCamera
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_aitviewer_cam(cam: dict):
    cam_intrinsics = np.array(cam['intrinsics'])
    cam_extrinsics = np.eye(4)
    cam_extrinsics[:3] = np.array(cam['extrinsics'])

    # # The OpenCVCamera class expects extrinsics with Y pointing down, so we flip both Y and Z axis to keep a
    # # positive determinant.
    # cam_extrinsics[1:3, :3] *= -1.0

    # Create an OpenCV camera.
    camera = OpenCVCamera(cam_intrinsics, cam_extrinsics[:3], w, h, viewer=viewer)
    return camera

def from_gs_to_ait(cam):
    extrinsic, intrinsic = np.zeros([3,4]), np.eye(3)
    extrinsic[:, :3] = cam.R.T
    extrinsic[:, 3] = cam.T

    intrinsic[0, 0], intrinsic[1, 1] = cam.fx, cam.fy
    intrinsic[0, 2], intrinsic[1, 2] = cam.cx, cam.cy

    h, w = cam.original_image.shape[1:]

    return OpenCVCamera(intrinsic, extrinsic, w, h, viewer=viewer)


def from_ait_to_gs(cam, w, h):
    cam = cam.to_opencv_camera() if hasattr(cam, 'to_opencv_camera') else cam
    intrinsic = cam.K[0]
    extrinsic = cam.Rt[0]
    
    R, T = np.transpose(extrinsic[:, :3]), extrinsic[:, 3]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[:2, 2]
    FovY, FovX = focal2fov(fy, h), focal2fov(fx, w)

    return Camera(R=R, T=T, FoVx=FovX, FoVy=FovY, fx=fx, fy=fy, cx=cx, cy=cy, colmap_id=None, image_name=None, uid=None,
                    image=torch.zeros([3,h,w]), gt_alpha_mask=torch.zeros([1,h,w]), data_device='cuda')

def adjust_color(num):
    """
    Adjusts color to look nice.
    :param color: [..., 4] RGBA color
    :return: updated color
    """
    color = np.array(cmap(num))
    color[..., :3] /= color[..., :3].max(axis=-1, keepdims=True)
    color[..., :3] *= 0.3
    color[..., :3] += 0.3
    return tuple(color)

def scan_render(vtmesh, cam=None):
    for l in viewer.scene.lights: l.enabled = False
    viewer.scene.nodes = viewer.scene.nodes[:4]
    viewer.scene.add(vtmesh)

    if cam is None:
        # horizontal view
        target = viewer.scene.nodes[-1].current_center
        position = target + np.array([0,0,3])
        camera = PinholeCamera(position, target, w, h, viewer=viewer)
    else:
        if isinstance(cam, Camera):
            target = viewer.scene.nodes[-1].current_center
            position = cam.T #  + (target - cam.T) * 0.2
            camera = PinholeCamera(position, target, w, h, viewer=viewer)
        else:
            camera = cam

    viewer.set_temp_camera(camera)
    image = np.array(viewer.get_frame())
    depth = np.array(viewer.get_depth())
    return torch.tensor(image), torch.tensor(depth), camera

def ait_render(verts=None, faces=None, vb=None, fb=None, cam=None):
    '''
    v, f : np.array
    '''
    for l in viewer.scene.lights: l.enabled = True
    viewer.scene.nodes = viewer.scene.nodes[:4]

    if verts is not None and faces is not None:
        for v, f in zip(verts, faces):
            mesh = Meshes(v, f, color=adjust_color(len(viewer.scene.nodes)*0.1), draw_edges=False)
            mesh.backface_culling = False
            viewer.scene.add(mesh)

    if vb is not None and fb is not None:
        body = Meshes(vb, fb, draw_edges=False)
        body.backface_culling = False
        viewer.scene.add(body)

    if cam is None:
        # horizontal view
        target = viewer.scene.nodes[-1].current_center
        position = target + np.array([0,0,3])
        camera = PinholeCamera(position, target, w, h, viewer=viewer)
    else:
        if isinstance(cam, Camera):
            # target = viewer.scene.nodes[-1].current_center # + np.array([0,0.3,0])
            # position = cam.T # + (target - cam.T) * 0.5
            # camera = PinholeCamera(position, target, w, h, viewer=viewer)
            camera = from_gs_to_ait(cam).to_pinhole_camera()
        else:
            camera = cam

    viewer.set_temp_camera(camera)
    image = np.array(viewer.get_frame())
    depth = np.array(viewer.get_depth())
    return torch.tensor(image), torch.tensor(depth), camera


def bake_texture(mesh_path, obstacles,  body_path=None, w=512, h=512, texture_margin=5, show=False, save=False):
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
            bpy.ops.wm.ply_import(filepath=body_path)
        except:
            print(body_path)
    for _o in obstacles:
        bpy.ops.wm.obj_import(filepath=_o)
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
    bpy.ops.object.bake(type='AO', margin=texture_margin)


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
    bpy.ops.object.bake(type='NORMAL', normal_space='OBJECT', margin=texture_margin)

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

class Doll(MeshGaussianModel):
    def __init__(self, args, garment_names) -> None:
        super(MeshGaussianModel, self).__init__(args.sh_degree)
        self.active_sh_degree = args.sh_degree

        # get garment gaussian (temporal version --> to include simulator)
        self.garments = []
        self.avatar_nets = []
        # for _avatar, _tmp in zip(args.avatar_list, args.garment_list):
        for garment_name in garment_names:
            _tmp = Path(DEFAULTS.output_root) / garment_name
            gaussian = AvatarSimulationModel(_tmp, texture_size=args.texture_size)
            self.garments.append(gaussian)
            avatar = AvatarNet(args, gaussian).to('cuda')
            avatar.load_ckpt(_tmp / DEFAULTS.stage3, load_optm=False)
            self.avatar_nets.append(avatar)

        # assign edited textures
        self.xyz_list = None # TODO: Do we need that?
        # self.xyz_list = [] # debug: reset xyz
        # if args.texture_list is None: args.texture_list = [None] * len(args.garment_list)
        # for i, texture in enumerate(args.texture_list):
        #     if texture is not None:
        #         _texture_path = os.path.join(args.avatar_list[i], "edit_texture", texture)
        #         self.garments[i].load_texture(_texture_path)
        #         self.xyz_list.append(self.garments[i]._xyz) # debug: reset xyz
        #     else:
        #         self.xyz_list.append(None) # debug: reset xyz

        self.body = None

    def update_garments(self, vert_list, ambient_list, normal_list, camera):
        style_list = []
        for i, vert in enumerate(vert_list):
            avatar, garment = self.avatar_nets[i], self.garments[i]
            garment.mesh.v = torch.tensor(vert, dtype=torch.float32).cuda()
            garment.update_face_coor()

            # forward avatar_net
            style_output, _ = avatar(ambient_list[i], normal_list[i], camera)
            style_list.append(style_output)
            # debug: reset xyz
            # if self.xyz_list[i] is not None:
            #     garment._xyz[self.xyz_list[i]!=0] = self.xyz_list[i][self.xyz_list[i]!=0]
            
        return style_list

    def get_o3d_mesh(self, verts, faces):
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(verts, o3d.core.float32)
        mesh.triangle.indices = o3d.core.Tensor(faces, o3d.core.int32)
        return mesh
        
    def prepare_gaussian(self, p_cam, style_list):
        # xyz = torch.cat([g.get_xyz for g in self.garments])
        xyz = torch.cat([g.get_barycentric_3d() for g in self.garments])
        bindings = torch.cat([g.binding for g in self.garments])
        vis_mask = self.get_visible_mask(p_cam, xyz, bindings)

        self.xyz = torch.cat([g.get_final_xyz for g in self.garments])[vis_mask]
        self.rotation = torch.cat([g.get_rotation for g in self.garments])[vis_mask]
        self.features = torch.cat([g.shs for g in self.garments])[vis_mask]
        self.raw_features = torch.cat([g.get_features for g in self.garments])[vis_mask]
        self.style_features = torch.cat([g for g in style_list])[vis_mask]
        self.opacity = torch.cat([g.get_opacity for g in self.garments])[vis_mask]
        self.scaling = torch.cat([g.get_scaling for g in self.garments])[vis_mask]
        self.covariance = torch.cat([g.get_covariance() for g in self.garments])[vis_mask]

    def get_visible_mask(self, camera, xyz, binding, dot_product_t=-0.7):
        '''
        compute the mask, for filtering gaussians that are visible
        params:
            @camera: torch.tensor(3,)
        output:
            @vis_mask: torch.tensor(num_gs)
        '''
        device = camera.device
        # init scene
        scene = o3d.t.geometry.RaycastingScene()
        for g in self.garments:
            scene.add_triangles(self.get_o3d_mesh(g.mesh.v.cpu().numpy(), g.mesh.f.cpu().numpy()))
        # scene.add_triangles(self.body)
        geometry_ids = torch.cat([i * torch.ones(g._xyz.shape[0], dtype=int) for i, g in enumerate(self.garments)])
        num_gs = len(self.garments) + 1

        # cast ray from camera to GS
        ray_o = camera[None].repeat(len(xyz),1)
        _d = xyz - ray_o
        norm_d = _d.norm(dim=1, keepdim=True)
        ray_d = _d / norm_d
        rays = o3d.core.Tensor(torch.cat([ray_o, ray_d], axis=-1).detach().cpu().numpy())

        # Compute the ray intersections.
        EPSILON = 5e-2 # max distance between intersection point and gaussians
        ans = scene.cast_rays(rays)
        # mesh id based mask
        vis_mask = (ans['geometry_ids'].numpy() == np.array(geometry_ids))
        vis_mask += (ans['geometry_ids'].numpy() >= num_gs)
        vis_mask = torch.tensor(vis_mask)
        return vis_mask.to(device)
        
    def render(self, viewpoint_camera, pc, args, bg, body_image, _body_mask, override_shs=None):
        image, depth, alpha = doll_render(viewpoint_camera, pc, args, bg, override_shs=override_shs)
        image = torch.clip(image, min=0, max=1).permute(1,2,0).detach().cpu()*255
        alpha = alpha.permute(1,2,0).detach().cpu()*255
        # garment image
        garment = torch.cat([image, alpha], dim=-1) * ~_body_mask.unsqueeze(-1)
        garment = Image.fromarray(np.array(garment, dtype=np.uint8))
        # body image
        body = torch.cat([body_image, torch.ones_like(alpha)*255], dim=-1)
        body= Image.fromarray(np.array(body, dtype=np.uint8))
        
        output = Image.alpha_composite(body, garment)
        return np.array(output)[...,:3]
        
class Simulation(Dataset):
    def __init__(self, args):
        # simulation and templates
        self.simu_pkl = pickle.load(open(args.simu_path, 'rb'))
        self.garment_names = self.simu_pkl['garment_names']
        # self.outfit_name = "_".join(self.garment_names)

        models_root = Path(DEFAULTS.output_root)
        # uv_tmp_path = [glob.glob(os.path.join("../datas", name, '*_uv.obj'))[0] for name in self.outfit_name]
        uv_tmp_path = [models_root / name / DEFAULTS.stage1 / 'template_uv.obj' for name in self.garment_names]
        self.uv_tmp_list = [read_obj(path) for path in uv_tmp_path]
        # camera info

        cam_params_path = models_root / self.garment_names[0] / DEFAULTS.stage1 / 'cameras.json'
        cam_params = json.load(open(cam_params_path, 'r'))
        self.gs_cams = {k:get_cam_info(v) for k,v in cam_params.items()}
        self.ait_cams = {k:get_aitviewer_cam(v) for k,v in cam_params.items()}

        # Construct Gaussian Doll
        self.doll = Doll(args, self.garment_names)

        # Simulation output path
        self.output = args.output_path
        os.makedirs(self.output, exist_ok=True)

    def __len__(self,):
        return len(self.simu_pkl['pred'])

    def __getitem__(self, index):
        return index
    
    def get_vertices(self, vertices):
        output = []
        for i, uv_mesh in enumerate(self.uv_tmp_list):
            output.append(vertices[:len(uv_mesh['vertices'])])
            try: vertices = vertices[len(uv_mesh['vertices']):]
            except: pass
        return output

    def prepare_frame(self, idx):
        vert_list = self.get_vertices(self.simu_pkl['pred'][idx])

        # store body
        os.makedirs(os.path.join(self.output, 'body'), exist_ok=True)
        _body = os.path.join(self.output, 'body', f'{idx:05d}.ply')
        if not os.path.exists(_body): 
            body = trimesh.Trimesh(vertices=self.simu_pkl['obstacle'][idx], faces=self.simu_pkl['obstacle_faces'], process=False)
            body.export(_body)
            
        # store meshes
        _mesh_list = [os.path.join(self.output, _g, "meshes", f"{idx:05d}.obj") for _g in self.garment_names]
        for i, _mesh in enumerate(_mesh_list):
            if not os.path.exists(_mesh):
                # store garments
                os.makedirs(os.path.dirname(_mesh), exist_ok=True)
                output = copy.deepcopy(self.uv_tmp_list[i])
                assert len(output['vertices']) == len(vert_list[i]), "Num of Vertices mismatch"
                output['vertices'] = vert_list[i]
                write_obj(output, _mesh)

        ambient_list, normal_list = [], []
        for _mesh in _mesh_list:
            # bake textures
            _ambient = _mesh.replace("meshes/","texture/ambient/").replace(".obj",".png")
            _normal = _mesh.replace("meshes/","texture/normal/").replace(".obj",".png")
            if not (os.path.exists(_ambient) and os.path.exists(_normal)):
                obstacle_meshes = [_m for _m in _mesh_list if _m != _mesh]
                ambient, normal = bake_texture(_mesh,  obstacle_meshes, _body, save=True)
            else:
                ambient = np.array(Image.open(_ambient)) / 255.
                normal = np.array(Image.open(_normal)) / 255.
            ambient = torch.tensor(ambient, dtype=torch.float32).unsqueeze(0).cuda()
            normal = torch.tensor(normal, dtype=torch.float32).permute(2,0,1).cuda()
            ambient_list.append(ambient)
            normal_list.append(normal)
            
        return vert_list, ambient_list, normal_list
    
    def forward(self, index, camera):
        vert_list, ambient_list, normal_list = self.prepare_frame(index)
        # update garments and body
        style_list = self.doll.update_garments(vert_list, ambient_list, normal_list, camera)
        self.doll.body = self.doll.get_o3d_mesh(self.simu_pkl['obstacle'][index], faces=self.simu_pkl['obstacle_faces'])
        # select visible gaussians
        self.doll.prepare_gaussian(camera.camera_center, style_list)

def registration_to_pkl(args):
    if not os.path.exists(args.simu_path):
        _meshes = glob.glob(os.path.join(args.scratch_path, 'Registration_artur_44', args.registration, 'meshes/frame_*.obj'))
        mesh_list = sorted(_meshes, key=lambda x: int(x[:-4].split('frame_')[-1]))
        
        take_folder = args.garment_list[0].split("Take")[0] + "T" +  args.registration.split('/')[-1][1:]
        try: 
            body_list = sorted(glob.glob(os.path.join(args.ssd_path, take_folder, "Meshes/smplx/*.ply")))
            assert len(body_list), "No body found"
        except:
            body_list = sorted(glob.glob(os.path.join(args.scratch_path, take_folder, "Meshes/smplx/*.ply")))
            assert len(body_list), "No body found"


        pred = []
        obstacle = []
        for _body, _mesh in zip(body_list, mesh_list):
            mesh = read_obj(_mesh)
            body = trimesh.load(open(_body, "rb"), "PLY")
            pred.append(mesh['vertices'])
            obstacle.append(body.vertices)
        
        output = {"pred": np.array(pred),
                "obstacle": np.array(obstacle),
                "obstacle_faces": np.array(body.faces),
                "garment_names": [args.registration.split('/')[-2]]}
        pickle.dump(output, open(args.simu_path, 'wb'))

def load_scan(args):
    scan_list = []
    atlas_mesh_list = sorted(glob.glob(os.path.join(args.source_path , args.gt_mesh, "Meshes_pkl/atlas-f*.pkl")))
    mesh_list = sorted(glob.glob(os.path.join(args.source_path , args.gt_mesh, "Meshes_pkl/mesh-f*.pkl")))

    for _atlas, _mesh in zip(atlas_mesh_list, mesh_list):
        try:
            # load scan mesh and atlas data
            _texture = _atlas.replace('.pkl', '.png')
            if not os.path.exists(_texture):
                texture = pickle.load(open(_atlas, "rb"))
                Image.fromarray(texture).transpose(method=Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB").save(_texture)

            mesh_dict = pickle.load(open(_mesh, "rb"))
            # init visualiation obj
            scan_mesh = VTMeshes([mesh_dict['vertices']], [mesh_dict['faces']], uv_coords=[mesh_dict['uvs']], texture_paths=[_texture])
            scan_mesh.backface_culling = False
        except:
            breakpoint()
        scan_list.append(scan_mesh)

    return scan_list

class Config(ParamGroup):
    def __init__(self, parser) -> None:
        self.name = "renders"
        self.texture_size = 512
        self.gender = 'female'
        self.add_position = False
        super().__init__(parser, "Visualize doll Parameters")


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    cp = Config(parser)
    parser.add_argument("--HQ", action="store_true")
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument('--gt_mesh', type=str, required=False, default='')
    parser.add_argument('--simu_path', type=str, required=True, default='')
    parser.add_argument('--output_path', type=str, required=True, default='')
    parser.add_argument("--avatar_list", "-al", required=False, nargs='+', type=str, default=[], help="path to avatar model folder")
    parser.add_argument("--garment_list", "-gl", required=False, nargs='+', type=str, default=[], help="template for garments")
    parser.add_argument("--texture_list", "-tl", required=False, nargs='+', type=str, help="path to texture pkl")
    parser.add_argument('--registration', type=str, required=False, default='')
    args = parser.parse_args(sys.argv[1:])
    # path processing

    if args.HQ: w, h = 3004, 4092
    else: w, h = 940, 1280

    args.source_path = DEFAULTS.data_root
    args.store_path = DEFAULTS.output_root
    viewer = HeadlessRenderer(size=(w, h))
    viewer.playback_fps = 30
    viewer.shadows_enabled = False
    viewer.scene.origin.enabled = False
    cmap = plt.get_cmap('gist_rainbow')

    args.output_path = args.output_path
    args.avatar_list = [os.path.join(args.scratch_path, 'Registration_artur_44', path) for path in args.avatar_list]
    if len(args.registration): registration_to_pkl(args)
    if len(args.gt_mesh): scan_list = load_scan(args)


    # init objects
    simu = Simulation(args)
    with torch.no_grad():
        bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        for idx in tqdm(simu, desc="Rendering Sequence"):
            if idx < args.start_from: continue
            if idx >= len(simu): break
            # body mesh
            _body, _body_depth, ait_cam = ait_render(vb=simu.simu_pkl['obstacle'][idx], fb=simu.simu_pkl['obstacle_faces'])
            gs_cam = from_ait_to_gs(ait_cam, w, h)
            simu.forward(idx, gs_cam)
            # garment mesh
            vert_list = [g.mesh.v.cpu().numpy() for g in simu.doll.garments]
            face_list = [g.mesh.f.cpu().numpy() for g in simu.doll.garments]
            _garment, _body_garm_depth, _ = ait_render(vert_list, face_list, simu.simu_pkl['obstacle'][idx], simu.simu_pkl['obstacle_faces'], ait_cam)
            _, _garm_depth, _ = ait_render(vert_list, face_list, cam=ait_cam)
            _body_mask = _body_garm_depth < _garm_depth

            if len(args.gt_mesh):
                _scan, _, _ = scan_render(scan_list[idx], ait_cam)

            # final
            _final = simu.doll.render(gs_cam, simu.doll, args, bg, _body, _body_mask)
            # raw gaussian
            _raw = simu.doll.render(gs_cam, simu.doll, args, bg, _body, _body_mask, simu.doll.raw_features)
            # style output
            _style = simu.doll.render(gs_cam, simu.doll, args, bg, _body, _body_mask, simu.doll.style_features)
            
            if len(args.gt_mesh):
                image = np.concatenate([_garment, _final, _scan], axis=1)
            else: 
                image = np.concatenate([_raw, _style, _final, _garment], axis=1)
            os.makedirs(os.path.join(simu.output, f"{args.name}"), exist_ok=True)
            Image.fromarray(np.array(image, dtype=np.uint8)).save(os.path.join(simu.output, f"{args.name}/{idx:04d}.png"))

    print("Simulation Finished")