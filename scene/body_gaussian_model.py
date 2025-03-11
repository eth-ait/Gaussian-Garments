# import os
# import glob
# import torch
# from torch import nn
# import numpy as np
# from plyfile import PlyData, PlyElement
# from simple_knn._C import distCUDA2
# from arguments import ModelParams
# from utils.defaults import DEFAULTS
# from utils.loss_utils import l1_loss, ssim
# import pickle
# import copy
# import trimesh
# import smplx
# import open3d as o3d
# from sklearn import neighbors
# from sklearn.cluster import KMeans
# from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw, quat_product, unitquat_to_rotmat
# from lbs import load_4DDress_smplx, prepare_lbs
# from recon.parse_scan import parse_scan, SURFACE_LABEL, rasterized
# from scene.mesh_model import MeshModel
# from scene.mesh_gaussian_model import MeshGaussianModel
# from utils.sh_utils import RGB2SH
# from utils.system_utils import mkdir_p
# from utils.graphics_utils import BasicPointCloud
# from utils.graphics_utils import compute_face_orientation
# from utils.io_utils import read_obj, fetchPly, storePly, write_obj
# from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation

# TODO: remove

# def segment_mesh(mesh, valid_mask):
#     '''
#     mseh: o3d.geometry.TriangleMesh
#     valid_mask: np.array(num_F,), bool
#     '''
#     mesh.remove_triangles_by_mask(~valid_mask)
#     mesh.remove_unreferenced_vertices()

#     triangle_clusters, cluster_n_triangles, _ = (mesh.cluster_connected_triangles())

#     # # keep only largest cluster
#     # triangles_to_remove = np.asarray(triangle_clusters) != np.asarray(cluster_n_triangles).argmax()
#     # mesh.remove_triangles_by_mask(triangles_to_remove)
    
#     # remove small clusters
#     triangles_to_remove = np.asarray(cluster_n_triangles)[np.asarray(triangle_clusters)] < 100
#     mesh.remove_triangles_by_mask(triangles_to_remove)

#     mesh.remove_unreferenced_vertices()
#     return mesh

# def get_hair_shoe(source_path):
#     _poisson = os.path.join(source_path, "poisson.obj")
#     # parse recon mesh
#     scan_data = trimesh.load(_poisson)
#     scan_labels = parse_scan(scan_data, source_path, only_fg=False)

#     # extract hair and shoes
#     mesh = o3d.io.read_triangle_mesh(_poisson)
#     hair_mask = (scan_labels[scan_data.faces] == SURFACE_LABEL.index('hair')).all(-1)
#     shoe_mask = (scan_labels[scan_data.faces] == SURFACE_LABEL.index('shoe')).all(-1)

#     hair = segment_mesh(copy.deepcopy(mesh), hair_mask)
#     shoe = segment_mesh(copy.deepcopy(mesh), shoe_mask)
#     # o3d.visualization.draw_geometries([shoe, hair])
#     return hair, shoe

# class BodyGaussianModel(MeshGaussianModel):
#     def __init__(self, args : ModelParams):
#         super(MeshGaussianModel, self).__init__(args.sh_degree)

#         seq_path = os.path.join(DEFAULTS.data_root, args.subject) # "00148/Inner/Take2" # source path
#         _smplx = os.path.join(seq_path, 'Meshes/smplx/*.ply')
#         _smplx = sorted(glob.glob(_smplx))[0]

#         colmap_path = glob.glob("../datas/{}".format("_".join(args.subject.split('/'))))[0]
#         # init body mesh
#         os.makedirs(os.path.join(args.subject_out), exist_ok=True)
#         tem_path = os.path.join(args.subject_out, "body.obj")
#         if not os.path.exists(tem_path):
#             hair, shoe = get_hair_shoe(colmap_path)
#             smplx_body = o3d.io.read_triangle_mesh(_smplx)
#             # combine together
#             smplx_body += shoe
#             smplx_body += hair
#             o3d.io.write_triangle_mesh(tem_path, smplx_body)
#             # o3d.visualization.draw_geometries([smsmplx_bodyplx])
#         else:
#             smplx_body = o3d.io.read_triangle_mesh(tem_path)

#         self.rs_body = rasterized(torch.tensor(np.array(smplx_body.vertices)), torch.tensor(np.array(smplx_body.triangles)), colmap_path)
#         tem_dict = {"vertices":np.array(smplx_body.vertices, dtype=np.float32), "faces": np.array(smplx_body.triangles, dtype=np.int64)} 

#         # load 4ddress data and body mesh
#         self.pose_list, tmp_pose = load_4DDress_smplx(seq_path)
#         # load smplx
#         _npz = os.path.join(DEFAULTS.data_root, f"smplx/SMPLX_{args.gender.upper()}.npz")
#         self.smplx_model = smplx.create(model_path=_npz, model_type='smplx', gender=args.gender.lower(), num_betas=10, use_pca=True, num_pca_comps=12)
#         # get canonical template (at origin)
#         self.cano_vertices, self.blend_weights, self.nb_idx = prepare_lbs(self.smplx_model, tmp_pose, tem_dict['vertices']-tmp_pose['transl'], unpose=True)


#         # init gaussians with body mesh
#         ply_path = os.path.join(args.subject_out, "input.ply")
#         if not os.path.exists(ply_path):
#             xyz = tem_dict['vertices'][tem_dict['faces']].mean(1)
#             dense_pc = o3d.io.read_point_cloud(os.path.join(colmap_path, 'point_cloud.ply'))
#             pc_xyz = np.array(dense_pc.points)
#             pc_rgb = np.array(dense_pc.colors)*255
#             _, nb_idx = neighbors.KDTree(pc_xyz).query(xyz)
#             rgb = pc_rgb[nb_idx.reshape(-1)]
#             # store PC as mesh.ply
#             storePly(ply_path, xyz, rgb)
#         self.create_from_pcd(fetchPly(ply_path), 3.6)

#         # init mesh values
#         self.mesh = MeshModel(tem_dict['vertices'], tem_dict['faces'])
#         self.binding = torch.arange(len(self.mesh.f)).cuda()
#         self.binding_counter = torch.ones(len(self.mesh.f), dtype=torch.int32).cuda()

#         # find occluded gaussians
#         self.occluded = torch.ones_like(self.binding, dtype=bool).cuda()
#         self.skin_color = None

#     # def __init__(self, body_path: str):
#     #     '''
#     #     body_path: the path to output folder 'body' (with body.obj inside)
#     #     '''
#     #     super(MeshGaussianModel, self).__init__(3)
#     #     body = trimesh.load(os.path.join(body_path,"body.obj"))
#     #     self.mesh = MeshModel(np.array(body.vertices, dtype=np.float32), body.faces)

#     #     _local_ply = glob.glob(os.path.join(body_path,"point_cloud/local_point_cloud.ply"))[0]
#     #     self.load_ply(_local_ply)
#     #     self.mesh.v.requires_grad = False

#     def paint_occluded(self, image, mask, name):
#         occluded = (self.rs_body[name][...,0]>0) * ~mask[0].to(torch.bool)
#         _x, _y = torch.where(occluded)
#         occluded = torch.cat([_x.unsqueeze(-1), _y.unsqueeze(-1)], dim=-1)

#         if self.skin_color is None:
#             if not name == '0000': return image, mask
#             visible = image.permute(1,2,0)[image.sum(0) > 0]
#             groups = KMeans(n_clusters=2)
#             groups.fit(visible.cpu())
#             skin_colors = groups.cluster_centers_
#             labels = groups.labels_
#             _, counts = np.unique(labels, return_counts=True)
#             self.skin_color = skin_colors[np.argmax(counts)]

#         image[:, occluded[:,0],occluded[:,1]] = torch.tensor(self.skin_color, dtype=torch.float32).cuda().unsqueeze(-1)
#         return image, torch.ones_like(mask).cuda()
    
#     def save(self, model_path):
#         point_cloud_path = os.path.join(model_path, "point_cloud")
#         self.save_ply(os.path.join(point_cloud_path, "local_point_cloud.ply"), save_local=True)
#         self.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

#     def lbs_frame(self, idx):
#         pose_param = self.pose_list[idx]
#         posed_vertices, _, _ = prepare_lbs(self.smplx_model, pose_param, self.cano_vertices, blend_weights=self.blend_weights, nn_ids=self.nb_idx)
#         posed_vertices += pose_param['transl']
#         self.mesh.v = torch.tensor(posed_vertices, dtype=torch.float32).cuda()
#         self.update_face_coor()

#     def post_each_frame(self, cameras, source_path, reset, is_ff=False):
#         with torch.no_grad():
#             if not is_ff:
#                 if reset:
#                     self._xyz = self.prev_xyz
#                     self._rotation = self.prev_rotation
#                     self._opacity = self.prev_opacity
#                     self._scaling = self.prev_scaling
#                     self._features_dc = self.prev_features_dc
#                     self._features_rest = self.prev_features_rest
#                 else:
#                     mask = self.get_vis_gaussians(cameras, source_path)
#                     self._xyz[~mask] = self.prev_xyz[~mask]
#                     self._rotation[~mask] = self.prev_rotation[~mask]
#                     self._opacity[~mask] = self.prev_opacity[~mask]
#                     self._scaling[~mask] = self.prev_scaling[~mask]
#                     self._features_dc[~mask] = self.prev_features_dc[~mask]
#                     self._features_rest[~mask] = self.prev_features_rest[~mask]
#             # update prev gs
#             self.prev_xyz = self._xyz.detach().clone()
#             self.prev_rotation = self._rotation.detach().clone()
#             self.prev_opacity = self._opacity.detach().clone()
#             self.prev_scaling = self._scaling.detach().clone()
#             self.prev_features_dc = self._features_dc.detach().clone()
#             self.prev_features_rest = self._features_rest.detach().clone()
    
#     def get_vis_gaussians(self, cameras, source_path):
#         pix2face= rasterized(self.mesh.v, self.mesh.f, source_path)
#         votes = torch.zeros(len(self.mesh.f))
#         mask_list = []
#         for cam in cameras: 
#             name = cam.image_name
#             vis_mask = cam.original_image.sum(0)>0
#             votes[np.unique(pix2face[name][(pix2face[name]>-1) * vis_mask[...,None]].cpu().numpy())] += 1 
#             votes[np.unique(pix2face[name][(pix2face[name]>-1) * ~vis_mask[...,None]].cpu().numpy())] -= 1 
#             mask_list.append(vis_mask)
#         valid_face = votes > 0
#         valid_gs = valid_face[self.binding.cpu()]
#         # breakpoint()
#         # import open3d as o3d
#         # pcd = o3d.geometry.PointCloud()
#         # pcd.points =  o3d.utility.Vector3dVector(self.get_xyz[valid_gs].detach().cpu().numpy())
#         # o3d.visualization.draw_geometries([pcd])
#         # from PIL import Image
#         # Image.fromarray(vis_mask.cpu().numpy()).show()
#         return valid_gs
        
#     def load_ply(self, path):
#         plydata = PlyData.read(path)

#         xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
#                         np.asarray(plydata.elements[0]["y"]),
#                         np.asarray(plydata.elements[0]["z"])),  axis=1)
#         opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

#         features_dc = np.zeros((xyz.shape[0], 3, 1))
#         features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
#         features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
#         features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

#         extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
#         extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
#         assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
#         features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
#         for idx, attr_name in enumerate(extra_f_names):
#             features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
#         # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
#         features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

#         scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
#         scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
#         scales = np.zeros((xyz.shape[0], len(scale_names)))
#         for idx, attr_name in enumerate(scale_names):
#             scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

#         rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
#         rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
#         rots = np.zeros((xyz.shape[0], len(rot_names)))
#         for idx, attr_name in enumerate(rot_names):
#             rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

#         # fixed when optimizing
#         self._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
#         self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
#         self._features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
#         self._features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
#         self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)
#         self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
#         self.active_sh_degree = self.max_sh_degree

#         binding_path = os.path.join(os.path.dirname(path), "binding.pkl")
#         with open(binding_path, 'rb') as f:
#             self.binding = pickle.load(f)

#         self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")