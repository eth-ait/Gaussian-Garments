import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from huepy import yellow
from scene.appearance import AppearanceNet, ConvUNet, get_embedder
from scene.avatar_gaussian_model import AvatarGaussianModel
from scene.styleunet.styleunet import SWGAN_unet

class AvatarNet(nn.Module):
    def __init__(self, args, gaussians: AvatarGaussianModel):
        super(AvatarNet, self).__init__()
        self.debug = args.debug
        self.embedder, embedder_dim = get_embedder(input_dim=7)
        self.texture_size = args.texture_size

        input_size = args.texture_size
        output_size = args.texture_size
        style_dim, n_mlp = args.texture_size, 2
        # # self.color_net = SWGAN_unet(input_size, 3, (args.sh_degree+1)**2 * 3 + 3, output_size, style_dim, n_mlp)
        # self.shadow_net = SWGAN_unet(input_size, 7, (args.sh_degree+1)**2 * 3, output_size, style_dim, n_mlp)
        # self.shadow_net = SWGAN_unet(input_size, embedder_dim, (args.sh_degree+1)**2 * 3, output_size, style_dim, n_mlp)
        # self.color_style = torch.ones([1, style_dim], dtype=torch.float32, device='cuda') / np.sqrt(style_dim)
        # if args.add_position:
        #     self.shadow_net = SWGAN_unet(input_size, 6, (args.sh_degree+1)**2 * 3 + 3, output_size, style_dim, n_mlp)
        # else:
        # self.shadow_net = SWGAN_unet(input_size, 4, (args.sh_degree+1)**2 * 3 + 7, output_size, style_dim, n_mlp)
        self.no_xyz = args.no_xyz
        if args.no_xyz: self.shadow_net = SWGAN_unet(input_size, 4, (args.sh_degree+1)**2 * 3, output_size, style_dim, n_mlp)
        else: self.shadow_net = SWGAN_unet(input_size, 4, (args.sh_degree+1)**2 * 3 + 3, output_size, style_dim, n_mlp)

        self.shadow_style = torch.ones([1, style_dim], dtype=torch.float32, device='cuda') / np.sqrt(style_dim)
        self.viewdir_net = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(64, 128, 4, 2, 1)
            )
        # self.pos_net = SWGAN_unet(input_size, 3, 7, output_size, style_dim, n_mlp)
        # self.pos_net = SWGAN_unet(input_size, 3, 3, output_size, style_dim, n_mlp)
        # self.pos_style = torch.ones([1, style_dim], dtype=torch.float32, device='cuda') / np.sqrt(style_dim)

        # self.pos_net = AppearanceNet(3, 7)
        # self.pos_net = AppearanceNet(3, 3)
        
        # self.convU = ConvUNet(embedder_dim, (args.sh_degree+1)**2 * 3)
        # self.shadow_mlp = AppearanceNet(4, (args.sh_degree+1)**2 * 3)
        # self.specular_mlp = AppearanceNet(6, (args.sh_degree+1)**2 * 3)

        self.gaussians = gaussians
        self.register_parameter("xyz", self.gaussians._xyz)
        self.register_parameter("feature_dc", self.gaussians._features_dc)
        self.register_parameter("feature_rest", self.gaussians._features_rest)
        self.register_parameter("scaling", self.gaussians._scaling)
        self.register_parameter("rotation", self.gaussians._rotation)
        self.register_parameter("opacity", self.gaussians._opacity)
        
    def training_setup(self, args):
        epoch = 1
        l = [
            # {'params': self.mlps.parameters(), 'lr': args.lr_init},
            # {'params': self.shadow_mlp.parameters(), 'lr': args.lr_init},
            # {'params': self.specular_mlp.parameters(), 'lr': args.lr_init},
            # {'params': self.convU.parameters(), 'lr': args.lr_init},
            # {'params': self.color_net.parameters(), 'lr': args.lr_init},
            # {'params': self.pos_net.parameters(), 'lr': args.lr_init},
            {'params': self.shadow_net.parameters(), 'lr': args.lr_init},
            {'params': self.viewdir_net.parameters(), 'lr': args.lr_init},
            {'params': [self.gaussians._xyz], 'lr': args.position_lr_init * 2.5},
            {'params': [self.gaussians._features_dc], 'lr': args.feature_lr},
            {'params': [self.gaussians._features_rest], 'lr': args.feature_lr / 20.0},
            {'params': [self.gaussians._scaling], 'lr': args.scaling_lr},
            {'params': [self.gaussians._rotation], 'lr': args.rotation_lr},
            {'params': [self.gaussians._opacity], 'lr': args.opacity_lr},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        if os.path.exists(os.path.join(args.ckpt_path,"ckpt/net.pt")):
            epoch = self.load_ckpt(args.ckpt_path) + 1 # next epoch
        if args.eval:
            self.eval()
        return epoch

    # def update_lr(self, optm, iter):
    #     alpha = 0.05
    #     progress = self.iter_idx / self.iter_num
    #     learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
    #     lr = self.lr_init * learning_factor
    #     for param_group in optm:
    #         param_group['lr'] = lr
    #     return lr

    def forward(self, ambient, normal, camera):
        visible_mask = None
        # ################### ray-casting finding visible faces ###################
        visible_mask = self.gaussians.get_visible_mask(camera.camera_center)

        # ################## barycentric project from 2d to 3d ###################
        # triangles = self.gaussians.mesh.v[self.gaussians.mesh.f][self.gaussians.binding]
        # a, b, c = self.gaussians.gs_bc
        # gs_3d = a[:,None] * triangles[:,0] + b[:,None] * triangles[:,1] + c[:,None] * triangles[:,2] 
        # breakpoint()
        # vis = gs_3d[visible_mask]
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points =  o3d.utility.Vector3dVector(vis.detach().cpu().numpy())
        # cam = o3d.geometry.PointCloud()
        # cam.points =  o3d.utility.Vector3dVector(camera[None].detach().cpu().numpy())
        # o3d.visualization.draw_geometries([pcd, cam])

        ################### compute dot(normal, view_direction) ###################
        # dir_pp = gs_3d - camera
        # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        # dir_normal = (normal*2-1).permute(1,2,0)[self.gaussians.gaussian_mask]
        # dir_normal_normalized = dir_normal/dir_normal.norm(dim=1, keepdim=True)
        # visible_mask = (dir_normal * dir_pp_normalized).sum(-1) < 0.1
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points =  o3d.utility.Vector3dVector(gs_3d[visible_mask].detach().cpu().numpy())
        # cam = o3d.geometry.PointCloud()
        # cam.points =  o3d.utility.Vector3dVector(camera[None].detach().cpu().numpy())
        # o3d.visualization.draw_geometries([pcd, cam])


        # #########################  color * shadow ################################
        # color_out = self.color_net([self.color_style], normal[None])[0].squeeze(0).permute(1,2,0)
        # shadow_out = self.shadow_net([self.shadow_style], ambient[None])[0].squeeze(0).permute(1,2,0)

        # self.gaussians._xyz = color_out[self.gaussians.gaussian_mask][:,:3]
        # shs = color_out[self.gaussians.gaussian_mask][:,3:] * shadow_out[self.gaussians.gaussian_mask]
        # shs = shs.reshape(-1,(self.gaussians.max_sh_degree+1)**2,3)
        # self.gaussians._features_dc = shs[:,:1,:]
        # self.gaussians._features_rest = shs[:,1:,:]
        # return color_out[self.gaussians.gaussian_mask][:,3:], shadow_out[self.gaussians.gaussian_mask]

        # ######################### StyleUnet : GS + shadow ################################
        # combined = torch.cat([ambient, normal])
        # shadow_out = self.shadow_net([self.shadow_style], combined[None], randomize_noise=False)[0].squeeze(0).permute(1,2,0)
        # self.gaussians.shs = self.gaussians.get_features + shadow_out[self.gaussians.gaussian_mask].reshape(-1,(self.gaussians.max_sh_degree+1)**2,3)
        # return shadow_out[self.gaussians.gaussian_mask], visible_mask

        ######################### StyleUnet : GS and shadow + directon ################################
        # local view
        # dir_pp = self.gaussians.get_barycentric_3d() - camera
        # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        # local_viewdir = torch.bmm(self.gaussians.face_orien_mat[self.gaussians.binding].permute(0,2,1), dir_pp_normalized[..., None]).squeeze(-1)
        # viewdir_map = torch.zeros_like(normal.permute(1,2,0)).cuda()
        # viewdir_map[self.gaussians.gs_u, self.gaussians.gs_u] = dir_pp_normalized # local_viewdir
        # combined = torch.cat([ambient, normal, viewdir_map.permute(2,0,1)])

        # combined = torch.cat([ambient, normal])
        # shadow_out = self.shadow_net([self.shadow_style], combined[None], randomize_noise=False)[0].squeeze(0).permute(1,2,0)
        # self.gaussians.shs = self.gaussians.get_features + shadow_out[self.gaussians.gaussian_mask].reshape(-1,(self.gaussians.max_sh_degree+1)**2,3)
        # return shadow_out[self.gaussians.gaussian_mask], visible_mask

        ######################### StyleUnet : GS & shadow + view direction in middle ################################
        # world normal into camera coor
        nw_pad = torch.cat([(normal*2-1), torch.ones([1,*normal.shape[1:]]).cuda()]) * (normal.sum(0, keepdim=True) > 0)
        nc = torch.einsum('ab,bcd->acd', camera.world_view_transform, nw_pad)[:3]
        normalized_nc = F.normalize(nc, dim=0)


        # local view
        dir_pp = self.gaussians.get_barycentric_3d() - camera.camera_center
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        local_viewdir = torch.bmm(self.gaussians.face_orien_mat[self.gaussians.binding].permute(0,2,1), dir_pp_normalized[..., None]).squeeze(-1)
        viewdir_map = torch.zeros_like(normal.permute(1,2,0)).cuda()
        viewdir_map[self.gaussians.gs_u, self.gaussians.gs_u] = local_viewdir
        view_feature = self.viewdir_net(viewdir_map.permute(2,0,1)[None])
        # combined = torch.cat([ambient, normal, viewdir_map.permute(2,0,1)])


        # # uv position
        # pos = torch.zeros([*normal.shape[1:],2]).cuda()
        # pos[self.gaussians.gaussian_mask] = torch.tensor(np.array([self.gaussians.gs_u, self.gaussians.gs_u]).T).cuda() / normal.shape[-1]
        # pos = pos.permute(2,0,1)
        # try: 
        #     combined = torch.cat([ambient, normalized_nc, pos])
        #     shadow_out = self.shadow_net([self.shadow_style], combined[None], randomize_noise=False, view_feature=view_feature)[0].squeeze(0).permute(1,2,0)
        # except:
        combined = torch.cat([ambient, normalized_nc])
        shadow_out = self.shadow_net([self.shadow_style], combined[None], randomize_noise=False, view_feature=view_feature)[0].squeeze(0).permute(1,2,0)
        # # pred xyz
        # triangles = pos.cuda()[self.gaussians.mesh.f][self.gaussians.binding]
        # a, b, c = self.gaussians.gs_bc
        # gs_3d = a[:,None] * triangles[:,0] + b[:,None] * triangles[:,1] + c[:,None] * triangles[:,2] 
        # pos_map = torch.zeros([self.texture_size, self.texture_size, 3]).cuda()
        # pos_map[self.gaussians.gaussian_mask] = gs_3d
        # pos_map = pos_map.permute(2, 0, 1)
        # # style7
        # pos_out = self.pos_net([self.pos_style], pos_map[None], randomize_noise=False)[0].squeeze(0).permute(1,2,0)
        # self.gaussians.local_xyz = self.gaussians._xyz + pos_out[self.gaussians.gaussian_mask][:,:3]
        # self.gaussians._rotation = pos_out[self.gaussians.gaussian_mask][:,3:]
        # shadow_out = shadow_out[self.gaussians.gaussian_mask].reshape(self.gaussians.num_gs,-1,3)
        # self.gaussians.shs = self.gaussians.get_features + shadow_out
        # # style3
        # pos_out = self.pos_net([self.pos_style], pos_map[None], randomize_noise=False)[0].squeeze(0).permute(1,2,0)
        # self.gaussians.local_xyz = self.gaussians._xyz + pos_out[self.gaussians.gaussian_mask]
        # shadow_out = shadow_out[self.gaussians.gaussian_mask].reshape(self.gaussians.num_gs,-1,3)
        # self.gaussians.shs = self.gaussians.get_features + shadow_out
        # # MLP7
        # pos_out = self.pos_net(gs_3d)
        # self.gaussians.local_xyz = self.gaussians._xyz + pos_out[:,:3]
        # self.gaussians._rotation = pos_out[:,3:]
        # shadow_out = shadow_out[self.gaussians.gaussian_mask].reshape(self.gaussians.num_gs,-1,3)
        # self.gaussians.shs = self.gaussians.get_features + shadow_out
        # # MLP3
        # pos_out = self.pos_net(gs_3d)
        # self.gaussians.local_xyz = self.gaussians._xyz + pos_out
        # shadow_out = shadow_out[self.gaussians.gaussian_mask].reshape(self.gaussians.num_gs,-1,3)
        # self.gaussians.shs = self.gaussians.get_features + shadow_out


        if self.no_xyz:
            # only rgb
            self.gaussians.local_xyz = self.gaussians._xyz
            shadow_out = shadow_out[self.gaussians.gaussian_mask].reshape(self.gaussians.num_gs,-1,3)
            self.gaussians.shs = self.gaussians.get_features + shadow_out
        else:    
            # rgb + xyz
            self.gaussians.local_xyz = self.gaussians._xyz + shadow_out[self.gaussians.gaussian_mask][:,:3]
            shadow_out = shadow_out[self.gaussians.gaussian_mask][:,3:].reshape(self.gaussians.num_gs,-1,3)
            self.gaussians.shs = self.gaussians.get_features + shadow_out

        return shadow_out, visible_mask

        # ######################### StyleUnet : embeded GS and shadow ################################
        # combined = torch.cat([ambient, normal]).permute(1,2,0)
        # embeded_feature = self.embedder(combined[self.gaussians.gaussian_mask])
        # embed_map = torch.zeros(combined.shape[0], combined.shape[1], embeded_feature.shape[-1]).cuda()
        # embed_map[self.gaussians.gs_u, self.gaussians.gs_u] = embeded_feature
        # embed_map = embed_map.permute(2,0,1)
        # shadow_out = self.shadow_net([self.shadow_style], embed_map[None], randomize_noise=False)[0].squeeze(0).permute(1,2,0)
        # self.gaussians.shs = self.gaussians.get_features + shadow_out[self.gaussians.gaussian_mask].reshape(-1,(self.gaussians.max_sh_degree+1)**2,3)
        # return shadow_out[self.gaussians.gaussian_mask], visible_mask


        # ######################### Phong : GS and specular shadow ################################
        # # local view
        # dir_pp = self.gaussians.get_barycentric_3d() - camera
        # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        # local_viewdir = torch.bmm(self.gaussians.face_orien_mat[self.gaussians.binding].permute(0,2,1), dir_pp_normalized[..., None]).squeeze(-1)
        # viewdir_map = torch.zeros_like(normal.permute(1,2,0)).cuda()
        # viewdir_map[self.gaussians.gs_u, self.gaussians.gs_u] = local_viewdir
        # specular_features = torch.cat([normal.permute(1,2,0), viewdir_map], dim=-1)[self.gaussians.gaussian_mask]
        # specular_offset = self.specular_mlp(specular_features).reshape(-1,(self.gaussians.max_sh_degree+1)**2,3)

        # shadow_features = torch.cat([ambient.permute(1,2,0), viewdir_map], dim=-1)[self.gaussians.gaussian_mask]
        # # shadow_features = ambient.permute(1,2,0)[self.gaussians.gaussian_mask]
        # shadow_offset = self.shadow_mlp(shadow_features).reshape(-1,(self.gaussians.max_sh_degree+1)**2,3)
        # self.gaussians.shs = self.gaussians.get_features * shadow_offset + specular_offset
        # return specular_offset, visible_mask

        # # ######################### ConvUNet : GS and shadow + direction ################################
        # dir_pp = self.gaussians.get_barycentric_3d() - camera
        # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        # viewdir_map = torch.zeros_like(normal.permute(1,2,0)).cuda()
        # viewdir_map[self.gaussians.gs_u, self.gaussians.gs_u] = dir_pp_normalized
        # combined = torch.cat([ambient, normal, viewdir_map.permute(2,0,1)]).permute(1,2,0)
        # embeded_feature = self.embedder(combined[self.gaussians.gaussian_mask])
        # embed_map = torch.zeros(combined.shape[0], combined.shape[1], embeded_feature.shape[-1]).cuda()
        # embed_map[self.gaussians.gs_u, self.gaussians.gs_u] = embeded_feature

        # shs_offset = self.convU(embed_map.permute(2,0,1))
        # self.gaussians.shs = self.gaussians.get_features + shs_offset[self.gaussians.gaussian_mask].reshape(-1,(self.gaussians.max_sh_degree+1)**2,3)
        # return shs_offset[self.gaussians.gaussian_mask], visible_mask

    def save_ckpt(self, model_path, epoch, save_optm = True, name="ckpt"):
        path = os.path.join(model_path, name)
        os.makedirs(path, exist_ok = True)
        with open(path+"/info.txt", 'w') as file:
            file.write(f"Epoch {epoch}\n")
        
        net_dict = {
            'epoch': epoch,
            'activate_sh_degree': self.gaussians.active_sh_degree,
            'avatar_net': self.state_dict(),
        }
        print('Saving networks to ', path + '/net.pt')
        torch.save(net_dict, path + '/net.pt')

        if save_optm:
            optm_dict = {
                'avatar_net': self.optimizer.state_dict(),
            }
            print('Saving optimizers to ', path + '/optm.pt')
            torch.save(optm_dict, path + '/optm.pt')

    def load_ckpt(self, path, load_optm = True, name="ckpt"):
        path = os.path.join(path, name)
        print(yellow('Loading networks from ', path + '/net.pt'))
        net_dict = torch.load(path + '/net.pt')
        if 'avatar_net' in net_dict:
            self.load_state_dict(net_dict['avatar_net'])
            if 'activate_sh_degree' in net_dict:
                self.gaussians.active_sh_degree = net_dict['activate_sh_degree']
            self.gaussians.active_sh_degree = 3
        else:
            print('[WARNING] Cannot find "avatar_net" from the network checkpoint!')
        epoch_idx = net_dict['epoch']

        if load_optm and os.path.exists(path + '/optm.pt'):
            print('Loading optimizers from ', path + '/optm.pt')
            optm_dict = torch.load(path + '/optm.pt')
            if 'avatar_net' in optm_dict:
                self.optimizer.load_state_dict(optm_dict['avatar_net'])
            else:
                print('[WARNING] Cannot find "avatar_net" from the optimizer checkpoint!')

        return epoch_idx
    
    def post_each_frame(self,):
        self.gaussians.prev_feature = self.gaussians.get_features.detach()