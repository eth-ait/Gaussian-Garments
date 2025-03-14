import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from huepy import yellow
from scene.appearance import get_embedder
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
        self.no_xyz = args.no_xyz
        if args.no_xyz: 
            self.shadow_net = SWGAN_unet(input_size, 4, (args.sh_degree+1)**2 * 3, output_size, style_dim, n_mlp)
        else: 
            self.shadow_net = SWGAN_unet(input_size, 4, (args.sh_degree+1)**2 * 3 + 3, output_size, style_dim, n_mlp)

        self.shadow_style = torch.ones([1, style_dim], dtype=torch.float32, device='cuda') / np.sqrt(style_dim)
        self.viewdir_net = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(64, 128, 4, 2, 1)
            )

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

    def forward(self, ambient, normal, camera):
        visible_mask = None
        # ################### ray-casting finding visible faces ###################
        visible_mask = self.gaussians.get_visible_mask(camera.camera_center)
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

        combined = torch.cat([ambient, normalized_nc])
        shadow_out = self.shadow_net([self.shadow_style], combined[None], randomize_noise=False, view_feature=view_feature)[0].squeeze(0).permute(1,2,0)

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