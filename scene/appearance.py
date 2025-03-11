import os
import torch
from torch import nn

# Fourier encoding
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            self.freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                out_dim += d

        self.out_dim = out_dim

    def forward(self, inputs):
        self.freq_bands = self.freq_bands.type_as(inputs)
        outputs = []
        if self.kwargs['include_input']:
            outputs.append(inputs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                outputs.append(p_fn(inputs * freq))
        return torch.cat(outputs, -1)


def get_embedder(multires=6, i=0, input_dim=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dim,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class AppearanceNet(nn.Module):
    def __init__(self, in_chl, out_chl, hidden_chl=256, num_layers=4) -> None:
        super().__init__()
        self.embedder, embedder_dim = get_embedder(input_dim=in_chl)

        layers = []
        for i in range(num_layers-2):
            layers.append(nn.Linear(hidden_chl, hidden_chl))
            layers.append(nn.Tanh())
        self.hidden = nn.Sequential(*layers)
        self.layer_in = nn.Sequential(nn.Linear(embedder_dim, hidden_chl), nn.Tanh())
        self.layer_out = nn.Sequential(nn.Linear(hidden_chl, out_chl))


    def forward(self, features):
        '''
        features: (num_gs, 4), concatednated textures (include normal and ambient occlusion)
        result: (num_gs, out_chl), predicted SH coefficients
        '''
        embeded_feature = self.embedder(features)
        # encode
        skip = self.layer_in(embeded_feature)
        out = self.hidden(skip) + skip
        return self.layer_out(out)


class ConvDown(nn.Module):
    def __init__(self, in_chl, out_chl, group=1) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_chl, in_chl, 3, 1, 1, groups=group),
            nn.LeakyReLU(),
            nn.Conv2d(in_chl, out_chl, 3, 2, 1, groups=group),
            nn.LeakyReLU(),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_chl, out_chl, 1, 2, groups=group),
        )

    def forward(self, features):
        return self.branch1(features) + self.branch2(features)
    
class ConvUp(nn.Module):
    def __init__(self, in_chl, out_chl, group=1) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_chl, in_chl, 3, 1, 1, groups=group),
            nn.LeakyReLU(),
            nn.Conv2d(in_chl, out_chl, 3, 1, 1, groups=group),
            nn.LeakyReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_chl, out_chl, 1, 1, groups=group),
        )

    def forward(self, features):
        out = self.upsample(features[None])
        return (self.branch1(out) + self.branch2(out)).squeeze(0)

# class ConvUNet(nn.Module):
#     def __init__(self, in_chl, out_chl, group=2) -> None:
#         super().__init__()

#         # encoder
#         self.cd512 = ConvDown(in_chl, 16, 1)
#         self.cd256 = ConvDown(16, 32, group)
#         self.cd128 = ConvDown(32, 64, group)
#         self.cd64 = ConvDown(64, 128, group)
#         self.cd32 = ConvDown(128, 256, group)
        
#         # encoder
#         self.cu512 = ConvUp(16, out_chl, 1)
#         self.cu256 = ConvUp(32, 16, group)
#         self.cu128 = ConvUp(64, 32, group)
#         self.cu64 = ConvUp(128, 64, group)
#         self.cu32 = ConvUp(256, 128, group)

#     def forward(self, features):
#         # encode
#         skip256 = self.cd512(features)
#         skip128 = self.cd256(skip256)
#         skip64 = self.cd128(skip128)
#         skip32 = self.cd64(skip64)
#         out16 = self.cd32(skip32)

#         # encode
#         out32 = self.cu32(out16)
#         out64 = self.cu64(out32 + skip32)
#         out128 = self.cu128(out64 + skip64)
#         out256 = self.cu256(out128 + skip128)
#         out512 = self.cu512(out256 + skip256)
#         return out512.permute(1,2,0)

class ConvUNet(nn.Module):
    def __init__(self, in_chl, out_chl, group=2) -> None:
        super().__init__()

        # encoder
        self.cd512 = ConvDown(in_chl, 64, 1)
        self.cd256 = ConvDown(64, 128, group)
        self.cd128 = ConvDown(128, 256, group)
        
        # encoder
        self.cu512 = ConvUp(64, out_chl, 1)
        self.cu256 = ConvUp(128, 64, group)
        self.cu128 = ConvUp(256, 128, group)

    def forward(self, features):
        # encode
        skip256 = self.cd512(features)
        skip128 = self.cd256(skip256)
        out64 = self.cd128(skip128)

        # encode
        out128 = self.cu128(out64)
        out256 = self.cu256(out128 + skip128)
        out512 = self.cu512(out256 + skip256)
        return out512.permute(1,2,0)

# class PhongNet(nn.Module):
#     def __init__(self, out_chl, hidden_dim=128, hidden_num=2) -> None:
#         super().__init__()
#         self.shadow_embedder, shadow_dim = get_embedder(input_dim=1)
#         self.specular_embedder, specular_dim = get_embedder(input_dim=6)

#         hidden = [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()] * hidden_num

#         self.shadow_mlp = nn.Sequential(
#                                         nn.Linear(shadow_dim, hidden_dim), 
#                                         nn.Tanh(),
#                                         *hidden,
#                                         nn.Linear(hidden_dim, out_chl)
#                                         )
#         self.specular_mlp = nn.Sequential(
#                                         nn.Linear(specular_dim, hidden_dim), 
#                                         nn.Tanh(),
#                                         *hidden,
#                                         nn.Linear(hidden_dim, out_chl)
#                                         )
#     def forward(self, shadow, specular):
#         '''
#         features: (num_gs, 4), concatednated textures (include normal and ambient occlusion)
#         result: (num_gs, out_chl), predicted SH coefficients
#         '''
#         embeded_shadow = self.shadow_embedder(shadow)
#         embeded_shadow = self.specular_embedder(specular)
#         # encode
#         out0 = self.layer0(embeded_feature)
#         out1 = self.layer1(out0)
#         out2 = self.layer2(out1)
#         # decode
#         out3 = self.layer3(torch.cat([out1, out2], dim=-1))
#         result = self.layer4(torch.cat([out0, out3], dim=-1))
#         return result