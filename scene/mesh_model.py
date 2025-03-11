import einops
import trimesh
import numpy as np
import torch
import torch.nn.functional as F
from utils.io_utils import write_obj
from utils.geometry_utils import *
   
class MeshModel():
    def __init__(self, vt, ft) -> None:
        # template
        self.vt = torch.tensor(vt).cuda()
        self.v = torch.tensor(vt).cuda()
        self.f = torch.tensor(ft).cuda()

        # inertial loss
        self.tar_v = self.vt.clone().cuda()
        self.body = None
        self.collision_faces_ids = None
        self.valid_faces = None

        # material parameters
        self.density = 0.20022
        self.lame_mu = 23600.0
        self.lame_lambda = 44400
        self.bending_coeff = 3.9625778333333325e-05
        self.thickness: float = 4.7e-4
        self.eps: float = 0

        self.init_compute(vt, ft)

    def init_compute(self, v, f):
        v = torch.tensor(v).cuda()
        f = torch.tensor(f).cuda()

        self.v_mass = self.make_v_mass(v, f, self.density)
        self.f_area = self.make_f_area(v, f)

        self.f_connectivity, self.f_connectivity_edges = self.make_connectivity(f)
        self.Dm_inv = self.make_Dm_inv(v, f)

        self.edges = get_vertex_connectivity(f) # list of edges with node index
        # self.T_edge = compute_edge_length(v, self.edges) # length of each edge, with same order as self.edge
        self.virtual_edge, self.ve_len = init_virtual_edge(self.vt, self.f)

               
    def momentum_update(self, v, f, Me=0.3, Mve=0.3):
        v = torch.tensor(v).cuda()
        f = torch.tensor(f).cuda()
        # momentum edge length
        new_Dm = torch.inverse(self.make_Dm_inv(v, f))
        tmp_Dm = torch.inverse(self.make_Dm_inv(self.vt, f))
        self.Dm_inv = torch.inverse(new_Dm * (1-Me) + tmp_Dm * Me)
        # momentum virtual edge length
        new_ve_len = get_ve_len(self.virtual_edge, v, f)
        tmp_ve_len = get_ve_len(self.virtual_edge, self.vt, f)
        self.ve_len = new_ve_len * (1-Mve) + tmp_ve_len * Mve

    def make_v_mass(self, v, f, density):
        v_mass = get_vertex_mass(v, f, density).to('cuda')
        v_mass = v_mass.unsqueeze(-1)
        return v_mass

    def make_f_area(self, v, f):
        """
        Compute areas of each face
        :param v: vertex positions [Vx3]
        :param f: faces [Fx3]
        :param device: pytorch device
        :return: face areas [Fx1]
        """
        f_area = torch.FloatTensor(get_face_areas(v, f)).to('cuda')  # +
        f_area = f_area.unsqueeze(-1)
        return f_area

    def make_connectivity(self, f):
        f_connectivity, f_connectivity_edges = get_face_connectivity_combined(f)
        return f_connectivity, f_connectivity_edges

    def make_Dm_inv(self, v, f):
        """
        Conpute inverse of the deformation gradient matrix (used in stretching energy loss)
        :param v: vertex positions [Vx3]
        :param f: faces [Fx3]
        :return: inverse of the deformation gradient matrix [Fx3x3]
        """
        tri_m = gather_triangles(v.unsqueeze(0), f)[0]

        edges = get_shape_matrix(tri_m)
        edges = edges.permute(0, 2, 1)
        edges_2d = edges_3d_to_2d(edges).permute(0, 2, 1)
        Dm_inv = torch.inverse(edges_2d)
        return Dm_inv

    def bending_energy(self):
        vertices = self.v
        faces = self.f

        f_connectivity = self.f_connectivity
        f_connectivity_edges = self.f_connectivity_edges
        f_area = self.f_area
        bending_coeff = self.bending_coeff

        fn = FaceNormals(vertices.unsqueeze(0), faces.unsqueeze(0))[0]

        n = gather(fn, f_connectivity, 0, 1, 1)
        n0, n1 = torch.unbind(n, dim=-2)

        v = gather(vertices, f_connectivity_edges, 0, 1, 1)
        v0, v1 = torch.unbind(v, dim=-2)
        e = v1 - v0
        l = torch.norm(e, dim=-1, keepdim=True)
        e_norm = e / l

        f_area_repeat = f_area.repeat(1, f_connectivity.shape[-1])
        a = torch.gather(f_area_repeat, 0, f_connectivity).sum(dim=-1)

        cos = (n0 * n1).sum(dim=-1)
        sin = (e_norm * torch.linalg.cross(n0, n1)).sum(dim=-1)
        theta = torch.atan2(sin, cos)

        scale = l[..., 0] ** 2 / (4 * a)

        energy = bending_coeff * scale * (theta ** 2) / 2
        loss = energy.sum()
        return loss
        
    def stretching_energy(self):
        Dm_inv = self.Dm_inv # (28992,2,2)
        f_area = self.f_area[None, ..., 0] # (1, 28992)
        device = 'cuda'

        v = self.v
        f = self.f
        triangles = gather_triangles(v.unsqueeze(0), f)[0]

        triangles_list = [triangles]
        lame_mu_stack = create_stack(triangles_list, torch.tensor([self.lame_mu]).to(device))
        lame_lambda_stack = create_stack(triangles_list, torch.tensor([self.lame_lambda]).to(device))
        triangles = torch.cat(triangles_list, dim=0)


        F = deformation_gradient(triangles, Dm_inv)
        G = green_strain_tensor(F)

        I = torch.eye(2).to(device)
        I = einops.repeat(I, 'm n -> k m n', k=G.shape[0])

        G_trace = G.diagonal(dim1=-1, dim2=-2).sum(-1)  # trace

        S = lame_mu_stack[:, None, None] * G + 0.5 * lame_lambda_stack[:, None, None] * G_trace[:, None, None] * I
        energy_density_matrix = S.permute(0, 2, 1) @ G
        energy_density = energy_density_matrix.diagonal(dim1=-1, dim2=-2).sum(-1)  # trace
        f_area = f_area[0]

        energy = f_area * self.thickness * energy_density
        return energy.sum()

    def panelize_virtual(self,):
        # only panelize compressed virtual edge
        return F.relu(self.ve_len - get_ve_len(self.virtual_edge, self.v, self.f)).mean()

    def init_body(self, o3d_body):
        if isinstance(o3d_body, trimesh.Trimesh):
            body = o3d_body
        else:
            body = trimesh.Trimesh(vertices=np.array(o3d_body.vertices), faces=np.array(o3d_body.triangles))

        self.body = body
        # Compute distances in the current timestep
        face_center = body.vertices[body.faces].mean(-2)
        normals = FaceNormals(torch.tensor(body.vertices).unsqueeze(0), torch.tensor(body.faces).unsqueeze(0))[0]

        self.nn_points = torch.tensor(face_center[self.collision_faces_ids]).cuda().squeeze(1)
        self.nn_normals = normals[self.collision_faces_ids].cuda().squeeze(1)

    def collision(self, eps: float = 1e-3):
        distance = ((self.v - self.nn_points) * self.nn_normals).sum(dim=-1)
        interpenetration = torch.maximum(eps - distance, torch.FloatTensor([0]).to('cuda'))

        interpenetration = interpenetration.pow(3)
        loss = interpenetration.sum(-1)
        return loss

    def inertial(self, timestep=1/30):
        x_diff = self.tar_v - self.v.detach()
        num = (x_diff * self.v_mass * x_diff).sum(dim=-1).unsqueeze(1)
        den = 2 * timestep ** 2
        loss = num / den
        return loss.sum()
    
    def gravitational_energy(self, g=9.81):
        energy = (g * self.v_mass * self.v[:,1]).sum()
        return energy
    
    def get_energy_loss(self, args, use_body):
        loss = {}
        loss['bending'] = self.bending_energy() *  args.lambda_bending
        loss['stretching'] = self.stretching_energy() *  args.lambda_stretching
        if use_body:
            loss['collision'] = self.collision() *  args.lambda_collision
        else:
            loss['virtual_edge'] = self.panelize_virtual() * args.lambda_virtual
        # if inertial: loss['inertial'] = self.inertial() *  args.lambda_inertial
        return loss
