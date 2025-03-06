import einops
import scipy
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d

def FaceNormals(v, f, normalize=True):
    """
    :param vertices: FloatTensor of shape (batch_size, num_vertices, 3)
    :param faces: LongTensor of shape (batch_size, num_faces, 3)
    :return: face_normals: FloatTensor of shape (batch_size, num_faces, 3)
    """

    if v.shape[0] > 1 and f.shape[0] == 1:
        f = f.repeat(v.shape[0], 1, 1)

    v_repeat = einops.repeat(v, 'b m n -> b m k n', k=f.shape[-1])
    f_repeat = einops.repeat(f, 'b m n -> b m n k', k=v.shape[-1])
    triangles = torch.gather(v_repeat, 1, f_repeat)

    # Compute face normals
    v0, v1, v2 = torch.unbind(triangles, dim=-2)
    e1 = v0 - v1
    e2 = v2 - v1
    face_normals = torch.linalg.cross(e2, e1)

    if normalize:
        face_normals = F.normalize(face_normals, dim=-1)

    return face_normals


def get_face_areas(vertices, faces):
    """
    Computes the area of each face in the mesh

    :param vertices: FloatTensor or numpy array of shape (num_vertices, 3)
    :param faces: LongTensor or numpy array of shape (num_faces, 3)
    :return: areas: FloatTensor or numpy array of shape (num_faces,)
    """
    if type(vertices) == torch.Tensor:
        vertices = vertices.detach().cpu().numpy()

    if type(faces) == torch.Tensor:
        faces = faces.detach().cpu().numpy()
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    u = v2 - v0
    v = v1 - v0

    if u.shape[-1] == 2:
        out = np.abs(np.cross(u, v)) / 2.0
    else:
        out = np.linalg.norm(np.cross(u, v), axis=-1) / 2.0
    return out
    
def get_vertex_mass(vertices, faces, density):
    '''
    Computes the mass of each vertex according to triangle areas and fabric density
    '''
    vertices = vertices.cpu()
    faces = faces.cpu()

    areas = get_face_areas(vertices, faces)
    triangle_masses = density * areas

    vertex_masses = np.zeros(vertices.shape[0])
    np.add.at(vertex_masses, faces[:, 0], triangle_masses / 3)
    np.add.at(vertex_masses, faces[:, 1], triangle_masses / 3)
    np.add.at(vertex_masses, faces[:, 2], triangle_masses / 3)

    vertex_masses = torch.FloatTensor(vertex_masses)

    return vertex_masses

def get_vertex_connectivity(faces):
    '''
    Returns a list of unique edges in the mesh.
    Each edge contains the indices of the vertices it connects
    '''
    device = 'cpu'
    if type(faces) == torch.Tensor:
        device = faces.device
        faces = faces.detach().cpu().numpy()

    edges = set()
    for f in faces:
        num_vertices = len(f)
        for i in range(num_vertices):
            j = (i + 1) % num_vertices
            edges.add(tuple(sorted([f[i], f[j]])))

    edges = torch.LongTensor(list(edges)).to(device)
    return edges

# def get_vertex_connectivity(faces, vertices=None):
#     '''
#     Returns a list of unique edges in the mesh.
#     Each edge contains the indices of the vertices it connects
#     '''
#     device = 'cpu'
#     if type(faces) == torch.Tensor:
#         device = faces.device
#         faces = faces.detach().cpu().numpy()

#     if vertices is not None:
#         vertices = vertices.detach().cpu().numpy()
#         num_v = vertices.shape[0]
#         weight_sum = torch.zeros((num_v, num_v), dtype=float)

#         edges = set()
#         for f in faces:
#             num_vertices = len(f)
#             for i in range(num_vertices):
#                 j = (i + 1) % num_vertices
#                 edges.add(tuple(sorted([f[i], f[j]])))

#                 # compute cot weight
#                 k = (i + 2) % num_vertices
#                 vki = vertices[f[i]] - vertices[f[k]]
#                 vkj = vertices[f[j]] - vertices[f[k]]
#                 cos = np.dot(vki, vkj) / (np.linalg.norm(vki)*np.linalg.norm(vkj))
#                 cos = torch.tensor(cos)
#                 cot = 1 / torch.tan(torch.acos(cos))

#                 weight_sum[f[i],f[j]] += 0.5*cot
#                 weight_sum[f[j],f[i]] += 0.5*cot


#         edges = torch.LongTensor(list(edges)).to(device)
#         weight_sum = weight_sum.to(device)
#         return edges, weight_sum
#     else:
#         edges = set()
#         for f in faces:
#             num_vertices = len(f)
#             for i in range(num_vertices):
#                 j = (i + 1) % num_vertices
#                 edges.add(tuple(sorted([f[i], f[j]])))

#         edges = torch.LongTensor(list(edges)).to(device)
#         return edges

def get_face_connectivity_combined(faces):
    """
    Finds the faces that are connected in a mesh
    :param faces: LongTensor of shape (num_faces, 3)
    :return: adjacent_faces: pairs of face indices LongTensor of shape (num_edges, 2)
    :return: adjacent_face_edges: pairs of node indices that comprise the edges connecting the corresponding faces
     LongTensor of shape (num_edges, 2)
    """

    device = 'cpu'
    if type(faces) == torch.Tensor:
        device = faces.device
        faces = faces.detach().cpu().numpy()

    edges = get_vertex_connectivity(faces).cpu().numpy()

    G = {tuple(e): [] for e in edges}
    for i, f in enumerate(faces):
        n = len(f)
        for j in range(n):
            k = (j + 1) % n
            e = tuple(sorted([f[j], f[k]]))
            G[e] += [i]

    adjacent_faces = []
    adjacent_face_edges = []

    for key in G:
        if len(G[key]) >= 3:
            G[key] = G[key][:2]
        if len(G[key]) == 2:
            adjacent_faces += [G[key]]
            adjacent_face_edges += [list(key)]

    adjacent_faces = torch.LongTensor(adjacent_faces).to(device)
    adjacent_face_edges = torch.LongTensor(adjacent_face_edges).to(device)

    return adjacent_faces, adjacent_face_edges

def gather_triangles(vertices, faces):
    """
    Generate a tensor of triangles from a tensor of vertices and faces

    :param vertices: FloatTensor of shape (batch_size, num_vertices, 3)
    :param faces: LongTensor of shape (num_faces, 3)
    :return: triangles: FloatTensor of shape (batch_size, num_faces, 3, 3)
    """
    F = faces.shape[-1]
    B, V, C = vertices.shape

    vertices = einops.repeat(vertices, 'b m n -> b m k n', k=F)
    faces = einops.repeat(faces, 'm n -> b m n k', k=C, b=B)
    triangles = torch.gather(vertices, 1, faces.long())

    return triangles

def get_shape_matrix(x):
    if len(x.shape) == 3:
        return torch.stack([x[:, 0] - x[:, 2], x[:, 1] - x[:, 2]], dim=-1)

    elif len(x.shape) == 4:
        return torch.stack([x[:, :, 0] - x[:, :, 2], x[:, :, 1] - x[:, :, 2]], dim=-1)

    raise NotImplementedError

def edges_3d_to_2d(edges):
    """
    :param edges: Edges in 3D space (in the world coordinate basis) (E, 2, 3)
    :return: Edges in 2D space (in the intrinsic orthonormal basis) (E, 2, 2)
    """
    # Decompose for readability
    device = edges.device

    edges0 = edges[:, 0]
    edges1 = edges[:, 1]

    # Get orthonormal basis
    basis2d_0 = (edges0 / torch.norm(edges0, dim=-1).unsqueeze(-1))
    n = torch.cross(basis2d_0, edges1, dim=-1)
    basis2d_1 = torch.cross(n, edges0, dim=-1)
    basis2d_1 = basis2d_1 / torch.norm(basis2d_1, dim=-1).unsqueeze(-1)

    # Project original edges into orthonormal basis
    edges2d = torch.zeros((edges.shape[0], edges.shape[1], 2)).to(device=device)
    edges2d[:, 0, 0] = (edges0 * basis2d_0).sum(-1)
    edges2d[:, 0, 1] = (edges0 * basis2d_1).sum(-1)
    edges2d[:, 1, 0] = (edges1 * basis2d_0).sum(-1)
    edges2d[:, 1, 1] = (edges1 * basis2d_1).sum(-1)

    return edges2d

def make_einops_str(ndims, insert_k=None):
    linds = ['l', 'm', 'n', 'o', 'p']

    if insert_k is None:
        symbols = linds[:ndims]
    else:
        symbols = linds[:insert_k]
        symbols.append('k')
        symbols += linds[insert_k:ndims]

    out_str = ' '.join(symbols)
    return out_str

def make_repeat_str(tensor, dim):
    ndims = len(tensor.shape)

    out_str = []
    out_str.append(make_einops_str(ndims))
    out_str.append('->')
    out_str.append(make_einops_str(ndims, insert_k=dim))

    out_str = ' '.join(out_str)

    return out_str

def gather(data: torch.Tensor, index: torch.LongTensor, dim_gather: int, dim_data: int, dim_index: int):
    input_repeat_str = make_repeat_str(data, dim_data)
    index_repeat_str = make_repeat_str(index, dim_index + 1)

    data_repeat = einops.repeat(data, input_repeat_str, k=index.shape[dim_index])
    index_repeat = einops.repeat(index, index_repeat_str, k=data.shape[dim_data])

    out = torch.gather(data_repeat, dim_gather, index_repeat)

    return out

def deformation_gradient(triangles, Dm_inv):
    Ds = get_shape_matrix(triangles)

    return Ds @ Dm_inv
    
def get_shape_matrix(x):
    if len(x.shape) == 3:
        return torch.stack([x[:, 0] - x[:, 2], x[:, 1] - x[:, 2]], dim=-1)

    elif len(x.shape) == 4:
        return torch.stack([x[:, :, 0] - x[:, :, 2], x[:, :, 1] - x[:, :, 2]], dim=-1)

    raise NotImplementedError

def green_strain_tensor(F):
    device = F.device
    I = torch.eye(2, dtype=F.dtype).to(device)

    Ft = F.permute(0, 2, 1)
    return 0.5 * (Ft @ F - I)

def make_pervertex_tensor_from_lens(lens, val_tensor):
    val_list = []
    for i, n in enumerate(lens):
        val_list.append(val_tensor[i].repeat(n).unsqueeze(-1))
    val_stack = torch.cat(val_list)
    return val_stack

def compute_edge_length(vertices, _edge):
    e_len = torch.norm(vertices[_edge[:,0]] - vertices[_edge[:,1]], dim=-1)
    return e_len

def create_stack(triangles_list, param):
    lens = [x.shape[0] for x in triangles_list]
    stack = make_pervertex_tensor_from_lens(lens, param)[:, 0]
    return stack

def init_virtual_edge(v, f, dot_product_t=-0.7):
    '''
    virtual edge: connects face centers on opposite sides
    '''
    EPSILON = 1e-6
    device = v.device

    # face normals
    fn = FaceNormals(v.unsqueeze(0), f.unsqueeze(0))[0].cpu().numpy()
    # face centers
    fc = v[f].mean(1).cpu().numpy()
    ray_o = fc - fn*EPSILON

    v, f = v.cpu().numpy(), f.cpu().numpy()
    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh(o3d.core.Device("CPU:0"))
    mesh.vertex.positions = o3d.core.Tensor(v, o3d.core.float32, o3d.core.Device("CPU:0"))
    mesh.triangle.indices = o3d.core.Tensor(f, o3d.core.int32, o3d.core.Device("CPU:0"))
    scene.add_triangles(mesh)

    rays = o3d.core.Tensor(np.concatenate([ray_o, -fn], axis=-1))

    # Compute the ray intersections.
    ans = scene.cast_rays(rays)

    f_id = torch.tensor(np.array(ans['primitive_ids'].numpy(), dtype=np.int32), device=device)
    dist = torch.tensor(np.array(ans['t_hit'].numpy(), dtype=np.float32), device=device)

    mask = ((fn * fn[f_id.cpu()]).sum(-1) < dot_product_t) * (np.array(f_id.cpu()) > -1)
    edge_id = torch.cat([torch.arange(len(f), device=device)[...,None], f_id[...,None]], dim=-1)

    return edge_id[mask], dist[mask]

def get_ve_len(face_pair, v, f):
    fc = v[f].mean(1)
    pairs = fc[face_pair]
    curr_ve_dist = (pairs[:,0]-pairs[:,1]).norm(dim=-1)
    return curr_ve_dist

# def barycentric_3D(triangle: torch.tensor, point: torch.tensor):
#         '''
#         For a given point, compute its barycentric coordinate within triangle
#         Input:
#             triangle: torch.tensor size(3,3)
#             point: torch.tensor size(3,)
#         Output:
#             alpha: float
#             beta: float
#             gamma: float
#             N: torch.tensor size(3,)
#         '''
#         # Define triangle vertices
#         A, B, C = triangle
#         # Compute two edge vectors
#         AB = B - A
#         AC = C - A
#         # Compute the normal vector of the triangle
#         N = torch.cross(AB, AC)
#         # Find the area of the triangle
#         area = torch.norm(N)
#         N = N / area # normalized normal vector

#         # Fine the projection point
#         AX = point - A
#         n = torch.dot(AX, N)
#         P = point - n*N

#         # Calculate the barycentric coordinates
#         alpha = torch.dot(torch.cross(B - P, C - P), N) / area
#         beta = torch.dot(torch.cross(C - P, A - P), N) / area
#         gamma = 1 - alpha - beta

#         return alpha, beta, gamma, n*N

def barycentric_2D(triangles, points):
        '''
        For a given point, compute its barycentric coordinate within triangle
        Input:
            triangle: torch.tensor size(N, 3, 2)
            point: torch.tensor size(N, 3)
        Output:
            alpha: float
            beta: float
            gamma: float
            N: torch.tensor size(3,)
        '''
        triangles = F.pad(triangles, (0,1))
        points = F.pad(points, (0,1))
        # Define triangle vertices
        A, B, C = triangles.permute(1,0,2)
        P = points
        # Compute two edge vectors
        AB = B - A
        AC = C - A
        # Compute the normal vector of the triangle
        N = torch.cross(AB, AC)
        # Find the area of the triangle
        area = torch.norm(N, dim=-1)
        N = N / area[:,None] # normalized normal vector

        # Calculate the barycentric coordinates
        alpha = (torch.cross(B - P, C - P) * N).sum(-1) / area
        beta = (torch.cross(C - P, A - P) * N).sum(-1) / area
        gamma = 1 - alpha - beta

        return alpha, beta, gamma
