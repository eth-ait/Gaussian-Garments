import numpy as np
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud


def write_obj(dict, filename):

    with open(filename, 'w') as f:
        if "vertices" in dict:
            for vertex in dict['vertices']:
                f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
        if "uvs" in dict:
            for uv in dict['uvs']:
                f.write(f'vt {uv[0]} {uv[1]}\n')
        if "faces" in dict:
            if "texture_faces" in dict:
                for i, face in enumerate(dict['faces']):
                    face = face + 1
                    t_face = dict['texture_faces'][i] + 1
                    f.write(f'f {face[0]}/{t_face[0]} {face[1]}/{t_face[1]} {face[2]}/{t_face[2]}\n')
            else:
                for face in dict['faces']:
                    face = face + 1
                    f.write('f {} {} {}\n'.format(face[0], face[1], face[2]))


def read_obj(filename):
    # Read the OBJ file and store vertices and faces in a dictionary
    vertices = []
    uvs = []
    faces = []
    texture_faces = []
    have_uv = False
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            if parts[0] == 'v':
                vertex = tuple(map(float, parts[1:]))
                vertices.append(vertex)
            elif parts[0] == 'vt':
                have_uv = True
                uv = tuple(map(float, parts[1:]))
                uvs.append(uv)
            elif parts[0] == 'f':
                # faces for 3d points
                face = tuple(map(int, [p.split('/')[0] for p in parts[1:]]))
                faces.append(face)
                if have_uv:
                    # facs for uv
                    texture_face = tuple(map(int, [p.split('/')[1] for p in parts[1:]]))
                    texture_faces.append(texture_face)

    vertices = np.array(vertices, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32)
    faces = np.array(faces) - 1
    texture_faces = np.array(texture_faces) - 1
    obj_data = {'vertices': vertices, "uvs": uvs, 'faces': faces, 'texture_faces': texture_faces}
    return obj_data

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)