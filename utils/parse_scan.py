import os
import json
import tqdm
import torch
import trimesh
import cv2 as cv
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer


# TODO: remove reliance on these global variables
# set 4DDress RGB cameras
# camera_ids = ['0000', '0001', '0004', '0005', '0008', '0009', '0012', '0013', '0016', '0017',
#                '0020', '0021', '0024', '0025', '0028', '0029', '0032', '0033', '0036', '0037',
#                '0040', '0041', '0044', '0045', '0048', '0049', '0052', '0053', '0056', '0057',
#                '0060', '0061', '0064', '0065', '0068', '0069', '0072', '0073', '0076', '0077',
#                '0080', '0081', '0084', '0085', '0088', '0089', '0092', '0093']

def render_label_colors(label):
    # init white color as background
    colors = np.ones((label.shape[0], 3)) * 255
    # assign label color
    colors[label == 1] = [128, 128, 128]
    # append color with the fourth channel
    colors = np.append(colors, np.ones((colors.shape[0], 1)) * 255, axis=-1) / 255.
    return colors

# load capture camera parameters: intrinsics_ori, extrinsics_ori
def load_capture_cameras(camera_params, camera_list, image_shape):
    # init camera_dict
    camera_dict = dict()
    # process all camera within camera_list
    for camera_id in camera_list:
        # unpack camera intrinsic and extrinsic matrices
        intrinsic = torch.tensor(camera_params[camera_id]["intrinsics"], dtype=torch.float32).cuda()
        extrinsic = torch.tensor(camera_params[camera_id]["extrinsics"], dtype=torch.float32).cuda()
        # unpack camera image shape
        image_size = torch.tensor([image_shape[0], image_shape[1]], dtype=torch.float32).unsqueeze(0).cuda()

        # unpack camera parameters
        f_xy = torch.cat([intrinsic[0:1, 0], intrinsic[1:2, 1]], dim=0).unsqueeze(0)
        p_xy = intrinsic[:2, 2].unsqueeze(0)
        R = extrinsic[:, :3].unsqueeze(0)
        T = extrinsic[:, 3].unsqueeze(0)
        # coordinate system adaption to PyTorch3D
        R[:, :2, :] *= -1.0
        # camera position in world space -> world position in camera space
        T[:, :2] *= -1.0
        R = torch.transpose(R, 1, 2)  # row-major

        # assign Pytorch3d PerspectiveCameras
        camera_dict[camera_id] = PerspectiveCameras(focal_length=f_xy, principal_point=p_xy, R=R, T=T, in_ndc=False, image_size=image_size).cuda()

    # assign Pytorch3d RasterizationSettings
    raster_settings = RasterizationSettings(image_size=image_shape, blur_radius=0.0, faces_per_pixel=2, max_faces_per_bin=80000)
    return camera_dict, raster_settings


def parse_scan(scan_data, source_path, only_fg=True):
    images_path = source_path / 'images'
    camera_ids = [f.stem for f in images_path.glob('*.png')]

    camera_fn = os.path.join(source_path, 'cameras.json')
    multi_view_labels = dict()

    # load mesh ply
    th_verts = torch.tensor(scan_data.vertices, dtype=torch.float32).unsqueeze(0).cuda()
    th_faces = torch.tensor(scan_data.faces, dtype=torch.long).unsqueeze(0).cuda()
    scan_mesh = Meshes(th_verts, th_faces).cuda()

    for camera_id in camera_ids:
        label_image_path = os.path.join(source_path, 'masks', f'{camera_id}.png.png')
        label_image = cv.imread(label_image_path) / 255.0
        label_value = np.zeros(label_image.shape)
        label_value[label_image > 0.5] = 1
        multi_view_labels[camera_id] = label_value


    # # -------------------- Load Capture Cameras -------------------- # #
    # load camera_list and camera_params: 
    camera_list = camera_ids
    # image_shape, image_mode = (4096, 3008), 'origin'
    image_shape, image_mode = (1280, 940), 'resize'
    camera_params = json.load(open(os.path.join(camera_fn), 'r'))
    # load pytorch3d camera_agents and raster_settings
    camera_agents, raster_settings = load_capture_cameras(camera_params, camera_list, image_shape)


    # # -------------------- Load Capture Image and Camera -------------------- # #

    # init scan_votes (nvt, nl)
    scan_votes = torch.zeros((scan_data.vertices.shape[0], 2)).cuda()

    for nc in tqdm.tqdm(range(len(camera_ids))):
        camera_id = camera_ids[nc]
        label_value = torch.tensor(multi_view_labels[camera_id]).cuda()

        # get capture_rasts from camera and scan_mesh
        capture_rasts = MeshRasterizer(cameras=camera_agents[camera_id], raster_settings=raster_settings)(scan_mesh)
        # get pix_to_face (H, W)
        pix_to_face = capture_rasts.pix_to_face[0, :, :, 0].cuda()

        # append label votes
        for nl in range(2):
            label_points = torch.stack(torch.where(label_value == nl), dim=0).T
            label_faces = pix_to_face[label_points[:, 0], label_points[:, 1]]
            # project to label_vertices (np, 3)
            label_vertices = th_faces.squeeze(0)[label_faces[label_faces > -1]]
            for nf in range(label_vertices.shape[-1]):
                scan_votes[label_vertices[:, nf], nl] += 1


    # obtain scan_labels
    scan_labels = (torch.argmax(scan_votes, dim=-1)).cpu().numpy()
    scan_label_colors = render_label_colors(scan_labels)
    segmented = trimesh.Trimesh(vertices=scan_data.vertices,
                                faces=scan_data.faces,
                                vertex_colors=scan_label_colors)


    if only_fg:
        face_mask = (scan_labels[scan_data.faces] == 1).all(-1)
        return segmented, face_mask
    else:
        return scan_labels

# extract label meshes from scan_mesh
def extract_label_meshes(vertices, faces, labels, surface_labels, colors=None, uvs=None, uv_path=None):
    # init label_meshes and face_labels
    label_meshes = dict()
    face_labels = labels[faces]
    # loop over all labels
    for nl in range(len(surface_labels)):
        # skip empty label
        if np.sum(labels == nl) == 0: continue
        # find label faces: with label vertices == 3
        vertex_label_nl = np.where(labels == nl)[0]
        face_label_nl = np.where(np.sum(face_labels == nl, axis=-1) == 3)[0]
        # find correct indices
        correct_indices = (np.zeros(labels.shape[0]) - 1).astype(int)
        correct_indices[vertex_label_nl] = np.arange(vertex_label_nl.shape[0])
        # extract label_mesh[vertices, faces]
        label_meshes[surface_labels[nl]] = {'vertices': vertices[vertex_label_nl], 'faces': correct_indices[faces[face_label_nl]]}
        # extract label_mesh colors
        label_meshes[surface_labels[nl]]['colors'] = colors[vertex_label_nl] if colors is not None else None
        # extract label_mesh uvs
        label_meshes[surface_labels[nl]]['uvs'] = uvs[vertex_label_nl] if uvs is not None else None
        # save label_mesh uv_path
        label_meshes[surface_labels[nl]]['uv_path'] = uv_path if uv_path is not None else None
    return label_meshes
    
