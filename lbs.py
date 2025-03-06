import os
import smplx
import socket
import trimesh
import pickle
import glob
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import neighbors
from argparse import ArgumentParser

from utils.defaults import DEFAULTS

def batch_rodrigues(rot_vecs, epsilon = 1e-8):
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                    F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms
    
def smplx_pose2mesh(smplx_params, smplx_model, erase_transl=False):
    # get smpl_output from smpl model and smpl data
    smplx_output = smplx_model.forward(
        betas=torch.tensor(smplx_params['betas']).unsqueeze(0),
        expression=torch.tensor(smplx_params['expression']).unsqueeze(0),
        global_orient=torch.tensor(smplx_params['global_orient']).unsqueeze(0),
        body_pose=torch.tensor(smplx_params['body_pose']).unsqueeze(0),
        left_hand_pose=torch.tensor(smplx_params['left_hand_pose']).unsqueeze(0),
        right_hand_pose=torch.tensor(smplx_params['right_hand_pose']).unsqueeze(0),
        jaw_pose=torch.tensor(smplx_params['jaw_pose']).unsqueeze(0),
        leye_pose=torch.tensor(smplx_params['leye_pose']).unsqueeze(0),
        reye_pose=torch.tensor(smplx_params['reye_pose']).unsqueeze(0),
        transl=torch.tensor(smplx_params['transl']).unsqueeze(0)
    )
    vertices = smplx_output.vertices.squeeze(0)
    # erase translation
    if erase_transl: vertices -= smplx_params['transl']

    # set smplx_mesh from smplx_output or smplx_trimesh
    smplx_mesh = {'vertices': vertices.detach().cpu().numpy(), 'faces': smplx_model.faces}
    return smplx_mesh

def prepare_lbs(smplx_model, smplx_params, vertices, blend_weights=None, nn_ids=None, unpose=False):
    '''
        vertices should be at origin
    '''
    # convert format
    vertices = torch.tensor(vertices)
    global_orient = torch.tensor(smplx_params['global_orient']).unsqueeze(0)
    body_pose = torch.tensor(smplx_params['body_pose']).unsqueeze(0)
    betas = torch.tensor(smplx_params['betas']).unsqueeze(0)
    left_hand_pose = torch.tensor(smplx_params['left_hand_pose']).unsqueeze(0)
    right_hand_pose = torch.tensor(smplx_params['right_hand_pose']).unsqueeze(0)
    jaw_pose = torch.tensor(smplx_params['jaw_pose']).unsqueeze(0)
    leye_pose = torch.tensor(smplx_params['leye_pose']).unsqueeze(0)
    reye_pose = torch.tensor(smplx_params['reye_pose']).unsqueeze(0)
    expression = torch.tensor(smplx_params['expression']).unsqueeze(0)
    transl = torch.tensor(smplx_params['transl']).unsqueeze(0)
    left_hand_pose = torch.einsum('bi,ij->bj', [left_hand_pose, smplx_model.left_hand_components])
    right_hand_pose = torch.einsum('bi,ij->bj', [right_hand_pose, smplx_model.right_hand_components])
    # get full pose
    full_pose = torch.cat([global_orient.reshape(-1, 1, 3),
                            body_pose.reshape(-1, smplx_model.NUM_BODY_JOINTS, 3),
                            jaw_pose.reshape(-1, 1, 3),
                            leye_pose.reshape(-1, 1, 3),
                            reye_pose.reshape(-1, 1, 3),
                            left_hand_pose.reshape(-1, 15, 3),
                            right_hand_pose.reshape(-1, 15, 3)],
                            dim=1).reshape(-1, 165)
    full_pose += smplx_model.pose_mean
    batch_size = max(betas.shape[0], global_orient.shape[0], body_pose.shape[0])
    # Concatenate the shape and expression coefficients
    scale = int(batch_size / betas.shape[0])
    if scale > 1:
        betas = betas.expand(scale, -1)
    shape_components = torch.cat([betas, expression], dim=-1)
    shapedirs = torch.cat([smplx_model.shapedirs, smplx_model.expr_dirs], dim=-1)
    # position of body and joints without translation
    G, body_vertices, pose_offset = lbs(shape_components, 
                                        full_pose, 
                                        smplx_model.v_template,
                                        shapedirs, 
                                        smplx_model.posedirs,
                                        smplx_model.J_regressor,
                                        smplx_model.parents,
                                        smplx_model.lbs_weights
                                        )
    if unpose:
        # compute inverse rotation matrics
        inv_G = np.zeros_like(G)
        for i in range(G.shape[0]):
            inv_G[i] = np.linalg.inv(G[i])
        # inv_G([55, 4, 4])
        G = inv_G

    if blend_weights is None:
        # for each garment vertices, find nearest body point and set equal lbs_weights
        _, nn_list = neighbors.KDTree(body_vertices).query(vertices)
        nn_ids = nn_list[..., 0]
        assert unpose, "forward pose should provide weights"
        blend_weights = smplx_model.lbs_weights[nn_ids].numpy()
        # blend_weights /= np.sum(blend_weights, axis=-1, keepdims=True)
    G = torch.einsum('ab,bcd->acd', torch.tensor(blend_weights), torch.tensor(G))
    # G([14249, 4, 4])

    if not unpose: vertices += pose_offset[nn_ids]

    gV = torch.tensor(vertices, dtype=torch.float32)
    gV = torch.cat([gV, torch.ones((gV.shape[0], 1))], axis=-1)
    # gV([14249, 4])
    # skinning
    vertices = torch.einsum('abc,ac->ab', G, gV)[:,:3]

    if unpose: vertices -= pose_offset[nn_ids]

    return np.array(vertices), blend_weights, nn_ids

    
def lbs(betas, pose, v_template, shapedirs, posedirs,
        J_regressor, parents, lbs_weights, with_body=True):
    '''
    Parameters
    ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional
        
    Return
    ----------
      A[0]: (55, 4, 4) --> rotation matrices for each joint
      v[0]: (N, 3) --> body position without translation
      v_offest[0]: (N, 3) --> body pose offset
    '''
    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype
    # Add shape contribution
    v_shaped = v_template + torch.einsum('bl,mkl->bmk', [betas, shapedirs])
    # Get the joints
    # NxJx3 array
    J = torch.einsum('bik,ji->bjk', [v_shaped, J_regressor])
    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    # get rotation matrix
    rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
    # (N x P) x (P, V * 3) -> N x V x 3
    pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    # add pose blend shapes
    v_posed = pose_offsets + v_shaped

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                            dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    v = v_homo[:, :, :3, 0]
    return A[0], v[0], pose_offsets[0]


def load_4DDress_smplx(source_path, tmp_path=None):
    if tmp_path is None: tmp_path = source_path
    # locate template pose (non_cano pose)
    colmap_path = glob.glob(os.path.join("../datas", "_".join(tmp_path.split('/')[-3:-1])+"_Take*"))[0]
    tmp_smplx = sorted(glob.glob(os.path.join('/', *tmp_path.split('/')[:-1], colmap_path.split("_")[-1], "Meshes/smplx/*.pkl")))[0]
    tmp_pose = pickle.load(open(tmp_smplx, 'rb'))

    # load target pose sequence
    smplx_seq = sorted(glob.glob(os.path.join(source_path, "Meshes/smplx/*.pkl")))
    pose_list = []
    for _smplx in smplx_seq:
        pose = pickle.load(open(_smplx, 'rb'))
        # replace target beta with template beta
        pose['betas'] = tmp_pose['betas']
        pose_list.append(pose)
    return pose_list, tmp_pose

