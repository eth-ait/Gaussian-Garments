import os
import json
import glob
import numpy as np
from PIL import Image
import logging

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def colmap(source_path, cam_type="PINHOLE", use_gpu=1, skip_dense=False):
    # input data folder
    _images = os.path.join(source_path, "images")
    _masks = os.path.join(source_path, "masks")
    _cam_json = os.path.join(source_path, "cameras.json")
    _txt = os.path.join(source_path, "txt")
    _sparse = os.path.join(source_path, "sparse")
    _dense = os.path.join(source_path, "dense")

    os.makedirs(_txt, exist_ok=True)
    # new empty points3D.txt
    with open(os.path.join(_txt,"points3D.txt"), 'w'):
        pass

    # locate camera_list within image_path
    image_fns = sorted(os.listdir(_images))
    camera_list = [image_fn.split('.')[0] for image_fn in image_fns]
    camera_params = json.load(open(_cam_json, 'r'))

    for ID, camera_id in enumerate(camera_list):
        m = "w" if ID == 0 else "a"

        _path = os.path.join(_images, camera_id + '.png')
        image = Image.open(_path) if os.path.exists(_path) else None
        width, height = image.size

        # get camera intrinsic and extrinsic matrices
        intrinsic = np.asarray(camera_params[camera_id]["intrinsics"])
        extrinsic = np.asarray(camera_params[camera_id]["extrinsics"])

        R, T = extrinsic[:, :3], extrinsic[:, 3]
        fx, fy = intrinsic[0,0], intrinsic[1,1]
        px, py = intrinsic[:2,2]

        # write cameras.txt
        content = f"{ID} {cam_type} {width} {height} {fx} {fy} {px} {py}"
        with open(os.path.join(_txt,"cameras.txt"), m) as file:
            file.write(content+"\n")

        qvec = rotmat2qvec(R)
        tvec = T

        # write images.txt
        content = f"{ID} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} {ID} {camera_id + '.png'}"
        with open(os.path.join(_txt, "images.txt"), m) as file:
            file.write(content+"\n"+"\n")

    ## Feature extraction
    feat_extracton_cmd = "colmap feature_extractor " +\
                        f"--database_path {source_path}/database.db " +\
                        f"--image_path {_images} " +\
                        f"--ImageReader.mask_path {_masks} " +\
                        f"--SiftExtraction.use_gpu {use_gpu} "
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = "colmap exhaustive_matcher " +\
                       f"--database_path {source_path}/database.db " +\
                       f"--SiftMatching.use_gpu {use_gpu} " +\
                       f"--ExhaustiveMatching.block_size 200 "
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## point triangulate
    triangulate_cmd = "colmap point_triangulator " +\
                      f"--database_path {source_path}/database.db " +\
                      f"--image_path {_images} " +\
                      f"--input_path {_txt} " +\
                      f"--output_path {_sparse} "
    os.makedirs(_sparse, exist_ok=True)
    exit_code = os.system(triangulate_cmd)
    if exit_code != 0:
        logging.error(f"Point triangulation failed with code {exit_code}. Exiting.")
        exit(exit_code)
    if skip_dense: return 
    
    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    os.makedirs(_dense, exist_ok=True)
    img_undist_cmd = "colmap image_undistorter " +\
                     f"--image_path {_images} " +\
                     f"--input_path {_sparse} " +\
                     f"--output_path {_dense} "
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Image Undistortion failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## dense reconstruction
    dense_recon_cmd = "colmap patch_match_stereo " +\
                     f"--workspace_path {_dense} " +\
                     f"--PatchMatchStereo.depth_min 0.0 " +\
                     f"--PatchMatchStereo.depth_max 20.0"
    exit_code = os.system(dense_recon_cmd)
    if exit_code != 0:
        logging.error(f"Dense Reconstruction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## fuse dense point-cloud
    fuse_mesh_cmd = "colmap stereo_fusion " +\
                   f"--workspace_path {_dense} " +\
                   f"--output_path {_dense}/fused.ply "
    exit_code = os.system(fuse_mesh_cmd)
    if exit_code != 0:
        logging.error(f"Fuse Mesh failed with code {exit_code}. Exiting.")
        exit(exit_code)
