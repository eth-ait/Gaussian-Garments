import os
import json
import numpy as np
from PIL import Image
import logging

import socket
import open3d as o3d
import pymeshlab
import trimesh
import pyacvd
import pyvista as pv
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

from .parse_scan import parse_scan
# if not "ait-server" in socket.gethostname():
#     from src.common.parse_scan import parse_scan

def remove_seperated_face(mesh):
    triangle_clusters, cluster_n_triangles, _ = (mesh.cluster_connected_triangles())
    largest_cluster_idx = np.asarray(cluster_n_triangles).argmax()
    triangles_to_remove = np.asarray(triangle_clusters) != largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    return mesh

def COLMAP_recon(_path, use_gpu=True, skip_dense=False):
    """
    Runs the COLMAP pipeline on the given source path.
    It saves the final mesh as 'template.obj' in the source path.
    Args:  
        _path (str): The path to the source folder containing images, masks, and txt files.
        use_gpu (bool): Whether to use GPU for feature extraction and matching.
        skip_dense (bool): Whether to skip dense reconstruction.
    """

    # input and output data folders
    _images = os.path.join(_path, "images")
    _masks = os.path.join(_path, "masks")
    _txt = os.path.join(_path, "txt")
    _sparse = os.path.join(_path, "sparse")
    _dense = os.path.join(_path, "dense")

    ## Feature extraction
    feat_extracton_cmd = "colmap feature_extractor " +\
                        f"--database_path {_path}/database.db " +\
                        f"--image_path {_images} " +\
                        f"--ImageReader.mask_path {_masks} " +\
                        f"--SiftExtraction.use_gpu {use_gpu} "
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = "colmap exhaustive_matcher " +\
                       f"--database_path {_path}/database.db " +\
                       f"--SiftMatching.use_gpu {use_gpu} " +\
                       f"--ExhaustiveMatching.block_size 200 "
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## point triangulate
    triangulate_cmd = "colmap point_triangulator " +\
                      f"--database_path {_path}/database.db " +\
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
    img_undist_cmd = "colmap image_undistorter " +\
                     f"--image_path {_images} " +\
                     f"--input_path {_sparse} " +\
                     f"--output_path {_dense} "
    os.makedirs(_dense, exist_ok=True)
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

def post_process(_path, vis):
    """
    Post-process the dense reconstruction results by filtering out background and outliers, poisson surface reconstruction, parsing, and smoothing.
    Args:  
        _path (str): The path to the source folder containing images, masks, and txt files.
        vis (bool): Whether to visualize the results.
    """
    print('[Post Process] Loading dense recon result')
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.join(_path, "dense/fused.ply"))
    points = ms.mesh(0).vertex_matrix()
    normals = ms.mesh(0).vertex_normal_matrix()

    print('[Post Process] Filter background / Outlier')
    # Remove background
    pcd = o3d.io.read_point_cloud(os.path.join(_path, "dense/fused.ply"))
    fg_mask = (np.array(pcd.colors) != np.array([[0,255,0]])).all(-1)
    pcd.points = o3d.utility.Vector3dVector(points[fg_mask])
    pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors)[fg_mask])
    pcd.normals = o3d.utility.Vector3dVector(normals[fg_mask])

    # Remove outliers
    # cl, ind = pcd.remove_radius_outlier(nb_points=50, radius=0.05)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.005)
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.5)
    pcd = voxel_down_pcd.select_by_index(ind)
    ouliers = voxel_down_pcd.select_by_index(ind, invert=True)
    ouliers.paint_uniform_color([1, 0, 0])
    if vis: o3d.visualization.draw_geometries([pcd, ouliers])

    print('[Post Process] Store point cloud')
    o3d.io.write_point_cloud(os.path.join(_path, "point_cloud.ply"), pcd)
    if vis: o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    print('[Post Process] Poisson Surface Reconstruction')
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=13)
    mesh = remove_seperated_face(mesh)
    if vis: o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

    # Save poisson mesh
    o3d.io.write_triangle_mesh(os.path.join(_path, "poisson.obj"), mesh)

    print('[Post Process] Parsing')
    tmesh = trimesh.Trimesh(vertices=np.array(mesh.vertices), faces=np.array(mesh.triangles), process=True)
    tmesh, valid_triangles = parse_scan(tmesh, _path, fg_label)
    tmesh.export(os.path.join(_path, "segmented.obj"))
    mesh.remove_triangles_by_mask(~valid_triangles)
    mesh.remove_unreferenced_vertices()

    # filter seperated parts
    mesh = remove_seperated_face(mesh)
    o3d.io.write_triangle_mesh(os.path.join(_path, "parser.obj"), mesh)
    if vis: o3d.visualization.draw_geometries([o3d.io.read_triangle_mesh(os.path.join(_path, "segmented.obj"))])
    if vis: o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # smooth surface remeshing
    mesh = pv.read(os.path.join(_path, "parser.obj"))
    if vis: mesh.plot(show_edges=True, color='w')
    clus = pyacvd.Clustering(mesh)
    clus.cluster(8000)
    remesh = clus.create_mesh()
    remesh.save(os.path.join(_path, "template.obj"))
    if vis: remesh.plot(color='w', show_edges=True)

    print('[Post Process] Done')