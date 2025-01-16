import os
import socket
import numpy as np
import open3d as o3d
import pymeshlab
import trimesh
import pyacvd
import pyvista as pv
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

# Add the parent directory of the project root to the Python path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

if not "ait-server" in socket.gethostname():
    from src.common.parse_scan import parse_scan

def remove_seperated_face(mesh):
    triangle_clusters, cluster_n_triangles, _ = (mesh.cluster_connected_triangles())
    largest_cluster_idx = np.asarray(cluster_n_triangles).argmax()
    triangles_to_remove = np.asarray(triangle_clusters) != largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    return mesh

def post_process(_path, fg_label, edge_len=0.01, vis=False):
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
    remove_seperated_face(mesh)
    if vis: o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

    # Save poisson mesh
    o3d.io.write_triangle_mesh(os.path.join(_path, "poisson.obj"), mesh)

    print('[Post Process] Parsing')
    tmesh = trimesh.Trimesh(vertices=np.array(mesh.vertices), faces=np.array(mesh.triangles), process=True)
    tmesh, valid_triangles = parse_scan(tmesh, _path, fg_label)
    tmesh.export(os.path.join(_path, "segmentated.obj"))
    mesh.remove_triangles_by_mask(~valid_triangles)
    mesh.remove_unreferenced_vertices()

    # filter seperated parts
    remove_seperated_face(mesh)
    o3d.io.write_triangle_mesh(os.path.join(_path, "parser.obj"), mesh)
    if vis: o3d.visualization.draw_geometries([o3d.io.read_triangle_mesh(os.path.join(_path, "segmentated.obj"))])
    if vis: o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # smooth surface remeshing
    mesh = pv.read(os.path.join(_path, "parser.obj"))
    if vis: mesh.plot(show_edges=True, color='w')
    clus = pyacvd.Clustering(mesh)
    clus.cluster(8000)
    remesh = clus.create_mesh()
    remesh.save(os.path.join(_path, "remesh.obj"))
    if vis: remesh.plot(color='w', show_edges=True)

    print('[Post Process] Done')

    
