# from igl import read_triangle_mesh
import numpy as np
import taichi as ti
import utils.pbd.geom2d as geom2d
import utils.pbd.mathlib as mathlib


@ti.data_oriented
class TrianMesh:

  def __init__(
      self,
      verts: np.ndarray,
      verts_ref: np.ndarray,
      faces: np.ndarray,
      dim=2,
      rho=1.0,
      get_edge=True,  # face edges
      get_edgeside=True,  # side vertex indices of edges
      get_edgeNeib=True,  # neighbor face indices of edges
      get_faceedge=True,  # edge infices of faces
      scale=1.0,
      repose=(0.0, 0.0, 0.0)):
    self.dim = dim

    n_vert = verts.shape[0]
    n_face = faces.shape[0] // 3
    if dim == 2 and verts.shape[1] == 3:
      verts = mathlib.np_3to2(verts)
      verts_ref = mathlib.np_3to2(verts_ref)
    edge_indices, edge_sides, edge_neib, face_edges = geom2d.edge_extractor(
        faces)
    assert faces.ndim == 1
    n_edge = edge_indices.shape[0] // 2

    verts = verts * scale
    verts_ref = verts_ref * scale
    for i in range(self.dim):
      verts[:, i] += repose[i]
      verts_ref[:, i] += repose[i]

    self.n_vert = n_vert
    self.n_edge = n_edge
    self.n_face = n_face

    self.v_p = ti.Vector.field(dim, dtype=ti.f32, shape=n_vert)
    self.v_p.from_numpy(verts)
    self.v_p_ref = ti.Vector.field(dim, dtype=ti.f32, shape=n_vert)
    self.v_p_ref.from_numpy(verts_ref)
    self.v_p_delta = ti.Vector.field(dim, dtype=ti.f32, shape=n_vert)
    self.f_i = ti.field(dtype=ti.i32, shape=n_face * 3)
    self.f_i.from_numpy(faces.flatten())

    self.verts_np = verts
    self.faces_np = faces

    if get_edge:
      self.e_i = ti.field(dtype=ti.i32, shape=n_edge * 2)
      self.e_i.from_numpy(edge_indices.flatten())
    if get_edgeside:
      self.e_sidei = ti.field(dtype=ti.i32, shape=n_edge * 2)
      self.e_sidei.from_numpy(edge_sides.flatten())
    if get_edgeNeib:
      self.e_neibi = ti.field(dtype=ti.i32, shape=n_edge * 2)
      self.e_neibi.from_numpy(edge_neib.flatten())
    if get_faceedge:
      self.f_edgei = ti.field(dtype=ti.i32, shape=n_face * 3)
      self.f_edgei.from_numpy(face_edges)

    self.compute_mass(rho)

  def set_texture_uv(self, uvs: np.ndarray):
    self.uvs = uvs

  def compute_mass(self, rho: float):
    self.v_invm = ti.field(dtype=ti.f32, shape=self.n_vert)
    self.f_mass = ti.field(dtype=ti.f32, shape=self.n_face)
    if self.dim == 3:
      self.get_mass_3d(rho)
    elif self.dim == 2:
      self.get_mass_2d(rho)

  @ti.kernel
  def get_mass_2d(self, rho: ti.f32):
    for k in range(self.n_face):
      p1 = self.f_i[k * 3]
      p2 = self.f_i[k * 3 + 1]
      p3 = self.f_i[k * 3 + 2]
      x1 = self.v_p[p1]
      x2 = self.v_p[p2]
      x3 = self.v_p[p3]
      self.f_mass[k] = rho * 0.5 * ti.abs((x2 - x1).cross(x3 - x1))
      self.v_invm[p1] += self.f_mass[k] / 3.0
      self.v_invm[p2] += self.f_mass[k] / 3.0
      self.v_invm[p3] += self.f_mass[k] / 3.0
    for k in range(self.n_vert):
      self.v_invm[k] = 1.0 / self.v_invm[k]

  @ti.kernel
  def get_mass_3d(self, rho: ti.f32):
    for k in range(self.n_face):
      p1 = self.f_i[k * 3]
      p2 = self.f_i[k * 3 + 1]
      p3 = self.f_i[k * 3 + 2]
      x1 = self.v_p[p1]
      x2 = self.v_p[p2]
      x3 = self.v_p[p3]
      self.f_mass[k] = rho * 0.5 * ((x2 - x1).cross(x3 - x1)).norm()
      self.v_invm[p1] += self.f_mass[k] / 3.0
      self.v_invm[p2] += self.f_mass[k] / 3.0
      self.v_invm[p3] += self.f_mass[k] / 3.0
    for k in range(self.n_vert):
      self.v_invm[k] = 1.0 / self.v_invm[k]

  @ti.kernel
  def get_pos_by_index(self, n: ti.i32, index: ti.template(),
                       pos: ti.template()):
    for k in range(n):
      pos[k] = self.v_p_ref[index[k]]

  @ti.kernel
  def set_pos_by_index(self, n: ti.i32, index: ti.template(),
                       pos: ti.template()):
    for k in range(n):
      self.v_p[index[k]] = pos[k]

  @ti.kernel
  def set_fixed_point(self, n: ti.i32, index: ti.template()):
    for k in range(n):
      self.v_invm[index[k]] = 0.0

  def get_render_draw(self, color=(0.5, 0.5, 0.5), wireframe=False):

    def render_draw(scene: ti.ui.Scene):
      scene.mesh(self.v_p, self.f_i, color=color, show_wireframe=wireframe)

    return render_draw
