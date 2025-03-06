import taichi as ti
from pxr import Usd, UsdGeom
from scene.pbd import gmesh
import numpy as np
import time


class TaichiRenderer3D:

  def __init__(self,
               title: str,
               res,
               fps,
               cameraPos,
               cameraLookat,
               vertColor=False) -> None:
    self.window = ti.ui.Window(title, res)
    self.gui = self.window.get_gui()
    self.canvas = self.window.get_canvas()
    self.scene = ti.ui.Scene()
    self.camera = ti.ui.Camera()
    self.camera.position(cameraPos[0], cameraPos[1], cameraPos[2])
    self.camera.lookat(cameraLookat[0], cameraLookat[1], cameraLookat[2])

    self.frame = 0
    self.time = 0.0
    self.prev_time = time.time()
    self.fps = fps
    self.frame_dt = 1.0 / fps
    self.vertColor = vertColor
    self.keyboard_input = {}
    self.gui_list = []
    self.scene_render_list = []

    def print_camera_info():
      print("Camera position: ", self.camera.curr_position)
      print("Camera look at: ", self.camera.curr_lookat)

    self.add_click_event('p', print_camera_info)

  def add_click_event(self, key: str, func):
    self.keyboard_input[key] = func

  def add_gui_draw(self, gui_draw_call):
    self.gui_list.append(gui_draw_call)

  def clear_scene_rendeer_draw(self):
    self.scene_render_list.clear()

  def add_scene_render_draw(self, scene_render_draw_call):
    self.scene_render_list.append(scene_render_draw_call)

  def handle_input(self):
    if self.window.get_event(ti.ui.PRESS):
      if self.window.event.key in self.keyboard_input:
        self.keyboard_input[self.window.event.key]()

  def render(self):
    self.camera.track_user_inputs(self.window,
                                  movement_speed=0.01,
                                  hold_key=ti.ui.RMB)
    self.scene.set_camera(self.camera)
    self.scene.ambient_light((0.8, 0.8, 0.8))
    self.scene.point_light(pos=self.camera.curr_position, color=(1, 1, 1))
    self.scene.ambient_light([0.2, 0.2, 0.2])

    for scene_render_draw in self.scene_render_list:
      scene_render_draw(self.scene)

    self.canvas.scene(self.scene)

    if len(self.gui_list) > 0:
      with self.gui.sub_window('gui', 0.0, 0.0, 0.3, 0.4):
        for gui_draw in self.gui_list:
          gui_draw(self.gui)

    self.window.show()

    spent_time = time.time() - self.prev_time
    if spent_time < self.frame_dt:
      time.sleep(self.frame_dt - spent_time)
    self.prev_time = time.time()

    self.frame += 1
    self.time = self.frame / self.fps


class TaichiRender2D:

  def __init__(self, title: str, res, fps) -> None:
    self.window = ti.ui.Window(name=title, res=res)
    self.gui = self.window.get_gui()
    self.canvas = self.window.get_canvas()
    self.frame = 0
    self.time = 0.0
    self.prev_time = time.time()
    self.fps = fps
    self.frame_dt = 1.0 / fps
    self.keyboard_input = {}
    self.gui_list = []
    self.canvas_render_list = []

  def add_canvas_render_draw(self, canvas_render_draw_call):
    self.canvas_render_list.append(canvas_render_draw_call)

  def render(self):

    for canvas_render_draw in self.canvas_render_list:
      canvas_render_draw(self.canvas)

    if len(self.gui_list) > 0:
      with self.gui.sub_window('gui', 0.0, 0.0, 0.3, 0.4):
        for gui_draw in self.gui_list:
          gui_draw(self.gui)

    self.window.show()

    spent_time = time.time() - self.prev_time
    if spent_time < self.frame_dt:
      time.sleep(self.frame_dt - spent_time)
    self.prev_time = time.time()

    self.frame += 1
    self.time = self.frame / self.fps


class USDMeshRenderer:

  def __init__(self, filepath, totalframes, fps) -> None:
    self.stage = Usd.Stage.CreateNew(filepath)
    UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
    self.stage.SetStartTimeCode(1)
    self.stage.SetEndTimeCode(totalframes)
    self.stage.SetTimeCodesPerSecond(fps)

    self.totalframes = totalframes
    self.fps = fps
    self.frame = 0
    self.time = 0.0
    self.mesh_prims = []
    self.mesh_verts = []

    self.rootXform = UsdGeom.Xform.Define(self.stage, '/root')

  def render(self):
    if self.frame >= self.totalframes:
      return
    for i in range(len(self.mesh_prims)):
      meshGeom = UsdGeom.Mesh(self.stage.GetPrimAtPath(self.mesh_prims[i]))
      meshGeom.GetPointsAttr().Set(value=self.mesh_verts[i].to_numpy(),
                                   time=self.frame + 1)
    self.frame += 1
    self.time = self.frame / self.fps

  def add_ground(self, size):
    groundGeom = UsdGeom.Mesh.Define(self.stage, '/root/ground')
    groundPoints = np.array([[-size[0] / 2.0, 0.0, -size[1] / 2.0],
                             [-size[0] / 2.0, 0.0, size[1] / 2.0],
                             [size[0] / 2.0, 0.0, size[1] / 2.0],
                             [size[0] / 2.0, 0.0, -size[1] / 2.0]])
    groundGeom.GetPointsAttr().Set(groundPoints)
    groundGeom.GetFaceVertexIndicesAttr().Set(np.array([0, 1, 2, 0, 2, 3]))
    groundGeom.GetFaceVertexCountsAttr().Set(np.array([3, 3]))

  def add_dynamic_mesh(self,
                       vert: ti.MatrixField,
                       face: ti.MatrixField,
                       meshname='mesh'):
    primpath = '/root/' + meshname
    self.mesh_prims.append(primpath)
    self.mesh_verts.append(vert)
    self.meshGeom = UsdGeom.Mesh.Define(self.stage, primpath)
    self.meshGeom.GetPointsAttr().Set(vert.to_numpy())
    self.meshGeom.GetFaceVertexIndicesAttr().Set(face.to_numpy())
    self.meshGeom.GetFaceVertexCountsAttr().Set(vert.shape[0] * [3])
    self.meshGeom.GetSubdivisionSchemeAttr().Set('none')

  def save(self):
    self.stage.Save()


def get_visibility_switch(item):

  def visibility_switch():
    item.visible = not item.visible

  return visibility_switch


def get_wireframe_switch(item):

  def wireframe_switch():
    item.wireframe = not item.wireframe

  return wireframe_switch