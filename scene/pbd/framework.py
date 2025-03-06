import taichi as ti

@ti.data_oriented
class pbd_framework:

  def __init__(self, n_vert, v_p, dt, damp=1.0) -> None:
    self.n_vert = n_vert
    self.v_p = v_p
    self.v_p_cache = ti.Vector.field(v_p.n, dtype=v_p.dtype, shape=n_vert)
    self.v_p_cache = ti.Vector.field(v_p.n, dtype=v_p.dtype, shape=n_vert)
    self.v_v = ti.Vector.field(v_p.n, dtype=v_p.dtype, shape=n_vert)
    self.dt = dt
    self.damp = ti.field(dtype=ti.f32, shape=())
    self.damp[None] = damp

    self.cons_list = [[]]
    self.initupdate = []
    self.preupdate = []
    self.collision = []

  @ti.kernel
  def make_prediction(self):
    ti.loop_config(serialize=True)
    for k in range(self.n_vert):
      self.v_p_cache[k] = self.v_p[k]
      self.v_p[k] = self.v_p[k] + self.v_v[k] * self.dt # + self.g * self.dt * self.dt

  # @ti.kernel
  # def update_vel(self):
  #   ti.loop_config(serialize=True)
  #   for k in range(self.n_vert):
  #     self.v_v[k] = self.damp[None] * (self.v_p[k] -
  #                                      self.v_p_cache[k]) / self.dt

  def add_cons(self, new_cons, layer_index=0):
    if layer_index > len(self.cons_list):
      print('wrong constraints index')
      return
    if layer_index == len(self.cons_list):
      self.cons_list.append([])
    self.cons_list[layer_index].append(new_cons)

  def add_preupdate(self, preupdate):
    self.preupdate.append(preupdate)

  def add_init(self, init):
    self.initupdate.append(init)

  def add_collision(self, obj):
    self.collision.append(obj)

  def init_rest_status(self, layer_index=0):
    for init in self.initupdate:
      init()
    for i in range(len(self.cons_list[layer_index])):
      self.cons_list[layer_index][i].init_rest_status()
    for coll in self.collision:
      coll(self.v_p)

  def preupdate_cons(self, index=0):
    for update in self.preupdate:
      update()
    for i in range(len(self.cons_list[index])):
      self.cons_list[index][i].preupdate_cons()
    for coll in self.collision:
      coll(self.v_p)

  def update_cons(self, layer_index=0):
    for i in range(len(self.cons_list[layer_index])):
      self.cons_list[layer_index][i].update_cons()

  @ti.kernel
  def get_kinematic_energy(self, invm: ti.template()) -> ti.f32:
    e = 0.0
    for i in range(self.n_vert):
      e += 0.5 * self.v_v[i].norm_sqr() / invm[i]
    return e
