import taichi as ti


@ti.data_oriented
class LengthCons:

  def __init__(
      self,
      v_p,  # vertex positions
      v_p_ref,  # vertex reference positions
      e_i,  # edge indices
      v_invm,  # vertex inverse mass
      dt,  # time step
      alpha=0.0  # inverse stiffness
  ) -> None:
    self.n = e_i.shape[0] // 2

    self.pos = v_p
    self.pos_ref = v_p_ref
    self.indices = e_i
    self.invm = v_invm

    self.rest_length = ti.field(dtype=ti.f32, shape=self.n)
    self.length = ti.field(dtype=ti.f32, shape=self.n)
    self.lambdaf = ti.field(dtype=ti.f32, shape=self.n)
    self.alpha = alpha / (dt * dt)

  @ti.kernel
  def compute_rest_length(self):
    for k in range(self.n):
      i = self.indices[k * 2]
      j = self.indices[k * 2 + 1]
      xi = self.pos_ref[i]
      xj = self.pos_ref[j]
      self.rest_length[k] = (xi - xj).norm()

  def init_rest_status(self):
    self.compute_rest_length()

  def preupdate_cons(self):
    self.lambdaf.fill(0.0)

  def update_cons(self):
    self.solve_cons()

  @ti.kernel
  def solve_cons(self):
    for k in range(self.n):
      i = self.indices[k * 2]
      j = self.indices[k * 2 + 1]
      xi = self.pos[i]
      xj = self.pos[j]
      xij = xi - xj
      self.length[k] = xij.norm()
      C = self.length[k] - self.rest_length[k]
      wi = self.invm[i]
      wj = self.invm[j]
      delta_lambda = -(C + self.alpha * self.lambdaf[k]) / (wi + wj +
                                                            self.alpha)
      self.lambdaf[k] += delta_lambda
      xij = xij / xij.norm()
      self.pos[i] += wi * delta_lambda * xij.normalized()
      self.pos[j] += -wj * delta_lambda * xij.normalized()
