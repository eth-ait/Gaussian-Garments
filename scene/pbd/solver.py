import taichi as ti
from scene.pbd import framework, length, bend


def solve(mesh, velocity, substep=600, solve_iters=1):
    dt = 1.0 / substep

    xpbd = framework.pbd_framework(n_vert=mesh.n_vert, v_p=mesh.v_p, dt=dt)
    length_cons = length.LengthCons(mesh.v_p,
                                    mesh.v_p_ref,
                                    mesh.e_i,
                                    mesh.v_invm,
                                    dt=dt,
                                    alpha=0.0)
    bend_cons = bend.Bend3D(mesh.v_p,
                            mesh.v_p_ref,
                            mesh.e_i,
                            mesh.e_sidei,
                            mesh.v_invm,
                            dt=dt,
                            alpha=200.0)
    xpbd.add_cons(length_cons)
    xpbd.add_cons(bend_cons)
    xpbd.init_rest_status()
    xpbd.v_v = velocity

    for sub in range(substep):
        # init XPBD solver
        xpbd.make_prediction()

        # solve constraints
        xpbd.preupdate_cons()
        for _ in range(solve_iters):
            xpbd.update_cons()
        # xpbd.update_vel()
    return xpbd.v_p.to_numpy()
                
