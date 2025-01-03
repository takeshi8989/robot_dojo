import argparse
import numpy as np
import genesis as gs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu, seed=0, precision='32', logging_level='debug')

    ########################## create a scene ##########################

    dt = 5e-4
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=10,
            gravity=(0, 0, 0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0, 0.8),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        mpm_options=gs.options.MPMOptions(
            dt=dt,
            lower_bound=(-1.0, -1.0, -0.2),
            upper_bound=(1.0,  1.0,  1.0),
        ),
        fem_options=gs.options.FEMOptions(
            dt=dt,
            damping=45.,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
    )

    ########################## entities ##########################
    scene.add_entity(morph=gs.morphs.Plane())

    E, nu = 3.e4, 0.45
    rho = 1000.

    robot_mpm = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.5, 0.2, 0.3),
            radius=0.1,
        ),
        material=gs.materials.MPM.Muscle(
            E=E,
            nu=nu,
            rho=rho,
            model='neohooken',
        ),
    )

    robot_fem = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.5, -0.2, 0.3),
            radius=0.1,
        ),
        material=gs.materials.FEM.Muscle(
            E=E,
            nu=nu,
            rho=rho,
            model='stable_neohooken',
        ),
    )

    ########################## build ##########################
    scene.build()

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, args.vis, robot_mpm, robot_fem))
    if args.vis:
        scene.viewer.start()


def run_sim(scene, enable_vis, robot_mpm, robot_fem):
    scene.reset()
    for i in range(1000):
        actu = np.array([0.2 * (0.5 + np.sin(0.01 * np.pi * i))])

        robot_mpm.set_actuation(actu)
        robot_fem.set_actuation(actu)
        scene.step()

    if enable_vis:
        scene.viewer.stop()


if __name__ == "__main__":
    main()
