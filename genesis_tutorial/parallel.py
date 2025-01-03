import argparse
import torch
import numpy as np
import genesis as gs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        show_viewer=args.vis,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, -1.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
    )

    ########################## build ##########################

    # create 20 parallel environments
    B = 20
    scene.build(n_envs=B, env_spacing=(1.0, 1.0))
    
    jnt_names = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7',
        'finger_joint1',
        'finger_joint2',
    ]
    dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

    # control all the robots
    franka.control_dofs_position(
        torch.tile(
            torch.tensor([0, 0, 0, -1.0, 0, 0, 0, 0.02, 0.02], device=gs.device), (B, 1)
        ),
    )

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, args.vis, franka, dofs_idx))
    if args.vis:
        scene.viewer.start()


def run_sim(scene, enable_vis, franka, dofs_idx):
    for i in range(1250):
        if i == 0:
            franka.control_dofs_position(
                np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
                dofs_idx,
            )
        elif i == 250:
            franka.control_dofs_position(
                np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
                dofs_idx,
            )
        elif i == 500:
            franka.control_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                dofs_idx,
            )
        elif i == 750:
            # control first dof with velocity, and the rest with position
            franka.control_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
                dofs_idx[1:],
            )
            franka.control_dofs_velocity(
                np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
                dofs_idx[:1],
            )
        elif i == 1000:
            franka.control_dofs_force(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                dofs_idx,
            )
            
        scene.step()

    if enable_vis:
        scene.viewer.stop()


if __name__ == "__main__":
    main()
