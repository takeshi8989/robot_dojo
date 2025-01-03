import argparse
import numpy as np
import genesis as gs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init()

    ########################## create a scene ##########################

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=200,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=False,
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
    )
    ########################## build ##########################
    n_envs = 16
    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, args.vis, robot, n_envs))
    if args.vis:
        scene.viewer.start()


def run_sim(scene, enable_vis, robot, n_envs=16):
    target_quat = np.tile(np.array([0, 1, 0, 0]), [n_envs, 1])  # pointing downwards
    center = np.tile(np.array([0.4, -0.2, 0.25]), [n_envs, 1])
    angular_speed = np.random.uniform(-10, 10, n_envs)
    r = 0.1

    ee_link = robot.get_link('hand')

    for i in range(0, 1000):
        target_pos = np.zeros([n_envs, 3])
        target_pos[:, 0] = center[:, 0] + np.cos(i / 360 * np.pi * angular_speed) * r
        target_pos[:, 1] = center[:, 1] + np.sin(i / 360 * np.pi * angular_speed) * r
        target_pos[:, 2] = center[:, 2]
        target_q = np.hstack([target_pos, target_quat])

        q = robot.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
            rot_mask=[False, False, True],  # for demo purpose: only restrict direction of z-axis
        )

        robot.set_qpos(q)
        scene.step()

    if enable_vis:
        scene.viewer.stop()


if __name__ == "__main__":
    main()
