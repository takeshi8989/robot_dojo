import os
import argparse
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
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    # when loading an entity, you can specify its pose in the morph.
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, 'model/g1.xml')
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=path
        ),
    )

    ########################## build ##########################
    scene.build()

    jnt_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
    ]
    dofs_idx = [robot.get_joint(name).dof_idx_local for name in jnt_names]

    ############ Optional: set control gains ############
    # set positional gains
    robot.set_dofs_kp(
        kp=np.array([100] * 29),
        dofs_idx_local=dofs_idx,
    )
    # set velocity gains
    robot.set_dofs_kv(
        kv=np.array([45] * 29),
        dofs_idx_local=dofs_idx,
    )
    # set force range for safety
    # robot.set_dofs_force_range(
    #     lower=np.array([-100] * 29),
    #     upper=np.array([100] * 29),
    #     dofs_idx_local=dofs_idx,
    # )

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, args.vis, robot, dofs_idx))
    if args.vis:
        scene.viewer.start()


def run_sim(scene, enable_vis, robot, dofs_idx):
    # Hard reset
    # for i in range(150):
    #     if i < 50:
    #         franka.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
    #     elif i < 100:
    #         franka.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), dofs_idx)
    #     else:
    #         franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), dofs_idx)

    #     scene.step()

    # PD control
    for i in range(1250):
        if i == 0:
            robot.control_dofs_position(
                np.array([0] * 29),
                dofs_idx,
            )
        # elif i == 250:
        #     zeros = np.array([0] * 29)
        #     zeros[0] = 0.5
        #     zeros[6] = 0.5
        #     robot.control_dofs_position(
        #         zeros,
        #         dofs_idx,
        #     )
        # elif i == 500:
        #     zeros = np.array([0] * 29)
        #     zeros[14] = 0.7
        #     robot.control_dofs_position(
        #         zeros,
        #         dofs_idx,
        #     )
        # elif i == 750:
        #     # control first dof with velocity, and the rest with position
        #     robot.control_dofs_position(
        #         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
        #         dofs_idx[1:],
        #     )
        #     franka.control_dofs_velocity(
        #         np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
        #         dofs_idx[:1],
        #     )
        # elif i == 1000:
        #     franka.control_dofs_force(
        #         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        #         dofs_idx,
        #     )
        # This is the control force computed based on the given control command
        # If using force control, it's the same as the given control command
        print('control force:', robot.get_dofs_control_force(dofs_idx))

        # This is the actual force experienced by the dof
        print('internal force:', robot.get_dofs_force(dofs_idx))

        scene.step()

    if enable_vis:
        scene.viewer.stop()


if __name__ == "__main__":
    main()
