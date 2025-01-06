import argparse
import genesis as gs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, 0.0, 7),
            camera_lookat=(0.3, -0.6, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            dt=0.01
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(gs.morphs.Plane())
    import os
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, "model/g1.xml")
    robot = scene.add_entity(
        gs.morphs.MJCF(file=path)
    )

    ########################## build ##########################
    scene.build()

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, args.vis, robot))
    if args.vis:
        scene.viewer.start()


def run_sim(scene, enable_vis, robot):
    i = 0
    while True:
        i += 1
        scene.step()

        if i > 200:
            break

    if enable_vis:
        scene.viewer.stop()


if __name__ == "__main__":
    main()
