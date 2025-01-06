import argparse
import os
import pickle

import torch
from hover_env import HoverEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("--ckpt", type=int, default=500)
    parser.add_argument("--record", action="store_true", default=False)
    args = parser.parse_args()

    gs.init()

    current_dir = os.path.dirname(__file__)
    log_dir = os.path.join(current_dir, "..", f"logs/{args.exp_name}")
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    # visualize the target
    env_cfg["visualize_target"] = True
    # for video recording
    env_cfg["visualize_camera"] = args.record
    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60

    env = HoverEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="mps")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="mps")

    env.setup_sim(policy)


if __name__ == "__main__":
    main()

"""
# evaluation
python genesis_tutorial/drone/hover_eval.py
"""
