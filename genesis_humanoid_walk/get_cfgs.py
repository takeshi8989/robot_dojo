import os
import pickle

exp_name = "humanoid_walking_vx"
current_dir = os.path.dirname(__file__)
log_dir = os.path.join(current_dir, f"logs/{exp_name}")
env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))

# print(env_cfg)
print(reward_cfg)
