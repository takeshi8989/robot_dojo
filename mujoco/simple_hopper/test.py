import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
import utils.helper as helper

env = make_vec_env("Hopper-v5", n_envs=1)
model = helper.load_model("simple_hopper_ppo")
env = gym.make("Hopper-v5", render_mode="human")
obs = env.reset()[0]

done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    obs = obs if not done else env.reset()[0]

env.close()
