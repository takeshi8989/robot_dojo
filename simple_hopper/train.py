import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from utils.logger import RewardLoggerCallback
from utils.helper import save_model

env = make_vec_env("Hopper-v5", n_envs=1)
model = PPO("MlpPolicy", env, verbose=1)
reward_logger = RewardLoggerCallback()
model.learn(total_timesteps=1_000_000, callback=reward_logger)

save_model(model, "simple_hopper_ppo")

env = gym.make("Hopper-v5", render_mode="human")
obs = env.reset()[0]
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)

env.close()
