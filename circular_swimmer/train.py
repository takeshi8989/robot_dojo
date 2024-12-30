import gymnasium as gym
from stable_baselines3 import PPO
from utils.logger import RewardLoggerCallback
from utils.helper import save_model
from env import CircularSwimmerEnv

base_env = gym.make("Swimmer-v5")
env = CircularSwimmerEnv(base_env)

model = PPO("MlpPolicy", env, verbose=1)

reward_logger = RewardLoggerCallback()
model.learn(total_timesteps=5_000_000, callback=reward_logger)

save_model(model, "circular_swimmer_ppo")

env = CircularSwimmerEnv(gym.make("Swimmer-v5", render_mode="human"))
obs = env.reset()[0]
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)

env.close()
