import gymnasium as gym
import utils.helper as helper
from env import CircularSwimmerEnv

model = helper.load_model("circular_swimmer_ppo")

base_env = gym.make("Swimmer-v5", render_mode="human")
env = CircularSwimmerEnv(base_env)

obs = env.reset()[0]
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    print(info)
    obs = obs if not done else env.reset()[0]

env.close()
