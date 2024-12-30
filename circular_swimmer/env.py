from gymnasium import Wrapper


class CircularSwimmerEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.target_radius = 3.0
        self.penalty_factor = 1.0

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        distance_from_origin = info.get("distance_from_origin")
        reward_ctrl = info.get("reward_ctrl")

        reward_radius = -self.penalty_factor * abs(distance_from_origin - self.target_radius)
        x_velocity, y_velocity = info.get("x_velocity"), info.get("y_velocity")
        reward_velocity = x_velocity ** 2 + y_velocity ** 2

        custom_reward = 1e-4 * reward_ctrl + 1.0 * reward_radius + 0.2 * reward_velocity

        info.update({
            "reward_radius": reward_radius,
            "reward_velocity": reward_velocity
        })

        return obs, custom_reward, done, truncated, info
