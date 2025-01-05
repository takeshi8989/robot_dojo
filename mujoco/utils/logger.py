import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if "episode" in self.locals.get("infos", [{}])[0]:
            self.episode_rewards.append(self.locals["infos"][0]["episode"]["r"])
            self.episode_lengths.append(self.locals["infos"][0]["episode"]["l"])
        return True

    def _on_training_end(self) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards, label="Episode Rewards")
        plt.title("Training Progress: Episode Rewards", fontsize=14)
        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.show()
