import os
from stable_baselines3 import PPO


def save_model(model, model_name):
    current_dir = os.path.dirname(__file__)
    model_save_path = os.path.join(current_dir, "..", "models", model_name + ".zip")
    model.save(model_save_path)
    print(f"Model saved at: {model_save_path}")


def load_model(model_name):
    current_dir = os.path.dirname(__file__)
    model_save_path = os.path.join(current_dir, "..", "models", model_name + ".zip")
    model_type = model_name.split("_")[-1]
    if model_type == "ppo":
        model = PPO.load(model_save_path)
    print(f"Model loaded from: {model_save_path}")
    return model
