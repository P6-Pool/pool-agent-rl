import gymnasium as gym
from environment import FastFizEnv
from gymnasium.envs.registration import register
from gymnasium.vector import VectorEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MultiInputPolicy, MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils import ModelManager
from environment import register_env
from typing import Optional
import json
import os


def read_config() -> dict:
    with open("src/config.json", "r") as fp:
        return json.load(fp)["train"]


def setup_config() -> dict:
    config = read_config()
    ent_coef = config["ent_coef"] if "ent_coef" in config else 0.01
    n_epochs = config["n_epochs"] if "n_epochs" in config else 10
    clip_range = config["clip_range"] if "clip_range" in config else 0.2
    n_steps = config["n_steps"] if "n_steps" in config else 2048
    batch_size = config["batch_size"] if "batch_size" in config else 64
    return {
        "ent_coef": ent_coef,
        "n_epochs": n_epochs,
        "clip_range": clip_range,
        "n_steps": n_steps,
        "batch_size": batch_size
    }


def create_model(env: gym.Env, log_dir: str) -> PPO:
    config = setup_config()
    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        tensorboard_log=log_dir,
        ent_coef=config["ent_coef"],
        n_epochs=config["n_epochs"],
        clip_range=config["clip_range"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=0.90,
        vf_coef=0.6,
    )

    return model


def train(
    timesteps=10_000,
    episodes=10,
    model: Optional[str] = None,
    version: Optional[int] = None,
    model_manager: Optional[ModelManager] = None,
) -> None:

    print(
        f'Training mode with {timesteps} timestep{"s" if timesteps != 1 else ""}, {episodes} episode{"s" if episodes != 1 else ""}'
    )

    try:
        for episode in range(episodes):
            model.learn(
                total_timesteps=timesteps,
                reset_num_timesteps=False,
                tb_log_name=str(model_manager.name_handler),
            )

            model.save(
                f"{model_manager.model_path}-{timesteps * (episode + 1)}")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model.")
        model.save(f"{model_manager.model_path}-{timesteps * (episode + 1)}")
        print(
            f"Model saved: {model_manager.model_path}-{timesteps * (episode + 1)}")
        return
    else:
        print("Training complete.")
        print(
            f"Model saved: {model_manager.model_path}-{timesteps * (episode + 1)}")
        return
