import gymnasium as gym
from model_manager import ModelManager, ModelMetadata
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from typing import Union, Any, Optional
from dataclasses import dataclass
import os


@dataclass(init=False)
class PPOMetadata(ModelMetadata):
    timesteps: Optional[int]
    episodes: Optional[int]
    ep_reward_mean: Optional[float]
    ep_length_mean: Optional[float]
    n_balls_train: Optional[int]
    fastfizenv_version: Optional[str]

    def __init__(
        self,
        name: str | None = None,
        version: int | None = None,
        model_dir: str | None = None,
        logs_dir: str | None = None,
        *,
        timesteps: int | None = None,
        episodes: int | None = None,
        ep_reward_mean: float | None = None,
        ep_length_mean: float | None = None,
        n_balls_train: int | None = None,
        fastfizenv_version: str | None = None,
    ) -> None:
        super().__init__(name, version, model_dir, logs_dir)
        self.timesteps = timesteps
        self.episodes = episodes
        self.ep_reward_mean = ep_reward_mean
        self.ep_length_mean = ep_length_mean
        self.n_balls_train = n_balls_train
        self.fastfizenv_version = fastfizenv_version


class PPOManager(ModelManager):
    def __init__(
        self,
        base_name: str,
        model_dir: str = None,
        logs_dir: str = None,
    ) -> None:
        super().__init__(base_name, model_dir, logs_dir)
        self.metadata: PPOMetadata = PPOMetadata(
            self.name_handler.name,
            self.name_handler.version,
            self.model_dir,
            self.logs_dir,
        )

    def save_model(self, model: PPO) -> None:
        if os.path.exists(self.model_path):
            raise FileExistsError(f"Model file '{self.model_path}' already exists.")

        model.save(self.model_path)

    def load_model(self, env: VecEnv) -> PPO:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' does not exist.")

        model = PPO.load(self.model_path, env=env)
        return model

    def save_metadata(self, metadata: Optional[PPOMetadata] = None) -> None:
        return super().save_metadata(metadata)


if __name__ == "__main__":
    LOGS_DIR = "logs/"
    MODEL_DIR = "models/"

    pm = PPOManager("ppo", MODEL_DIR, LOGS_DIR)

    metadata = PPOMetadata(
        "ppo",
        1,
        timesteps=100,
        episodes=10,
        ep_reward_mean=10,
        ep_length_mean=10,
        n_balls_train=9,
        fastfizenv_version="v1",
    )

    pm.metadata = metadata

    print(pm.model_path)
    print(pm.metadata_path)
    pm.save_metadata()
