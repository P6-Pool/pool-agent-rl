import gymnasium as gym
from gymnasium import register
from gymnasium.vector import VectorEnv
from .fastfizenv_v2 import FastFizEnv


# Register the environment if not already registered


def register_env(kwargs: dict) -> gym.Env:
    print(f"Registering environment with kwargs: {kwargs}")
    register(
        id="FastFiz-v3",
        entry_point="environment.fastfizenv_v3:FastFizEnv",
        max_episode_steps=200,
        kwargs=kwargs,
    )

    return gym.make("FastFiz-v3")


# if gym.registry.get("FastFizEnv-v1") is not None:
#     Env: gym.Env = gym.make("FastFizEnv-v1", render_mode="human")
