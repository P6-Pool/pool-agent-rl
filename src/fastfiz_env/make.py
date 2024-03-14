import gymnasium as gym
from typing import Optional
from .utils import RewardFunction


def make(env_id, reward_function: RewardFunction, num_balls: Optional[int] = 15,  max_episode_steps: Optional[int] = 100,
         disable_env_checker: Optional[bool] = True, **kwargs) -> gym.Env:
    """
    Create an instance of the specified environment.

    Args:
        env_id (str): The environment id.
        num_balls (int, optional): The number of balls in the environment. Defaults to 15.
        max_episode_steps (int, optional): The maximum number of steps in an episode. Defaults to 100.
        disable_env_checker (bool, optional): Whether to disable the environment checker. Defaults to True.
        **kwargs: Additional keyword arguments to pass to the environment.

    Returns:
        gym.Env: The environment instance.
    """
    return gym.make(env_id, reward_function=reward_function, num_balls=num_balls, max_episode_steps=max_episode_steps, disable_env_checker=disable_env_checker, **kwargs)
