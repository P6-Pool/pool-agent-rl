from gymnasium.envs.registration import EnvSpec
import gymnasium as gym

from fastfiz_env.wrappers.action import ActionSpaces, FastFizActionWrapper
from .reward_functions import RewardFunction, DefaultReward
from stable_baselines3.common.env_util import make_vec_env


def make(
    env_id: str | EnvSpec,
    *,
    reward_function: RewardFunction = DefaultReward,
    num_balls: int = 16,
    max_episode_steps: int = 100,
    disable_env_checker: bool = True,
    **kwargs
) -> gym.Env:
    """
    Create an instance of the specified environment.

    Args:
        env_id (str): The environment id.
        reward_function (RewardFunction): The reward function to use in the environment.
        num_balls (int, optional): The number of balls in the environment. Defaults to 16.
        max_episode_steps (int, optional): The maximum number of steps in an episode. Defaults to 100.
        disable_env_checker (bool, optional): Whether to disable the environment checker. Defaults to True.
        **kwargs: Additional keyword arguments to pass to the environment constructor.

    Returns:
        gym.Env: The environment instance.
    """
    return gym.make(
        env_id,
        reward_function=reward_function,
        num_balls=num_balls,
        max_episode_steps=max_episode_steps,
        disable_env_checker=disable_env_checker,
        **kwargs,
    )


def make_wrapped_env(
    env_id: str, num_balls: int, max_episode_steps: int, reward_function: RewardFunction
):
    env = make(
        env_id,
        reward_function=reward_function,
        num_balls=num_balls,
        max_episode_steps=max_episode_steps,
        disable_env_checker=False,
    )
    env = FastFizActionWrapper(env, action_space_id=ActionSpaces.NO_OFFSET_3D)
    return env


def make_wrapped_vec_env(
    env_id: str,
    num_balls: int,
    max_episode_steps: int,
    n_envs: int,
    reward_function: RewardFunction,
):

    def make_env():
        return make_wrapped_env(env_id, num_balls, max_episode_steps, reward_function)

    env = make_vec_env(
        make_env,
        n_envs=n_envs,
    )
    return env
