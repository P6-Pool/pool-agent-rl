import numpy as np
import gymnasium as gym
import fastfiz as ff
from ..utils.fastfiz import create_table_state, shot_params_from_action
from ..utils import RewardFunction, DefaultReward
from typing import Optional


class BaseFastFiz(gym.Env):
    """Base class for FastFiz environments."""

    EPSILON_THETA = 0.001  # To avoid max theta (from FastFiz.h)
    TOTAL_BALLS = 16  # Including the cue ball

    def __init__(
        self, reward_function: RewardFunction = DefaultReward, num_balls: int = 15
    ) -> None:
        super().__init__()
        self.num_balls = num_balls
        self.reward = reward_function
        self.table_state = create_table_state(self.num_balls)
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)

        self.table_state = create_table_state(self.num_balls)
        self.reward.reset(self.table_state)
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute an action in the environment.
        """
        prev_table_state = self.table_state
        shot_params = shot_params_from_action(self.table_state, action)

        possible_shot = self._possible_shot(shot_params)

        if possible_shot:
            self.table_state.executeShot(shot_params)

        observation = self._get_observation()
        reward = self.reward.get_reward(
            prev_table_state, self.table_state, possible_shot
        )
        terminated = self._is_terminal_state()
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render() -> None:
        """
        Render the environment.
        """
        raise NotImplementedError("This method must be implemented")

    def _get_observation(self) -> np.ndarray:
        """
        Get the observation of the environment.
        """
        raise NotImplementedError("This method must be implemented")

    def _get_info(self) -> dict:
        """
        Get the info of the environment.
        """
        raise NotImplementedError("This method must be implemented")

    def _is_terminal_state(self) -> bool:
        """
        Check if the state is terminal
        """
        raise NotImplementedError("This method must be implemented")

    def _game_won(self) -> bool:
        """
        Check if the game is won.
        """
        raise NotImplementedError("This method must be implemented")

    def _possible_shot(self, shot_params: ff.ShotParams) -> bool:
        """
        Check if the shot is possible.
        """
        return (
            self.table_state.isPhysicallyPossible(shot_params)
            == ff.TableState.OK_PRECONDITION
        )

    def _observation_space(self) -> gym.Space:
        """
        Get the observation space.
        """
        raise NotImplementedError("This method must be implemented")

    def _action_space(self) -> gym.Space:
        """
        Get the action space.
        """
        raise NotImplementedError("This method must be implemented")
