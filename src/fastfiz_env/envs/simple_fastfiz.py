from logging import warn
import fastfiz as ff
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from fastfiz_env.utils.fastfiz.fastfiz import num_balls_pocketed

from ..utils.fastfiz import (
    create_random_table_state,
    get_ball_positions,
    normalize_ball_positions,
)
from .utils import game_won, terminal_state, possible_shot
from typing import Optional
from ..reward_functions import RewardFunction, DefaultReward


class SimpleFastFiz(gym.Env):
    """FastFiz environment for using different action spaces."""

    TOTAL_BALLS = 16

    def __init__(
        self,
        *,
        reward_function: RewardFunction = DefaultReward,
        num_balls: int = 16,
        options: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.num_balls = num_balls
        self.table_state = create_random_table_state(self.num_balls)
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()
        self.reward = reward_function
        self.max_episode_steps = None
        self.elapsed_steps = None
        self.options = options or {}
        self._quick_terminate = self.options.get("quick_terminate", False)

    def _get_time_limit_attrs(self):
        try:
            self.max_episode_steps = self.get_wrapper_attr("_max_episode_steps")
            self.elapsed_steps = self.get_wrapper_attr("_elapsed_steps")
        except AttributeError:
            warn.Warning(
                "The environment does not have a TimeLimit and/or TimeLimitInjection wrapper. The max_episode_steps attribute will not be available."
            )
            self.max_episode_steps = None
            self.elapsed_steps = None

        self.reward.max_episode_steps = self.max_episode_steps

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if self.max_episode_steps is None or self.elapsed_steps is None:
            self._get_time_limit_attrs()

        self.table_state = create_random_table_state(self.num_balls)
        self.reward.reset(self.table_state)
        self._prev_pocketed = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute an action in the environment.
        """
        self._prev_pocketed = num_balls_pocketed(self.table_state)
        prev_table_state = ff.TableState(self.table_state)
        shot_params = ff.ShotParams(*action)

        if self._possible_shot(shot_params):
            self.table_state.executeShot(shot_params)

        observation = self._get_observation()
        reward = self.reward.get_reward(prev_table_state, self.table_state, action)
        terminated = self._is_terminal_state()
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    @classmethod
    def compute_observation(cls, table_state: ff.TableState) -> np.ndarray:
        ball_positions = get_ball_positions(table_state)[: cls.TOTAL_BALLS]
        ball_positions = normalize_ball_positions(ball_positions) * 2 - 1
        observation = np.zeros((cls.TOTAL_BALLS, 2), dtype=np.float32)
        for i, ball_pos in enumerate(ball_positions):
            observation[i] = [*ball_pos]

        return np.array(observation)

    def _get_observation(self):
        return self.compute_observation(self.table_state)

    def _observation_space(self) -> spaces.Box:
        """
        Get the observation space of the environment.

        The observation space is a 16-dimensional box with the position of each ball:
        - x: The x-coordinate of the ball.
        - y: The y-coordinate of the ball.

        All values are in the range `[-1, 1]`.
        """
        lower = np.full((self.TOTAL_BALLS, 2), [-1, -1])
        upper = np.full((self.TOTAL_BALLS, 2), [1, 1])
        return spaces.Box(low=lower, high=upper, dtype=np.float32)

    def _action_space(self) -> spaces.Box:
        """
        Get the action space of the environment.

        The action space is a 3-dimensional box with the following parameters:
        - theta: The angle of the shot.
        - phi: The angle of the shot.
        - velocity: The velocity of the shot.

        All values are in the range `[-1, 1]`.
        """
        lower = np.array([-1, -1, -1])
        upper = np.array([1, 1, 1])
        return spaces.Box(low=lower, high=upper, dtype=np.float32)

    def _possible_shot(self, shot_params: ff.ShotParams) -> bool:
        return possible_shot(self.table_state, shot_params)

    def _is_terminal_state(self) -> bool:
        if self._quick_terminate:
            pocketed = num_balls_pocketed(self.table_state)

            if pocketed <= self._prev_pocketed:
                return True

        return terminal_state(self.table_state)

    def _game_won(self) -> bool:
        return game_won(self.table_state)

    def _get_info(self):
        return {
            "is_success": self._game_won(),
        }
