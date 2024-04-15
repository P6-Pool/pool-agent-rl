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


class ReplayFastFiz(gym.Env):
    """FastFiz environment for using different action spaces."""

    TOTAL_BALLS = 16

    def __init__(
        self,
        *,
        reward_function: RewardFunction = DefaultReward,
        num_balls: int = 16,
    ) -> None:
        super().__init__()
        self.num_balls = num_balls
        self.table_state = create_random_table_state(self.num_balls)
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()
        self.reward = reward_function
        self.max_episode_steps = None

    def _max_episode_steps(self):
        if self.get_wrapper_attr("_time_limit_max_episode_steps") is not None:
            self.max_episode_steps = self.get_wrapper_attr(
                "_time_limit_max_episode_steps"
            )
            self.reward.max_episode_steps = self.max_episode_steps

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if self.max_episode_steps is None:
            self._max_episode_steps()

        self.table_state = create_random_table_state(self.num_balls, seed=seed)
        self.reward.reset(self.table_state)
        self._prev_pocketed = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute an action in the environment.
        """

        self.prev_table_state = ff.TableState(self.table_state)
        shot_params = ff.ShotParams(*action)

        if self._possible_shot(shot_params):
            self.table_state.executeShot(shot_params)

        observation = self._get_observation()
        reward = self.reward.get_reward(self.prev_table_state, self.table_state, action)
        terminated = self._is_terminal_state()
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        ball_positions = get_ball_positions(self.table_state)[: self.TOTAL_BALLS]
        ball_positions = normalize_ball_positions(ball_positions) * 2 - 1
        observation = np.zeros((self.TOTAL_BALLS, 3), dtype=np.float32)
        for i, ball_pos in enumerate(ball_positions):
            pocketed = self.table_state.getBall(i).isPocketed()
            observation[i] = [*ball_pos, int(pocketed)]

        return np.array(observation)

    def compute_reward(self, achieved_goal, desired_goal, info):
        pass

    def _observation_space(self) -> spaces.Dict:
        """
        Get the observation space of the environment.

        The observation space is a 16-dimensional box with the position of each ball:
        - x: The x-coordinate of the ball.
        - y: The y-coordinate of the ball.

        All values are in the range `[-1, 1]`.
        """
        lower = np.full((self.TOTAL_BALLS, 3), [-1, -1, 0])
        upper = np.full((self.TOTAL_BALLS, 3), [1, 1, 1])

        obs_space = spaces.Dict(
            {
                "observation": spaces.Box(low=lower, high=upper, dtype=np.float32),
                "achieved_goal": spaces.Box(low=lower, high=upper, dtype=np.float32),
                "desired_goal": spaces.Box(low=lower, high=upper, dtype=np.float32),
            }
        )

        return obs_space

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
        return terminal_state(self.table_state)

    def _game_won(self) -> bool:
        return game_won(self.table_state)

    def _get_info(self):
        return {
            "is_success": self._game_won(),
        }
