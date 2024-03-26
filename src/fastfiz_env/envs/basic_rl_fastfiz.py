import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
from ..utils.fastfiz import (
    create_random_table_state,
    get_ball_positions,
    normalize_ball_positions,
    shot_params_from_action,
)
from ..utils import RewardFunction, DefaultReward
import fastfiz as ff


class BasicRLFastFiz(gym.Env):
    """FastFiz environment with random initial state, used for reinforcemet learning."""

    TOTAL_BALLS = 16

    def __init__(
        self,
        reward_function: RewardFunction = DefaultReward,
        num_balls: int = 16,
    ) -> None:
        super().__init__()
        self.num_balls = num_balls
        self.table_state = create_random_table_state(self.num_balls)
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()
        self.reward = reward_function

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.table_state = create_random_table_state(self.num_balls, seed=seed)
        self.reward.reset(self.table_state)

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute an action in the environment.
        """
        prev_table_state = ff.TableState(self.table_state)
        shot_params = shot_params_from_action(self.table_state, [0, 0, 0.65, *action])

        impossible_shot = not self._possible_shot(shot_params)

        if not impossible_shot:
            self.table_state.executeShot(shot_params)
        else:
            print("Impossible shot")

        observation = self._get_observation()
        reward = self.reward.get_reward(
            prev_table_state, self.table_state, impossible_shot
        )
        terminated = self._is_terminal_state()
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        ball_positions = get_ball_positions(self.table_state)
        ball_positions = normalize_ball_positions(ball_positions) * 2 - 1
        observation = np.zeros((self.TOTAL_BALLS, 3), dtype=np.float32)
        for i, ball_pos in enumerate(ball_positions):
            observation[i] = [*ball_pos, not self.table_state.getBall(i).isInPlay()]

        return np.array(observation)

    def _get_info(self):
        return {
            "is_success": self._game_won(),
        }

    def _is_terminal_state(self) -> bool:
        if self.table_state.getBall(0).isPocketed():
            return True

        return self._game_won()

    def _game_won(self) -> bool:
        for i in range(1, self.num_balls):
            if not self.table_state.getBall(i).isPocketed():
                return False
        return True

    def _observation_space(self):
        """
        Get the observation space of the environment.

        The observation space is a 16-dimensional box with the position of each ball:
        - x: The x-coordinate of the ball.
        - y: The y-coordinate of the ball.

        All values are in the range `[0, TABLE_WIDTH]` and `[0, TABLE_LENGTH]`.
        """
        table = self.table_state.getTable()
        lower = np.full((self.TOTAL_BALLS, 3), [-1, -1, 0])
        upper = np.full((self.TOTAL_BALLS, 3), [1, 1, 1])
        return spaces.Box(low=lower, high=upper, dtype=np.float32)

    def _action_space(self):
        """
        Get the action space of the environment.

        The action space is a 5-dimensional box:
        - a-offset: The offset of the cue ball in the x-coordinate.
        - b-offset: The offset of the cue ball in the y-coordinate.
        - theta: The angle of the shot in the yz-plane.
        - phi: The angle of the shot.
        - velocity: The power of the shot.

        All values are in the range `[0, 1]`.
        """
        return spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32,
        )

    def _possible_shot(self, shot_params: ff.ShotParams) -> bool:
        """
        Check if the shot is possible.
        """
        return (
            self.table_state.isPhysicallyPossible(shot_params)
            == ff.TableState.OK_PRECONDITION
        )
