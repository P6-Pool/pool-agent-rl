import numpy as np
from gymnasium import spaces
import gymnasium as gym
import fastfiz as ff
from typing import Optional
from ..utils.fastfiz import create_random_table_state, get_ball_positions, num_balls_pocketed, distances_to_closest_pockets
from . import BaseFastFiz

# Reward weights
RW_GAME_WON = 100
RW_BALL_POCKETED = 5
RW_SHOT_MADE = -1
RW_CUE_BALL_POCKETED = -100
RW_IMPOSSIBLE_SHOT = -100
RW_CUE_BALL_NOT_MOVED = -100


class BaseRLFastFiz(BaseFastFiz):
    """FastFiz environment with random initial state, used for reinforcemet learning."""
    EPSILON_THETA = 0.001  # To avoid max theta (from FastFiz.h)
    TOTAL_BALLS = 16  # Including the cue ball

    def __init__(self, num_balls: Optional[int] = 15, ) -> None:
        super().__init__(num_balls=num_balls)
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.table_state = create_random_table_state(self.num_balls, seed=seed)
        self.min_dist = distances_to_closest_pockets(get_ball_positions(self.table_state))[
            1:self.num_balls]

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _get_observation(self):
        ball_positions = get_ball_positions(self.table_state)

        observation = []
        for i, ball_pos in enumerate(ball_positions):
            if self.table_state.getBall(i).isInPlay():
                observation.append(ball_pos)
            else:
                observation.append([-1, -1])

        return np.array(observation)

    def _get_reward(self, prev_table_state: ff.TableState, possible_shot: bool) -> float:
        if self._game_won():
            return RW_GAME_WON

        if self.table_state.getBall(0).isPocketed():
            return RW_CUE_BALL_POCKETED

        if not possible_shot:
            return RW_IMPOSSIBLE_SHOT

        if self.table_state.getBall(0).getPos() == prev_table_state.getBall(0).getPos():
            return RW_CUE_BALL_NOT_MOVED

        prev_pocketed = num_balls_pocketed(prev_table_state)
        pocketed = num_balls_pocketed(self.table_state)
        step_pocketed = pocketed - prev_pocketed

        reward = step_pocketed * RW_BALL_POCKETED

        new_min_dist = distances_to_closest_pockets(
            get_ball_positions(self.table_state))[1:self.num_balls]

        if sum(new_min_dist) < sum(self.min_dist):
            reward += 1

        reward += RW_SHOT_MADE
        return reward

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
        lower = np.full((self.TOTAL_BALLS, 2), [0, 0])
        upper = np.full((self.TOTAL_BALLS, 2), [
                        table.TABLE_WIDTH, table.TABLE_LENGTH])
        return spaces.Box(
            low=lower, high=upper, dtype=np.float64)

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
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([0.0, 0.0, 1, 1, 1]),
            dtype=np.float64,
        )
