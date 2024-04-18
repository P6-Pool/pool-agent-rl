import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
from ..wrappers import ActionSpaces, FastFizActionWrapper

from fastfiz_env.utils.fastfiz.fastfiz import table_state_to_string
from ..utils.fastfiz import (
    create_random_table_state,
    get_ball_positions,
    normalize_ball_positions,
    shotparams_to_string,
)
from ..reward_functions import RewardFunction, DefaultReward
import fastfiz as ff
import logging
import time


class TestingFastFiz(gym.Env):
    """FastFiz environment for testing."""

    TOTAL_BALLS = 16

    def __init__(
        self,
        reward_function: RewardFunction = DefaultReward,
        num_balls: int = 16,
        *,
        options: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.options = options
        self.num_balls = num_balls
        self.table_state = create_random_table_state(self.num_balls)
        self.observation_space = self._observation_space()
        action_space_id = self.options.get("action_space_id", ActionSpaces.NO_OFFSET_3D)
        self.action_space = FastFizActionWrapper.get_action_space(action_space_id)
        self.max_episode_steps = None
        self.reward = reward_function

        # Logging
        self.logger = logging.getLogger(__name__)
        logs_dir = self.options.get("logs_dir", "")
        os.makedirs(logs_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(logs_dir, f"{time.strftime('%m-%d_%H:%M:%S')}.log"),
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
            level=self.options.get("log_level", logging.INFO),
        )

        self.n_episodes = 0
        self.n_step = 0

        self.logger.info(
            "TestFastFiz initialized with:\n- balls: %s\n- rewards: %s\n- options: %s\n- action space: %s\n- observation space: %s",
            self.num_balls,
            self.reward,
            self.options,
            self.action_space,
            self.observation_space,
        )

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

        seed = self.options.get("seed", None)
        self.logger.info("Reset(%s) - total n_steps: %s", self.n_episodes, self.n_step)
        self.logger.info("Reset(%s) - table state seed: %s", self.n_episodes, seed)
        self.table_state = create_random_table_state(self.num_balls, seed=seed)
        self.reward.reset(self.table_state)

        self.logger.info(
            "Reset(%s) - table state:\n%s",
            self.n_episodes,
            table_state_to_string(self.table_state),
        )

        observation = self._get_observation()
        info = self._get_info()

        self.logger.info(
            "Reset(%s) - initial observation:\n%s", self.n_episodes, observation
        )
        self.logger.info("Reset(%s) - initial info: %s", self.n_episodes, info)

        self.n_episodes += 1
        self.n_step = 0

        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute an action in the environment.
        """

        prev_table_state = ff.TableState(self.table_state)
        # action_space = spaces.Box(
        #     low=np.array([0, 0, -1, -1, -1]),
        #     high=np.array([0, 0, 1, 1, 1]),
        #     dtype=np.float32,
        # )
        # shot_params = action_to_shot([0, 0, *action], action_space)

        shot_params = ff.ShotParams(*action)

        self.logger.info(
            "Step(%s) - Action:\n- action: %s\n- shot params: %s",
            self.n_step,
            action,
            shotparams_to_string(shot_params),
        )

        impossible_shot = not self._possible_shot(shot_params)

        self.logger.info("Step(%s) - impossible shot: %s", self.n_step, impossible_shot)

        if not impossible_shot:
            self.table_state.executeShot(shot_params)

        observation = self._get_observation()

        reward = self.reward.get_reward(prev_table_state, self.table_state, action)

        terminated = self._is_terminal_state()
        truncated = False
        info = self._get_info()

        self.logger.debug("Step(%s) - observation:\n%s", self.n_step, observation)
        self.logger.info("Step(%s) - reward: %s", self.n_step, reward)
        self.logger.info("Step(%s) - terminated: %s", self.n_step, terminated)
        self.logger.debug("Step(%s) - truncated: %s", self.n_step, truncated)
        self.logger.info("Step(%s) - info: %s", self.n_step, info)

        self.n_step += 1

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        ball_positions = get_ball_positions(self.table_state)[: self.TOTAL_BALLS]
        # ball_positions = normalize_ball_positions(ball_positions)  # Normalize to [0, 1]
        ball_positions = (
            normalize_ball_positions(ball_positions) * 2 - 1
        )  # Normalize to [-1, 1] (symmetric)
        observation = np.zeros((self.TOTAL_BALLS, 2), dtype=np.float32)
        for i, ball_pos in enumerate(ball_positions):
            observation[i] = [*ball_pos]

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
        if self.table_state.getBall(0).isPocketed():
            return False

        for i in range(1, self.num_balls):
            if not self.table_state.getBall(i).isPocketed():
                return False
        return True

    def _observation_space(self) -> spaces.Box:
        """
        Get the observation space of the environment.

        The observation space is a 16-dimensional box with the position of each ball:
        - x: The x-coordinate of the ball.
        - y: The y-coordinate of the ball.

        All values are in the range `[0, TABLE_WIDTH]` and `[0, TABLE_LENGTH]`.
        """
        lower = np.full((self.TOTAL_BALLS, 2), [-1, -1])
        upper = np.full((self.TOTAL_BALLS, 2), [1, 1])
        return spaces.Box(low=lower, high=upper, dtype=np.float32)

    def _action_space(self) -> spaces.Box:
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
        # return spaces.Box(
        #     low=np.array([-1, -1, -1]),
        #     high=np.array([1, 1, 1]),
        #     dtype=np.float32,
        # )
        return FastFizActionWrapper.get_action_space(ActionSpaces.NO_OFFSET_4D)

    def _possible_shot(self, shot_params: ff.ShotParams) -> bool:
        """
        Check if the shot is possible.
        """
        return (
            self.table_state.isPhysicallyPossible(shot_params)
            == ff.TableState.OK_PRECONDITION
        )
