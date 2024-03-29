import numpy as np
import gymnasium as gym
import fastfiz as ff
from gymnasium import spaces
from ..utils.fastfiz import (
    shot_params_from_action,
    get_ball_positions,
    create_random_table_state,
    get_ball_velocity,
    normalize_ball_positions,
    normalize_ball_velocity,
    is_pocketed_state,
    GameBall,
)
from ..utils import RewardFunction, DefaultReward
from typing import Optional
import warnings
import vectormath as vmath


class SimpleFastFiz(gym.Env):
    """Base class for FastFiz environments."""

    EPSILON_THETA = 0.001  # To avoid max theta (from FastFiz.h)
    EVNET_SEQUENCE_LENGTH = 10
    TOTAL_BALLS = 16  # Including the cue ball

    def __init__(
        self, reward_function: RewardFunction = DefaultReward, num_balls: int = 16
    ) -> None:
        super().__init__()
        if num_balls < 2:
            warnings.warn(
                f"FastFiz environment initalized with num_balls={num_balls}.",
                UserWarning,
            )

        self.num_balls = num_balls
        self.reward = reward_function
        self.table_state = create_random_table_state(self.num_balls)
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)

        self.table_state = create_random_table_state(self.num_balls, seed=seed)
        self.reward.reset(self.table_state)
        observation = self._compute_observation(self.table_state, None)
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute an action in the environment.
        """
        prev_table_state = ff.TableState(self.table_state)

        shot_params = shot_params_from_action(self.table_state, action)

        impossible_shot = not self._possible_shot(shot_params)

        shot = None
        if not impossible_shot:
            shot = self.table_state.executeShot(shot_params)

        observation = self._compute_observation(prev_table_state, shot)
        reward = self.reward.get_reward(
            prev_table_state, self.table_state, impossible_shot
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

    def _get_observation(
        self, prev_table_state: ff.TableState, event_list: ff.EventVector
    ):
        return SimpleFastFiz.get_observation(
            prev_table_state, self.table_state, event_list
        )

    @staticmethod
    def get_observation(
        prev_table_state: ff.TableState,
        table_state: ff.TableState,
        event_list: ff.EventVector,
    ):
        initial_event_seq = SimpleFastFiz.event_sequence_from_table_state(
            prev_table_state
        )
        obs_sequence = np.zeros(
            (SimpleFastFiz.EVNET_SEQUENCE_LENGTH, SimpleFastFiz.TOTAL_BALLS, 4)
        )

        obs_sequence[0] = initial_event_seq
        for i, event in enumerate(event_list, start=1):
            obs_sequence[i] = obs_sequence[i - 1]
            event: ff.Event = event
            ball1_id = event.getBall1()
            ball2_id = event.getBall2()
            ball1: ff.Ball = event.getBall1Data()
            ball2: ff.Ball = event.getBall2Data()
            ball1_pocketed = ball1.isPocketed()
            ball1_pos = (ball1.getPos().x, ball1.getPos().y)
            ball2_pos = (ball2.getPos().x, ball2.getPos().y)
            ball1_vel = normalize_ball_velocity(get_ball_velocity(ball1))
            ball2_vel = normalize_ball_velocity(get_ball_velocity(ball2))

            obs_sequence[i][ball1_id] = [
                *normalize_ball_positions(ball1_pos),
                ball1_vel,
                int(ball1_pocketed),
            ]

            if ball2_id != ff.Ball.UNKNOWN_ID:
                ball2_pocketed = ball2.isPocketed()
                obs_sequence[i][ball2_id] = [
                    *normalize_ball_positions(ball2_pos),
                    ball2_vel,
                    int(ball2_pocketed),
                ]
            if i == SimpleFastFiz.EVNET_SEQUENCE_LENGTH - 1:
                final_event_seq = SimpleFastFiz.event_sequence_from_table_state(
                    table_state
                )
                obs_sequence[i] = final_event_seq
                break

        for ob in obs_sequence:
            if not np.all(ob == 0):
                print(ob)

        return obs_sequence

    @staticmethod
    def event_sequence_from_table_state(table_state: ff.TableState) -> np.ndarray:
        ball_positions = normalize_ball_positions(get_ball_positions(table_state))
        event_seq = np.zeros((SimpleFastFiz.TOTAL_BALLS, 4))
        for i, ball in enumerate(ball_positions):
            pocketed = table_state.getBall(i).isPocketed()
            event_seq[i] = [*ball, 0, int(pocketed)]
        return event_seq

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

        lower = np.full((self.TOTAL_BALLS, 4), [0, 0, 0, 0])
        # upper = np.full(
        #     (self.TOTAL_BALLS, 4),
        #     [
        #         table.TABLE_WIDTH,
        #         table.TABLE_LENGTH,
        #         self.table_state.MAX_VELOCITY * 1.580,
        #         1,
        #     ],
        # )
        upper = np.full(
            (self.TOTAL_BALLS, 4),
            [1, 1, 1, 1],
        )
        # Outerbox is shape (inner_box,10) inner boxes
        lower = np.full((self.EVNET_SEQUENCE_LENGTH, self.TOTAL_BALLS, 4), lower)
        upper = np.full((self.EVNET_SEQUENCE_LENGTH, self.TOTAL_BALLS, 4), upper)
        outer_box = spaces.Box(
            low=lower,
            high=upper,
            dtype=np.float32,
        )

        return outer_box

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
            low=np.array([0.0, 0.0, -1, -1, -1]),
            high=np.array([0.0, 0.0, 1, 1, 1]),
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

    def _compute_observation(
        self, prev_table_state: ff.TableState, shot: ff.Shot
    ) -> np.ndarray:
        frames = self.EVNET_SEQUENCE_LENGTH
        frames_seq = np.zeros((frames, self.TOTAL_BALLS, 4), dtype=np.float32)

        if shot is None:
            ball_positions = get_ball_positions(self.table_state)
            for frame in range(frames):
                for i, pos in enumerate(ball_positions):
                    pocketed = self.table_state.getBall(i).isPocketed()
                    frames_seq[frame][i] = [*pos, 0, pocketed]
            return frames_seq

        table: ff.Table = self.table_state.getTable()
        rolling = table.MU_ROLLING
        sliding = table.MU_SLIDING
        gravity = table.g
        total_time = shot.getDuration()
        interval = total_time / (frames - 1)

        game_balls: list[GameBall] = []
        for i in range(self.num_balls):
            position = prev_table_state.getBall(i).getPos()
            pos_vec = vmath.Vector2(position.x, position.y)
            game_balls.append(GameBall(0, i, pos_vec, ff.Ball.STATIONARY))

        for frame in range(frames):
            time = frame * interval
            for gb in game_balls:
                gb.update(time, shot, sliding, rolling, gravity)
                pocketed = is_pocketed_state(gb.state)
                frames_seq[frame][gb.number] = [
                    gb.position.x,
                    gb.position.y,
                    normalize_ball_velocity(np.hypot(gb.velocity.x, gb.velocity.y)),
                    pocketed,
                ]
        return frames_seq
