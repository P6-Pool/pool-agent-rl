import fastfiz as ff
import numpy as np

from ...utils.fastfiz import (
    distances_to_closest_pocket,
    get_ball_positions,
    num_balls_in_play,
    pocket_centers,
)
from ..reward_function import RewardFunction


class DeltaBestTotalDistanceReward(RewardFunction):
    """
    Reward function that gives a reward based on the delta of the best total distance of the balls.
    """

    def reset(self, table_state) -> None:
        super().reset(table_state)
        self.pockets = pocket_centers(table_state)
        # num_balls = num_balls_in_play(table_state)
        ball_positions = get_ball_positions(table_state)[1 : self.num_balls]
        self.min_total_dist = np.sum(distances_to_closest_pocket(ball_positions, self.pockets))

    def reward(
        self,
        prev_table_state: ff.TableState,
        table_state: ff.TableState,
        action: np.ndarray,
    ) -> float:
        """
        Reward function that gives a reward based on the total distance of the balls to the centers of the pockets. The reward is the difference between the best total distance and the current total distance.
        """
        num_balls = num_balls_in_play(table_state)
        ball_positions = get_ball_positions(table_state)[1:num_balls]
        new_total_dist = np.sum(distances_to_closest_pocket(ball_positions, self.pockets))

        reward = float(self.min_total_dist - new_total_dist)

        if new_total_dist < self.min_total_dist:
            self.min_total_dist = new_total_dist

        return reward
