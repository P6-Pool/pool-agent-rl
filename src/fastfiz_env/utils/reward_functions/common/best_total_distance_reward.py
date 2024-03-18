from ..reward_function import RewardFunction
from ....utils.fastfiz import (
    distances_to_closest_pocket,
    get_ball_positions,
    num_balls_in_play,
    pocket_centers,
)
import numpy as np


class BestTotalDistanceReward(RewardFunction):
    """
    Reward function that gives a reward based on the best total distance of the balls.
    """

    def reset(self, table_state) -> None:
        self.pockets = pocket_centers(table_state)
        num_balls = num_balls_in_play(table_state)
        ball_positions = get_ball_positions(table_state)[1:num_balls]
        self.min_total_dist = np.sum(
            distances_to_closest_pocket(ball_positions, self.pockets)
        )

    def get_reward(self, prev_table_state, table_state, impossible_shot) -> float:
        """
        Reward function that gives a reward based on the total distance of the balls to the centers of the pockets. The reward is the difference between the best total distance and the current total distance.
        """
        num_balls = num_balls_in_play(table_state)
        ball_positions = get_ball_positions(table_state)[1:num_balls]
        new_total_dist = np.sum(
            distances_to_closest_pocket(ball_positions, self.pockets)
        )

        reward = float(self.min_total_dist - new_total_dist)

        if new_total_dist < self.min_total_dist:
            self.min_total_dist = new_total_dist

        return reward
