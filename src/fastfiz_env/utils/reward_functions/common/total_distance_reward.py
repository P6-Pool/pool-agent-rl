import numpy as np
from ..reward_function import RewardFunction
from ....utils.fastfiz import distances_to_closest_pocket, get_ball_positions, num_balls_in_play


class TotalDistanceReward(RewardFunction):
    """
    Reward function that gives a reward based on the total distance of the balls.
    """

    def reset(self, table_state) -> None:
        pass

    def get_reward(self, prev_table_state, table_state, possible_shot) -> float:
        """
        Reward function that gives a reward based on the total distance of the balls to the centers of the pockets.
        """
        balls_in_play = num_balls_in_play(table_state)
        ball_positions = get_ball_positions(table_state)[1:balls_in_play]
        total_distance = np.sum(distances_to_closest_pocket(ball_positions))
        return -float(total_distance)
