import numpy as np
from ..reward_function import RewardFunction
from ....utils.fastfiz import (
    distances_to_closest_pocket,
    get_ball_positions,
    num_balls_in_play,
    pocket_centers,
)
import fastfiz as ff


class TotalDistanceReward(RewardFunction):
    """
    Reward function that gives a reward based on the total distance of the balls to the centers of the pockets.
    """

    def reset(self, table_state) -> None:
        super().reset(table_state)
        self.pockets = pocket_centers(table_state)

    def reward(
        self,
        prev_table_state: ff.TableState,
        table_state: ff.TableState,
        action: np.ndarray,
    ) -> float:
        """
        Reward function that gives a reward based on the total distance of the balls to the centers of the pockets.
        """
        num_balls = num_balls_in_play(table_state)
        ball_positions = get_ball_positions(table_state)[1:num_balls]
        total_distance = np.sum(
            distances_to_closest_pocket(ball_positions, self.pockets)
        )
        return total_distance
