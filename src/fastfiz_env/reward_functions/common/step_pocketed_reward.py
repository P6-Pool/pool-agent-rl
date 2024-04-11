from ..reward_function import RewardFunction
from ...utils.fastfiz.fastfiz import num_balls_pocketed
import fastfiz as ff
import numpy as np


class StepPocketedReward(RewardFunction):
    """
    Reward function that gives a reward based on the number of balls pocketed for the action.
    """

    # def reset(self, table_state) -> None:
    #     super().reset(table_state)

    def reward(
        self,
        prev_table_state: ff.TableState,
        table_state: ff.TableState,
        action: np.ndarray,
    ) -> float:
        """
        Reward function that gives a reward based on the number of balls pocketed for the action.
        """
        prev_pocketed = num_balls_pocketed(prev_table_state, range_start=1)
        pocketed = num_balls_pocketed(table_state, range_start=1)
        return pocketed - prev_pocketed
