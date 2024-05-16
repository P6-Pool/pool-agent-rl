from ..reward_function import RewardFunction
import fastfiz as ff
import numpy as np


class ConstantReward(RewardFunction):
    """
    Reward function that always returns 1. Inteded to be used in combination with other reward functions.
    """

    def reward(
        self,
        prev_table_state: ff.TableState,
        table_state: ff.TableState,
        action: np.ndarray,
    ) -> float:
        """
        Reward function that always returns 1. Inteded to be used in combination with other reward functions.
        """
        return 1
