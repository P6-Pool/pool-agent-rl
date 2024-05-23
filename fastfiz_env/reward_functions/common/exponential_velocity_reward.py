import fastfiz as ff
import numpy as np

from ..reward_function import RewardFunction


class ExponentialVelocityReward(RewardFunction):
    """
    Reward function that gives a reward based on velocity of the action.
    """

    def reward(
        self,
        prev_table_state: ff.TableState,
        table_state: ff.TableState,
        action: np.ndarray,
    ) -> float:
        """
        Reward function that gives a reward based on velocity of the action.
        """
        reward = float(np.interp(action[4], [0, 10], [0, 1]))
        return reward**10
