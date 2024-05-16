from ...reward_functions import BinaryReward
from ...utils.fastfiz.fastfiz import num_balls_pocketed
import fastfiz as ff
import numpy as np


class StepNoBallsPocketedReward(BinaryReward):
    """
    Reward function that rewards based on whether no balls were pocketed in the step.
    """

    def reward(
        self,
        prev_table_state: ff.TableState,
        table_state: ff.TableState,
        action: np.ndarray,
    ) -> float:
        """
        Reward function returns 1 if any number of balls were pocketed in the step, 0 otherwise.
        """
        prev_pocketed = num_balls_pocketed(prev_table_state, range_start=1)
        pocketed = num_balls_pocketed(table_state, range_start=1)
        return float(pocketed == prev_pocketed)
