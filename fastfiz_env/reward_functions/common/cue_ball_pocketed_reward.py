import fastfiz as ff
import numpy as np

from ..binary_reward import BinaryReward


class CueBallPocketedReward(BinaryReward):
    """
    Reward function that reward based on whether the cue ball is pocketed.
    """

    def reward(
        self,
        prev_table_state: ff.TableState,
        table_state: ff.TableState,
        action: np.ndarray,
    ) -> float:
        """
        Reward function returns 1 if the cue ball is pocketed, 0 otherwise.
        """
        cue_ball_pocketed = table_state.getBall(0).isPocketed()
        return 1 if cue_ball_pocketed else 0
