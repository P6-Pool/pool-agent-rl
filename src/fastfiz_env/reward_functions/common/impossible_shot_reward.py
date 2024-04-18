from .. import BinaryReward
import fastfiz as ff
import numpy as np


class ImpossibleShotReward(BinaryReward):
    """
    Reward function that rewards based on whether the shot is possible.
    """

    def reward(
        self,
        prev_table_state: ff.TableState,
        table_state: ff.TableState,
        action: np.ndarray[float, np.dtype[np.float32]],
    ) -> float:
        """
        Reward function returns 1 if the shot is impossible, 0 otherwise.
        """
        shot_params = ff.ShotParams(*action)
        impossible_shot = (
            table_state.isPhysicallyPossible(shot_params)
            != ff.TableState.OK_PRECONDITION
        )
        return 1 if impossible_shot else 0
