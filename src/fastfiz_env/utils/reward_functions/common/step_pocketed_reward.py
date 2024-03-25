from ..reward_function import RewardFunction
from ....utils.fastfiz.fastfiz import num_balls_pocketed


class StepPocketedReward(RewardFunction):
    """
    Reward function that gives a reward based on the number of balls pocketed for the action.
    """

    def reset(self, table_state) -> None:
        pass

    def get_reward(self, prev_table_state, table_state, impossible_shot) -> float:
        """
        Reward function that gives a reward based on the number of balls pocketed for the action.
        """
        prev_pocketed = num_balls_pocketed(prev_table_state)
        pocketed = num_balls_pocketed(table_state)
        return float(pocketed - prev_pocketed)
