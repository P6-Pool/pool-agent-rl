from ..reward_function import RewardFunction


class ConstantReward(RewardFunction):
    """
    Reward function that always returns 1. Inteded to be used in combination with other reward functions.
    """

    def reset(self, table_state) -> None:
        pass

    def get_reward(self, prev_table_state, table_state, impossible_shot) -> float:
        """
        Reward function that always returns 1. Inteded to be used in combination with other reward functions.
        """
        return 1
