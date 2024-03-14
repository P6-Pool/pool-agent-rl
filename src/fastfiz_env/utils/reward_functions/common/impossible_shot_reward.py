from ..reward_function import RewardFunction


class ImpossibleShotReward(RewardFunction):
    def reset(self, table_state) -> None:
        pass

    def get_reward(self, prev_table_state, table_state, possible_shot) -> float:
        """
        Reward function returns -1 if the shot is impossible, 0 otherwise.
        """
        return 0 if possible_shot else -1
