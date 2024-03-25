from .. import BinaryReward


class ImpossibleShotReward(BinaryReward):
    """
    Reward function that rewards based on whether the shot is possible.
    """

    def reset(self, table_state) -> None:
        pass

    def get_reward(self, prev_table_state, table_state, impossible_shot) -> float:
        """
        Reward function returns 1 if the shot is impossible, 0 otherwise.
        """
        return 1 if impossible_shot else 0
