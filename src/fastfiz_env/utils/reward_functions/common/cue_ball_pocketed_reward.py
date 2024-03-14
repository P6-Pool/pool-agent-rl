from ..reward_function import RewardFunction


class CueBallPocketedReward(RewardFunction):
    """
    Reward function that reward based on whether the cue ball is pocketed.
    """

    def reset(self, table_state) -> None:
        pass

    def get_reward(self, prev_table_state, table_state, possible_shot) -> float:
        """
        Reward function returns -1 if the cue ball is pocketed, 0 otherwise.
        """
        cue_ball_pocketed = table_state.getBall(0).isPocketed()
        return -1 if cue_ball_pocketed else 0
