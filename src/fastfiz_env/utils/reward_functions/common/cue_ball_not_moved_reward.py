from ..reward_function import RewardFunction


class CueBallNotMovedReward(RewardFunction):
    """
    Reward function that reward based on whether the cue ball has moved.
    """

    def reset(self, table_state) -> None:
        pass

    def get_reward(self, prev_table_state, table_state, possible_shot) -> float:
        """
        Reward function returns 1 if the cue ball has not moved, 0 otherwise.
        """
        cue_ball_moved = table_state.getBall(
            0).getPos() != prev_table_state.getBall(0).getPos()
        return 1 if not cue_ball_moved else 0
