from .. import BinaryReward


class CueBallNotMovedReward(BinaryReward):
    """
    Reward function that reward based on whether the cue ball has moved.
    """

    def reset(self, table_state) -> None:
        pass

    def get_reward(self, prev_table_state, table_state, impossible_shot) -> float:
        """
        Reward function returns 1 if the cue ball has not moved, 0 otherwise.
        """

        prev_pos = prev_table_state.getBall(0).getPos()
        pos = table_state.getBall(0).getPos()
        cue_ball_not_moved = prev_pos.x == pos.x and prev_pos.y == pos.y

        return 1 if cue_ball_not_moved else 0
