from ...utils.fastfiz import any_ball_has_moved, get_ball_positions
from .. import BinaryReward


class BallsNotMovedReward(BinaryReward):
    """
    Reward function that reward based on whether any balls has moved.
    """

    def reward(self, prev_table_state, table_state, action) -> float:
        """
        Reward function returns 1 if any balls has moved, 0 otherwise.
        """
        prev_ball_positions = get_ball_positions(prev_table_state)[1:]
        ball_positions = get_ball_positions(table_state)[1:]

        not_moved = not any_ball_has_moved(prev_ball_positions, ball_positions)

        return 1 if not_moved else 0
