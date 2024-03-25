from .. import BinaryReward
from ....utils.fastfiz import num_balls_in_play


class GameWonReward(BinaryReward):
    """
    Reward function that rewards based on whether the game is won.
    """

    def reset(self, table_state) -> None:
        self.num_balls = num_balls_in_play(table_state)

    def get_reward(self, prev_table_state, table_state, impossible_shot) -> float:
        """
        Reward function that returns 1 if the game is won, 0 otherwise.
        """
        return 1 if self._game_won(table_state) else 0

    def _game_won(self, table_state) -> bool:
        """
        Checks if the game is won based on the table state.

        Args:
            table_state (ff.TableState): The table state object representing the current state of the pool table.

        Returns:
            bool: True if the game is won, False otherwise.
        """
        for i in range(1, self.num_balls):
            if not table_state.getBall(i).isPocketed():
                return False
        return True
