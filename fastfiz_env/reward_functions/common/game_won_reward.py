import fastfiz as ff
import numpy as np

from .. import BinaryReward


class GameWonReward(BinaryReward):
    """
    Reward function that rewards based on whether the game is won.
    """

    def reward(
        self,
        prev_table_state: ff.TableState,
        table_state: ff.TableState,
        action: np.ndarray,
    ) -> float:
        """
        Reward function that returns 1 if the game is won, 0 otherwise.
        """
        return 1 if self._game_won(table_state) else 0

    def _game_won(self, table_state: ff.TableState) -> bool:
        """
        Checks if the game is won based on the table state.

        Args:
            table_state (ff.TableState): The table state object representing the current state of the pool table.

        Returns:
            bool: True if the game is won, False otherwise.
        """
        if table_state.getBall(0).isPocketed():
            return False
        for i in range(1, self.num_balls):
            if not table_state.getBall(i).isPocketed():
                return False
        return True
