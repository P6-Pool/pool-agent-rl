from ..reward_function import RewardFunction


class GameWonReward(RewardFunction):
    def reset(self, table_state) -> None:
        pass

    def get_reward(self, prev_table_state, table_state, possible_shot) -> float:
        """
        Reward function returns -1 if the shot is impossible, 0 otherwise.
        """
        return 0 if possible_shot else -1

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
