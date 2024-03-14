from .reward_function import RewardFunction
from ...utils.fastfiz import num_balls_in_play, num_balls_pocketed, distances_to_closest_pocket, get_ball_positions
import fastfiz as ff

# Reward weights
RW_GAME_WON = 100
RW_BALL_POCKETED = 5
RW_SHOT_MADE = -1
RW_CUE_BALL_POCKETED = -100
RW_IMPOSSIBLE_SHOT = -100
RW_CUE_BALL_NOT_MOVED = -100


class DefaultReward(RewardFunction):
    """
    Default reward function that gives a reward of 0.
    """

    def reset(self, table_state) -> None:
        self.num_balls = num_balls_in_play(table_state)
        self.min_dist = distances_to_closest_pocket(
            get_ball_positions(table_state))[1:self.num_balls]

    def get_reward(self, prev_table_state, table_state, possible_shot) -> float:
        if not possible_shot:
            return RW_IMPOSSIBLE_SHOT

        if table_state.getBall(0).isPocketed():
            return RW_CUE_BALL_POCKETED

        if self._game_won():
            return RW_GAME_WON

        if table_state.getBall(0).getPos() == prev_table_state.getBall(0).getPos():
            return RW_CUE_BALL_NOT_MOVED

        prev_pocketed = num_balls_pocketed(prev_table_state)
        pocketed = num_balls_pocketed(table_state)
        step_pocketed = pocketed - prev_pocketed

        reward = step_pocketed * RW_BALL_POCKETED

        reward += RW_SHOT_MADE
        return reward

    def _game_won(self, table_state: ff.TableState) -> bool:
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
