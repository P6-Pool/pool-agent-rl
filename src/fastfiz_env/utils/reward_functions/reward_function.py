from abc import ABC, abstractmethod
import fastfiz as ff


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.
    """

    @abstractmethod
    def reset(self, table_state: ff.TableState) -> None:
        """
        Resets the reward function.

        Args:
            table_state (ff.TableState): The current state of the pool table.

        Returns:
            None
        """
        pass

    @abstractmethod
    def get_reward(
        self,
        prev_table_state: ff.TableState,
        table_state: ff.TableState,
        possible_shot: bool,
    ) -> float:
        """
        Calculates the reward for a given table state transition.

        Args:
            prev_table_state (ff.TableState): The previous table state.
            table_state (ff.TableState): The current table state.
            possible_shot (bool): Indicates whether a shot is possible.

        Returns:
            float: The calculated reward value.
        """
        pass
