from abc import ABC, abstractmethod
from typing import Callable, Union
import fastfiz as ff
from .reward_function import RewardFunction
import numpy as np


class BinaryReward(RewardFunction, ABC):
    def __init__(
        self,
        weight: Union[float, Callable[[int, int, int], float]] = 1,
        *,
        max_episode_steps: int = None,
        short_circuit: bool = True,
    ) -> None:
        """
        Initializes a BinaryReward object.

        Args:
            short_circuit (bool, optional): Determines whether to short circuit the reward calculation when used in `CombinedReward`.
                If set to True, the reward will be calculated based on the first condition that is met.
                If set to False, all conditions will be evaluated. Defaults to True.
        """
        super().__init__(weight=weight, max_episode_steps=max_episode_steps)
        self.short_circuit = short_circuit

    @abstractmethod
    def reward(
        self,
        prev_table_state: ff.TableState,
        table_state: ff.TableState,
        action: np.ndarray,
    ) -> float:
        """
        Calculates the reward for a given table state transition.

        Args:
            prev_table_state (ff.TableState): The previous table state.
            table_state (ff.TableState): The current table state.
            possible_shot (bool): Indicates whether a shot is possible.

        Returns:
            float: If the condition is met, return 1, otherwise return 0.
        """
        pass

    def __str__(self) -> str:
        return super().__str__() + f"({self.short_circuit})"
