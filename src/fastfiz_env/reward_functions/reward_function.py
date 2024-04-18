from abc import ABC, abstractmethod
import fastfiz as ff
from typing import Union, Callable, Optional, TypeAlias
import numpy as np
from fastfiz_env.utils.fastfiz.fastfiz import num_balls_in_play


Weight: TypeAlias = Union[float, Callable[[int, int, Optional[int]], float]]
"""
Type alias for a weight value or a callable that calculates the weight of a reward function.

The callable takes the following arguments.
Args:
    int: The number of balls in play.
    int: The current step.
    int (optional): The maximum number of steps.

Returns:
    float: The calculated weight.
"""


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.
    """

    def __init__(
        self,
        weight: Union[Weight, float] = 1,
        *,
        max_episode_steps: Optional[int] = None,
    ) -> None:
        """
        Initializes the reward function.

        Args:
            weight (Union[float, Callable[[int, int, int], float]]): The weight of the reward function. If a callable is provided, it should take the number of balls in play, the current step, and the maximum number of steps as arguments. Otherwise, a float value is expected.
            max_episode_steps (Optional[int], default=None): The maximum number of steps in an episode. Defaults to None.

        """
        super().__init__()
        self.__reset_called = False
        self.__weight = weight
        # if max_episode_steps is None:
        #     raise ValueError(
        #         "max_episode_steps must be provided when instantiating a RewardFunction. Try using MaxEpisodeStepsInjectionWrapper with your environment."
        #     )
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.num_balls = 0

    def reset(self, table_state: ff.TableState) -> None:
        """
        Resets the reward function.

        Args:
            table_state (ff.TableState): The current state of the pool table.

        Returns:
            None
        """
        self.__reset_called = True
        self.num_balls = num_balls_in_play(table_state)
        self.current_step = 0

    def get_reward(
        self,
        prev_table_state: ff.TableState,
        table_state: ff.TableState,
        action: np.ndarray[float, np.dtype[Union[np.float32, np.float64]]],
    ) -> float:
        """
        Calculates the weigthed reward for a given table state transition.

        Args:
            prev_table_state (ff.TableState): The previous table state.
            table_state (ff.TableState): The current table state.

        Returns:
            float: The calculated reward value.
        """
        if not self.__reset_called:
            raise RuntimeError(f"{self.__class__.__name__} reset() method must be called before calling get_reward().")
        self.current_step += 1
        return self.reward(prev_table_state, table_state, action) * self.weight()

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

        Returns:
            float: The calculated reward value.
        """

    def weight(
        self,
    ) -> float:
        """
        Calculates the weight of the reward function.

        Returns:
            float: The weight of the reward function.
        """
        if callable(self.__weight):
            return self.__weight(self.num_balls, self.current_step, self.max_episode_steps)
        return self.__weight

    def __str__(self):
        return self.__class__.__name__
