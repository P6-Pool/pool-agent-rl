from .reward_function import RewardFunction, Weight
from .binary_reward import BinaryReward
import fastfiz as ff
import numpy as np


class CombinedReward(RewardFunction):
    """
    Reward function that combines multiple reward functions and weights them.
    """

    def __init__(
        self,
        weight: Weight = 1,
        max_episode_steps: int = None,
        *,
        reward_functions: list[RewardFunction],
        short_circuit: bool = False,
    ) -> None:
        """
        Initialize the CombinedReward object.

        Args:
            reward_functions (list[RewardFunction]): A list of reward functions to combine.
            weights (list[float]): A list of weights corresponding to each reward function.
            short_circuit (Optional[bool], default=False): If True, enable short circuiting of the reward calculation. Otherwise, all reward functions will be evaluated. Defaults to False.

        Returns:
            None
        """
        self.reward_functions = reward_functions
        super().__init__(weight, max_episode_steps=max_episode_steps)

        # Set max_episode_steps for all reward functions

        self.short_circuit = short_circuit
        self.max_episode_steps = max_episode_steps

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    @max_episode_steps.setter
    def max_episode_steps(self, value: int) -> None:
        self._max_episode_steps = value
        for reward in self.reward_functions:
            reward.max_episode_steps = value

    def reset(self, table_state: ff.TableState) -> None:
        super().reset(table_state)
        for reward in self.reward_functions:
            reward.reset(table_state)

    def reward(
        self,
        prev_table_state: ff.TableState,
        table_state: ff.TableState,
        action: np.ndarray,
    ) -> float:
        """
        Calculates the combined reward based on the given table states and possible shot flag.
        If short_circuit is True, the reward calculation will stop when a binary reward returns 1.
        Args:
            prev_table_state (ff.TableState): The previous table state.
            table_state (ff.TableState): The current table state.
            possible_shot (bool): Flag indicating if a shot is possible.

        Returns:
            float: The combined, weighted reward.

        """
        total_reward = 0
        for reward_function in self.reward_functions:
            reward = reward_function.get_reward(prev_table_state, table_state, action)
            total_reward += reward

            if issubclass(reward_function.__class__, BinaryReward):
                if reward == 1 * reward_function.weight() and self.short_circuit and reward_function.short_circuit:
                    return total_reward

        return total_reward

    def __str__(self) -> str:
        return (
            f"CombinedReward({[str(reward) for reward in self.reward_functions]}, {None}, short_circuit={self.short_circuit})"
        )
