from .reward_function import RewardFunction
from .binary_reward import BinaryReward
import fastfiz as ff


class CombinedReward(RewardFunction):
    """
    Reward function that combines multiple reward functions and weights them.
    """

    def __init__(
        self,
        reward_functions: list[RewardFunction],
        weights: list[float | int],
        *,
        short_circuit: bool = False
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
        self.weights = weights
        self.short_circuit = short_circuit

    def reset(self, table_state: ff.TableState) -> None:
        super().reset(table_state)
        for reward in self.reward_functions:
            reward.reset(table_state)

    def get_reward(
        self,
        prev_table_state: ff.TableState,
        table_state: ff.TableState,
        impossible_shot: bool,
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
        for i, reward_function in enumerate(self.reward_functions):
            reward = reward_function.get_reward(
                prev_table_state, table_state, impossible_shot
            )
            total_reward += reward * self.weights[i]

            if issubclass(reward_function.__class__, BinaryReward):
                if reward == 1 and self.short_circuit and reward_function.short_circuit:
                    return total_reward

        return total_reward
