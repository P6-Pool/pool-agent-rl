from .reward_function import RewardFunction
import fastfiz as ff
import numpy as np


class CombinedReward(RewardFunction):
    """
    Reward function that combines multiple reward functions and weights them.
    """

    def __init__(self, reward_functions: list[RewardFunction], weights: list[float]) -> None:
        self.reward_functions = reward_functions
        self.weights = weights

    def reset(self, table_state: ff.TableState) -> None:
        super().reset(table_state)
        for reward in self.reward_functions:
            reward.reset(table_state)

    def get_reward(self, prev_table_state: ff.TableState, table_state: ff.TableState, possible_shot: bool) -> float:
        """
        Reward function that combines multiple reward functions and weights them.
        """
        reward = np.dot([reward.get_reward(prev_table_state, table_state, possible_shot)
                        for reward in self.reward_functions], self.weights)
        print(f"Reward: {reward}")
        return reward
