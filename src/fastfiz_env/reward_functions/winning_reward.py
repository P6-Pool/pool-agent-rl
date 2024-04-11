from .combined_reward import CombinedReward
from .common import *

rewards = [
    ImpossibleShotReward(NegativeConstantWeightMaxSteps),
    CueBallPocketedReward(NegativeConstantWeight),
    ConstantReward(NegativeConstantWeightMaxSteps),
    StepPocketedReward(ConstantWeight),
    ExponentialVelocityReward(NegativeConstantWeight),
]


WinningReward = CombinedReward(reward_functions=rewards, short_circuit=True)
"""
Winning reward function.

Uses the following weighted reward functions:
- ImpossibleShotReward: -1 / max_episode_steps
- CueBallPocketedReward: -1
- ConstantReward: -1 / max_episode_steps
- StepPocketedReward: 1
- ExponentialVelocityReward: -1

Returns:
    CombinedReward: The reward function.
"""
