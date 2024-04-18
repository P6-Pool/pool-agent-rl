from .combined_reward import CombinedReward
from .common import *

rewards = [
    GameWonReward(ConstantWeight),
    CueBallPocketedReward(NegativeConstantWeight),
    StepNoBallsPocketedReward(NegativeConstantWeight),
    ConstantReward(NegativeConstantWeightMaxSteps),
    StepPocketedReward(ConstantWeightBalls),
    ExponentialVelocityReward(NegativeConstantWeight),
]

WinningReward = CombinedReward(reward_functions=rewards, short_circuit=True)
"""
Winning reward function.

Uses the following weighted reward functions:
- GameWonReward: 1
- CueBallPocketedReward: -1
- StepNoBallsPocketedReward: -1
- ConstantReward: -1 / max_episode_steps
- StepPocketedReward: 1 / (num_balls - 1)
- ExponentialVelocityReward: -1

Returns:
    CombinedReward: The default reward function.
"""
