from .combined_reward import CombinedReward
from .common import *

rewards = [
    GameWonReward(ConstantWeight),
    CueBallPocketedReward(NegativeConstantWeight),
    ConstantReward(NegativeConstantWeightMaxSteps),
    BallsNotMovedReward(NegativeConstantWeightMaxSteps),
    StepPocketedReward(ConstantWeightBalls),
    ExponentialVelocityReward(NegativeConstantWeight),
]

DefaultReward = CombinedReward(reward_functions=rewards, short_circuit=True)
"""
Default reward function.

Uses the following weighted reward functions:
- GameWonReward: 1
- CueBallPocketedReward: -1
- ConstantReward: -1 / max_episode_steps
- BallsNotMovedReward: -1 / max_episode_steps
- StepPocketedReward: 1 / (num_balls - 1)
- ExponentialVelocityReward: -1

Returns:
    CombinedReward: The default reward function.
"""
