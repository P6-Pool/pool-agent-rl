from .combined_reward import CombinedReward
from .common import *

rewards_functions = [
    StepPocketedReward(),
    BestTotalDistanceReward(),
    GameWonReward(),
    ImpossibleShotReward(),
    CueBallNotMovedReward(),
    CueBallPocketedReward(),
    ConstantReward(),
]
reward_weights = [1, 0.5, 10, -10, -10, -10, -0.1]

DefaultReward = CombinedReward(rewards_functions, reward_weights, short_circuit=True)
"""
Default reward function.

Uses the following weighted reward functions:
- StepPocketedReward: 1
- BestTotalDistanceReward: 0.5 
- GameWonReward: 10
- ImpossibleShotReward: -10
- CueBallNotMovedReward: -10
- CueBallPocketedReward: -10
- ConstantReward: -0.1

Returns:
    CombinedReward: The default reward function.
"""
