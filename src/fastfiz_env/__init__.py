"""
Gymnasium environments for pool, using FastFiz to simulate the physics of the game.

Avaliable environments:
    - `SimpleFastFiz-v0`: Observes the position of the balls.
    - `VelocityFastFiz-v0`: Observes the velocity of the balls.
    - `TestingFastFiz-v0`: Observes the position of the balls. Used for testing purposes with options e.g. seed, logging, action_space_id.


### Example

Use the environment for training a reinforcement learning agent:

```python
from stable_baselines3 import PPO
import fastfiz_env
from fastfiz_env.utils.reward_functions.common import StepPocketedReward

reward_function = StepPocketedReward()
env = fastfiz_env.make("SimpleFastFiz-v0", reward_function=reward_function, num_balls=2)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10_000)

```

"""

from .make import make
from .reward_functions import DefaultReward, RewardFunction, CombinedReward
from . import envs, utils, wrappers, reward_functions

__all__ = [
    "make",
    "DefaultReward",
    "RewardFunction",
    "CombinedReward",
    "envs",
    "utils",
    "wrappers",
    "reward_functions",
]

from gymnasium.envs.registration import register


register(
    id="VelocityFastFiz-v0",
    entry_point="fastfiz_env.envs:VelocityFastFiz",
    additional_wrappers=(wrappers.MaxEpisodeStepsInjectionWrapper.wrapper_spec(),),
)

register(
    id="SimpleFastFiz-v0",
    entry_point="fastfiz_env.envs:SimpleFastFiz",
    additional_wrappers=(wrappers.MaxEpisodeStepsInjectionWrapper.wrapper_spec(),),
)


register(
    id="TestingFastFiz-v0",
    entry_point="fastfiz_env.envs:TestingFastFiz",
    additional_wrappers=(wrappers.MaxEpisodeStepsInjectionWrapper.wrapper_spec(),),
)
