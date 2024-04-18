"""
Gymnasium environments for pool, using FastFiz to simulate the physics of the game.

Avaliable environments:
    - `SimpleFastFiz-v0`: Observes the position of the balls.
    - `FramesFastFiz-v0`: Observes the position of the balls and the frames of the simulation.
    - `PocketsFastFiz-v0`: Observes the position of the balls and in play state. Pocketed balls position always corresponds to given pocket center.


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

__version__ = "0.0.1"

from .make import make, make_wrapped_env, make_callable_wrapped_env
from .reward_functions import DefaultReward, RewardFunction, CombinedReward
from . import envs, utils, wrappers, reward_functions

__all__ = [
    "make",
    "make_wrapped_env",
    "make_callable_wrapped_env",
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
    id="SimpleFastFiz-v0",
    entry_point="fastfiz_env.envs:SimpleFastFiz",
    additional_wrappers=(wrappers.TimeLimitInjectionWrapper.wrapper_spec(),),
)


register(
    id="FramesFastFiz-v0",
    entry_point="fastfiz_env.envs:FramesFastFiz",
    additional_wrappers=(wrappers.TimeLimitInjectionWrapper.wrapper_spec(),),
)

register(
    id="PocketsFastFiz-v0",
    entry_point="fastfiz_env.envs:PocketsFastFiz",
    additional_wrappers=(wrappers.TimeLimitInjectionWrapper.wrapper_spec(),),
)
