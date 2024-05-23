"""
Gymnasium environments for FastFiz pool simulator.

Avaliable environments:
    - `FastFiz-v0`: Observes the position of the balls.
    - `PocketsFastFiz-v0`: Observes the position of the balls and in play state. Pocketed balls position always corresponds to given pocket center.


### Example

Use the environment for training a reinforcement learning agent:

```python
from stable_baselines3 import PPO
from fastfiz_env import DefaultReward, make

env = make("FastFiz-v0", reward_function=DefaultReward, num_balls=2)

model = PPO("MlpPolicy", env)

model.learn(total_timesteps=100_000)

```

"""

__version__ = "0.0.1"

from . import envs, reward_functions, utils, wrappers
from .make import make, make_callable_wrapped_env, make_wrapped_env
from .reward_functions import CombinedReward, DefaultReward, RewardFunction

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
    id="FastFiz-v0",
    entry_point="fastfiz_env.envs:FastFiz",
    additional_wrappers=(wrappers.TimeLimitInjectionWrapper.wrapper_spec(),),
)

register(
    id="PocketsFastFiz-v0",
    entry_point="fastfiz_env.envs:PocketsFastFiz",
    additional_wrappers=(wrappers.TimeLimitInjectionWrapper.wrapper_spec(),),
)
