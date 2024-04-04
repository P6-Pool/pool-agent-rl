"""
Gymnasium environments for pool, using FastFiz to simulate the physics of the game.

Avaliable environments:
    - `BaseFastFiz-v0`: Base class for FastFiz.
    - `BaseRLFastFiz-v0`: Base class for FastFiz with reinforcement learning, using initial random table state.
    - `PocketRLFastFiz-v0`: Subclass of BaseRLFastFiz. Observes if a ball is pocketed.


### Example

Use the environment for training a reinforcement learning agent:

```python
from stable_baselines3 import PPO
import fastfiz_env
from fastfiz_env.utils.reward_functions.common import StepPocketedReward

reward_function = StepPocketedReward()
env = fastfiz_env.make("BaseRLFastFiz-v0", reward_function=reward_function, num_balls=2)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10_000)

```

"""

from .make import make
from . import envs, utils

__all__ = ["make", "envs", "utils"]

from gymnasium.envs.registration import register

register(
    id="BaseFastFiz-v0",
    entry_point="fastfiz_env.envs:BaseFastFiz",
)

register(
    id="BaseRLFastFiz-v0",
    entry_point="fastfiz_env.envs:BaseRLFastFiz",
)

register(
    id="PocketRLFastFiz-v0",
    entry_point="fastfiz_env.envs:PocketRLFastFiz",
)


register(
    id="EventFastFiz-v0",
    entry_point="fastfiz_env.envs:EventFastFiz",
)

register(
    id="VelocityFastFiz-v0",
    entry_point="fastfiz_env.envs:VelocityFastFiz",
)

register(
    id="SimpleFastFiz-v0",
    entry_point="fastfiz_env.envs:SimpleFastFiz",
)

register(
    id="BasicRLFastFiz-v0",
    entry_point="fastfiz_env.envs:BasicRLFastFiz",
)


register(
    id="TestingFastFiz-v0",
    entry_point="fastfiz_env.envs:TestingFastFiz",
)

register(
    id="ActionFastFiz-v0",
    entry_point="fastfiz_env.envs:ActionFastFiz",
)
