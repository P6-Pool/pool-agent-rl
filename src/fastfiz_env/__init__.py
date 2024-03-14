"""
Gymnasium environments for pool, using FastFiz to simulate the physics of the game.

Avaliable environments:
    - `BaseFastFiz-v0`: Base class for FastFiz.
    - `BaseRLFastFiz-v0`: Base class for FastFiz with reinforcement learning, using initial random table state.
    - `PocketRLFastFiz-v0`: Subclass of BaseRLFastFiz. Observes if a ball is pocketed.


### Example

Use the environment for training a reinforcement learning agent:

```python
import gymnasium as gym
from stable_baselines3 import PPO
import fastfiz_env # Register the environments


env = gym.make("BaseRLFastFiz-v0", num_balls=2, max_episode_steps=100)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)

```

"""

from .make import make

__all__ = ["make"]

from gymnasium.envs.registration import register

register(
    id='BaseFastFiz-v0',
    entry_point="fastfiz_env.envs:BaseFastFiz",
)

register(
    id='BaseRLFastFiz-v0',
    entry_point="fastfiz_env.envs:BaseRLFastFiz",
)

register(
    id='PocketRLFastFiz-v0',
    entry_point="fastfiz_env.envs:PocketRLFastFiz",
)
