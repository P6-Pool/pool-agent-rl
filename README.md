# Pool Agent Reinforcement Learning

Gymnasium environments for 8-ball pool, using FastFiz to simulate the physics of the game.

## Installation

Install the package using the following command:

```
pip install -e .
```

## Usage

Use the environment for training a reinforcement learning agent:

```python
from stable_baselines3 import PPO
import fastfiz_env
from fastfiz_env.utils import DefaultReward

env = fastfiz_env.make("BaseRLFastFiz-v0", reward_function=DefaultReward, num_balls=2)

model = PPO("MlpPolicy", env)

model.learn(total_timesteps=10_000)
```
