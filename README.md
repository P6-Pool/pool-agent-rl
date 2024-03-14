# Pool Agent Reinforcement Learning

Gymnasium environments for pool, using FastFiz to simulate the physics of the game.

## Installation

Install the package using the following command:

```
pip install .
```

## Usage

Use the environment for training a reinforcement learning agent:

```python
from stable_baselines3 import PPO
import fastfiz_env
from fastfiz_env.utils.reward_functions.common import StepPocketedReward

reward_function = StepPocketedReward()
env = fastfiz_env.make("BaseRLFastFiz-v0", reward_function=reward_function, num_balls=2, max_episode_steps=100)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)

```
