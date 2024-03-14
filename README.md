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
import gymnasium as gym
from stable_baselines3 import PPO
import fastfiz_env


env = gym.make("BaseRLFastFiz-v0", num_balls=2, max_episode_steps=100)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)

```
