# Pool Agent Reinforcement Learning

Gymnasium environment for 8-ball pool, using FastFiz to simulate the physics of the game.

## Preqrequisites

The package, `python3-opengl` is required to run the environment. Install it using the following command:

```
apt-get install python3-opengl
```

## Installation

Install the package using the following command:

```
pip install .
```

## Usage

Use the environment for training a reinforcement learning agent:

```python
from stable_baselines3 import PPO
from fastfiz_env import DefaultReward, make

env = make("SimpleFastFiz-v0", reward_function=DefaultReward, num_balls=2)

model = PPO("MlpPolicy", env)

model.learn(total_timesteps=100_000)
```
