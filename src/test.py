import fastfiz_env
from fastfiz_env.utils.reward_functions import DefaultReward
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from fastfiz_env.utils.fastfiz import action_to_shot
from gymnasium import spaces
import logging

BALLS = 4


TEST_OPTIONS = {
    "seed": 99,
    "log_level": logging.INFO,
    "logs_dir": "logs/env_test_fastfiz",
}


env = fastfiz_env.make(
    "TestingFastFiz-v0",
    reward_function=DefaultReward,
    num_balls=BALLS,
    max_episode_steps=100,
    test_options=TEST_OPTIONS,
)


# model = PPO.load(MODEL_PATH, env=env)

vec_env = make_vec_env(lambda: env, n_envs=1)


# action_space = spaces.Box(
#     low=np.array([0, 0, -1, -1, -1], dtype=np.float32),
#     high=np.array([0, 0, 1, 1, 1], dtype=np.float32),
# )


# action_to_shot([0, 0, 0.65, 0.5, 0.5], action_space)


# exit(0)

# vec_env = model.get_env()


# for _ in range(10):
#     done = False
#     obs = vec_env.reset()
#     while not done:
#         action, _states = model.predict(obs[0])
#         obs, rewards, done, info = vec_env.step(action)
#     print(rewards)


# action, _states = model.predict(obs)
# a, b,

# action = np.array([[0.0, 0.0, 0.01, 0, 1]], dtype=np.float64)
# action = [0.0, 0.0, 0.15, 0.6, 0.3]
for _ in range(10):
    obs = vec_env.reset()
    action = vec_env.action_space.sample()
    obs, rewards, done, info = vec_env.step([action])
