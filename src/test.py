import fastfiz_env
from fastfiz_env.utils.reward_functions import DefaultReward
from fastfiz_env.utils.reward_functions.common import (
    ConstantReward,
    ConstantWeightMaxSteps,
    NegativeConstantWeight,
    ExponentialVelocityReward,
)
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from fastfiz_env.utils.fastfiz import action_to_shot
from fastfiz_env.utils.wrappers import (
    ActionSpaces,
    FastFizActionWrapper,
)
from gymnasium import spaces
import logging

BALLS = 2


TEST_OPTIONS = {
    "seed": 99,
    "log_level": logging.WARNING,
    "logs_dir": "logs/env_test_fastfiz",
    "action_space_id": ActionSpaces.NO_OFFSET_3D,
}


env = fastfiz_env.make(
    "ActionFastFiz-v0",
    reward_function=ExponentialVelocityReward(NegativeConstantWeight),
    num_balls=BALLS,
    max_episode_steps=20,
    options=TEST_OPTIONS,
)

# env = MaxEpisodeStepsWrapper(env)
env = FastFizActionWrapper(env, ActionSpaces.NO_OFFSET_3D)
# env = TimeLimit(env, max_episode_steps=77)
# env = FlattenObservation(env)

# env = MaxEpisodeStepsWrapper(env)

# env = TimeLimit(env, max_episode_steps=87)
# print(env.get_wrapper_attr("_time_limit_max_episode_steps"))
# print(env.get_wrapper_attr("_time_limit_max_episode_steps"))
# env = MaxEpisodeStepsWrapper(env)
# print(env.get_wrapper_attr("_time_limit_max_episode_steps"))
# env = TimeLimit(env, max_episode_steps=85)


vec_env = make_vec_env(lambda: env, n_envs=1)
vec_env.reset()
# vec_env.set_attr("_max_episode_steps", env.get_wrapper_attr("_max_episode_steps"))

# print(vec_env.get_wrapper_attr("_max_episode_steps"))

# print(vec_env.get_attr("_max_episode_steps"))
# model = PPO.load(MODEL_PATH, env=env)


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
    print(action)
    action_ = env.action(action)
    print(action_)
    print(rewards)
