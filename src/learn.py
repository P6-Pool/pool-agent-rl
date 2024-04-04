from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3 import PPO, SAC, TD3
import numpy as np
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
import fastfiz_env
from fastfiz_env.utils import CombinedReward
from fastfiz_env.utils.reward_functions.common import *
import os
import logging
from fastfiz_env.utils.wrappers import FastFizActionWrapper, ActionSpaces

ACTION_ID = ActionSpaces.NO_OFFSET_3D

# Get next version
if os.path.exists("models/"):
    versions = [
        int(d.split("-")[1].split("v")[1])
        for d in os.listdir("models/")
        if d.startswith(("ppo", "ddpg"))
    ]
    versions.append(0)
    VERSION = max(versions) + 1
else:
    VERSION = 1


N_ENVS = 4
N_STEPS = 2048 * 4
BATCH_SIZE = int(N_ENVS * N_STEPS)


MAX_EP_STEPS = 15
BALLS = 2
ENV_NAME = "ActionFastFiz-v0"
MODEL_NAME = f"ppo-v{VERSION}-{ENV_NAME.split('FastFiz')[0].lower()}-{BALLS}_balls-{ACTION_ID.name.lower()}"
TB_LOGS_DIR = "logs/tb_logs/"
LOGS_DIR = f"logs/{MODEL_NAME}"
MODEL_DIR = f"models/{MODEL_NAME}/"
BEST_MODEL_DIR = f"models/{MODEL_NAME}/best/"


rewards_functions = [
    GameWonReward(),
    ImpossibleShotReward(),
    CueBallNotMovedReward(),
    CueBallPocketedReward(),
    DeltaBestTotalDistanceReward(),
    StepPocketedReward(),
    ConstantReward(),
]
reward_weights = [
    10,
    0,
    0,
    0,
    0.025,
    (10 / BALLS),
    0,
]


reward_function = CombinedReward(rewards_functions, reward_weights, short_circuit=True)


OPTIONS = {
    # "seed": 99,
    "log_level": logging.WARNING,
    "logs_dir": "logs/env_test_fastfiz",
    "action_space_id": ACTION_ID,
}


def make_env():
    env = fastfiz_env.make(
        ENV_NAME,
        reward_function=reward_function,
        num_balls=BALLS,
        max_episode_steps=MAX_EP_STEPS,
        disable_env_checker=False,
        options=OPTIONS,
    )

    env = FastFizActionWrapper(env, ACTION_ID)
    return env


# env = VecNormalize(
#     make_vec_env(
#         make_env,
#         n_envs=4,
#         vec_env_cls=SubprocVecEnv,
#         vec_env_kwargs={"start_method": "fork"},
#     ),
#     training=True,
#     norm_obs=True,
#     norm_reward=True,
# )

env = make_vec_env(make_env, n_envs=N_ENVS)

# env = fastfiz_env.make(ENV_NAME, reward_function=reward_function, num_balls=BALLS)

action_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(env.action_space.shape),
    sigma=float(0.5) * np.ones(env.action_space.shape),
)


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=TB_LOGS_DIR,
)


checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=MODEL_DIR,
    name_prefix=MODEL_NAME,
    save_vecnormalize=True,
)


eval_callback = EvalCallback(
    eval_env=env,
    n_eval_episodes=10,
    eval_freq=25000,
    log_path=LOGS_DIR,
    best_model_save_path=BEST_MODEL_DIR,
)

callback = CallbackList([checkpoint_callback, eval_callback])

print(f"Training model: {MODEL_NAME}")
try:
    model.learn(
        total_timesteps=50_000_000,
        callback=callback,
        tb_log_name=MODEL_NAME,
        progress_bar=True,
    )

except KeyboardInterrupt:
    print(f"Training interrupted. Saving model: {MODEL_DIR + MODEL_NAME}")
    model.save(MODEL_DIR + MODEL_NAME)
else:
    print(f"Training finished. Saving model: {MODEL_DIR + MODEL_NAME}")
    model.save(MODEL_DIR + MODEL_NAME)
