from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3 import PPO
import fastfiz_env
from fastfiz_env.reward_functions import DefaultReward, WinningReward
from fastfiz_env.reward_functions.common import *
import os
from fastfiz_env.wrappers import FastFizActionWrapper, ActionSpaces
from torch.nn import ReLU


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


# Settings
MAX_EP_STEPS = 100
BALLS = 2
N_ENVS = 4
ACTION_ID = ActionSpaces.NO_OFFSET_3D

# Paths
ENV_NAME = "FramesFastFiz-v0"
MODEL_NAME = f"ppo-v{VERSION}-{ENV_NAME.split('FastFiz')[0].lower()}-{BALLS}_balls-{ACTION_ID.name.lower()}"
TB_LOGS_DIR = "logs/tb_logs/"
LOGS_DIR = f"logs/{MODEL_NAME}"
MODEL_DIR = f"models/{MODEL_NAME}/"
BEST_MODEL_DIR = f"models/{MODEL_NAME}/best/"


def make_env():
    env = fastfiz_env.make(
        ENV_NAME,
        reward_function=DefaultReward,
        num_balls=BALLS,
        max_episode_steps=MAX_EP_STEPS,
        disable_env_checker=False,
    )
    env = FastFizActionWrapper(env, ACTION_ID)
    return env


env = make_vec_env(make_env, n_envs=N_ENVS)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=TB_LOGS_DIR,
    device="cpu",
    # **hyperparameters,
)


checkpoint_callback = CheckpointCallback(
    save_freq=int(50_000 / N_ENVS),
    save_path=MODEL_DIR,
    name_prefix=MODEL_NAME,
    save_vecnormalize=True,
)

eval_callback = EvalCallback(
    eval_env=env,
    n_eval_episodes=10,
    eval_freq=int(10_000 / N_ENVS),
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
