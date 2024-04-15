from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3 import PPO
import fastfiz_env
from fastfiz_env.reward_functions import DefaultReward
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
MAX_EP_STEPS = 10
BALLS = 2
N_ENVS = 4
ACTION_ID = ActionSpaces.NO_OFFSET_3D

# Paths
ENV_NAME = "SimpleFastFiz-v0"
MODEL_NAME = f"ppo-v{VERSION}-{ENV_NAME.split('FastFiz')[0].lower()}-{BALLS}_balls-{ACTION_ID.name.lower()}"
TB_LOGS_DIR = "logs/tb_logs/"
LOGS_DIR = f"logs/{MODEL_NAME}"
MODEL_DIR = f"models/{MODEL_NAME}/"
BEST_MODEL_DIR = f"models/{MODEL_NAME}/best/"


hyperparameters = {
    "batch_size": 256,
    "n_steps": 256,
    "gamma": 0.9,
    "learning_rate": 0.000145540594755511612,
    "ent_coef": 0.04088822213828693,
    "clip_range": 0.4,
    "n_epochs": 1,
    "gae_lambda": 0.92,
    "max_grad_norm": 2,
    "vf_coef": 0.10074536077062124,
    "sde_sample_freq": 8,
    "policy_kwargs": {
        "log_std_init": -0.36854877110355,
        "net_arch": dict(pi=[64], vf=[64]),
        "activation_fn": ReLU,
        "ortho_init": False,
    },
}


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
    **hyperparameters,
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
