from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import fastfiz_env
from fastfiz_env.utils import DefaultReward, CombinedReward
from fastfiz_env.utils.reward_functions.common import *
import os
from torch import nn as nn

params = {
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.9999,
    "learning_rate": 0.0007350588182212963,
    "ent_coef": 0.00014455517747388256,
    "clip_range": 0.2,
    "n_epochs": 1,
    "gae_lambda": 0.92,
    "max_grad_norm": 0.3,
    "vf_coef": 0.9179505500616013,
    "sde_sample_freq": 256,
    "policy_kwargs": dict(
        log_std_init=0.6144189947543124,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=nn.Tanh,
        ortho_init=False,
    ),
}


# Get next version
if os.path.exists("models/"):
    versions = [
        int(d.split("-")[1].split("v")[1])
        for d in os.listdir("models/")
        if d.startswith("ppo")
    ]
    versions.append(0)
    VERSION = max(versions) + 1
else:
    VERSION = 1

BALLS = 2

MODEL_NAME = f"ppo-v{VERSION}-b{BALLS}-vel-norm"
TB_LOGS_DIR = "logs/tb_logs/"
LOGS_DIR = f"logs/{MODEL_NAME}"
MODEL_DIR = f"models/{MODEL_NAME}/"
BEST_MODEL_DIR = f"models/{MODEL_NAME}/best/"


rewards_functions = [
    StepPocketedReward(),
    BestTotalDistanceReward(),
    GameWonReward(),
    ImpossibleShotReward(),
    CueBallNotMovedReward(),
    CueBallPocketedReward(),
    ConstantReward(),
]
reward_weights = [1, 0.5, 10, -10, -10, -10, -0.1]


reward_function = CombinedReward(rewards_functions, reward_weights, short_circuit=True)


def make_env():
    return fastfiz_env.make(
        "VelocityFastFiz-v0",
        reward_function=DefaultReward,
        num_balls=BALLS,
        max_episode_steps=100,
        disable_env_checker=False,
    )


env = VecNormalize(
    make_vec_env(make_env, n_envs=2), training=True, norm_obs=True, norm_reward=True
)

# env = make_env()

model = PPO(
    MlpPolicy,
    env,
    verbose=1,
    tensorboard_log=TB_LOGS_DIR,
    # **params,
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
    eval_freq=2500,
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
    )
except KeyboardInterrupt:
    print(f"Training interrupted. Saving model: {MODEL_DIR + MODEL_NAME}")
    model.save(MODEL_DIR + MODEL_NAME)
else:
    print(f"Training finished. Saving model: {MODEL_DIR + MODEL_NAME}")
    model.save(MODEL_DIR + MODEL_NAME)
