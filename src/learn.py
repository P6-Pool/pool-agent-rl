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
from fastfiz_env.utils import DefaultReward
import os

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

MODEL_NAME = f"ppo-v{VERSION}-b{BALLS}"
TB_LOGS_DIR = "logs/tb_logs/"
LOGS_DIR = f"logs/{MODEL_NAME}"
MODEL_DIR = f"models/{MODEL_NAME}/"
BEST_MODEL_DIR = f"models/{MODEL_NAME}/best/"


def make_env():
    return fastfiz_env.make(
        "BaseRLFastFiz-v0",
        reward_function=DefaultReward,
        num_balls=BALLS,
        max_episode_steps=100,
    )


env = VecNormalize(
    make_vec_env(make_env, n_envs=4), training=True, norm_obs=True, norm_reward=True
)

model = PPO(
    MlpPolicy,
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
