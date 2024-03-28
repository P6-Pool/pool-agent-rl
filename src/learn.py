from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.ppo.policies import MlpPolicy
import fastfiz_env
from fastfiz_env.utils import CombinedReward
from fastfiz_env.utils.reward_functions.common import *
import os

n_envs = 1
n_steps = 5120
batch_size = int((n_steps * n_envs) / 2) 

params = {
    "n_steps": n_steps,
    "batch_size": batch_size,
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

ENV_ID = "VelocityFastFiz-v0"
BALLS = 2

MODEL_NAME = f"ppo-v{VERSION}-b{BALLS}-{ENV_ID}"
TB_LOGS_DIR = "logs/tb_logs/"
LOGS_DIR = f"logs/{MODEL_NAME}"
MODEL_DIR = f"models/{MODEL_NAME}/"
BEST_MODEL_DIR = f"models/{MODEL_NAME}/best/"


rewards_functions = [
    StepPocketedReward(),
    DeltaBestTotalDistanceReward(),
    GameWonReward(),
    ImpossibleShotReward(),
    CueBallNotMovedReward(),
    CueBallPocketedReward(),
    ConstantReward(),
]
reward_weights = [
    5,
    0.2,
    10,
    -10,
    -10,
    -10,
    -0.1,
]


reward_function = CombinedReward(rewards_functions, reward_weights, short_circuit=True)

def make_env(env_id):
    return fastfiz_env.make(
        env_id,
        reward_function=reward_function,
        num_balls=BALLS,
        max_episode_steps=250
    )

n_envs = 4

env = make_env(ENV_ID)

env = VecNormalize(make_vec_env(ENV_ID, n_envs=n_envs))

n_steps = 20480
batch_size = int((n_steps * n_envs) / 4)

params = {
    "n_steps": n_steps,
    "batch_size": batch_size,
}

model = PPO(
    MlpPolicy,
    env,
    verbose=1,
    tensorboard_log=TB_LOGS_DIR,
    **params,
    policy_kwargs=dict(enable_critic_lstm=False),
)

save_freq = 50_000
eval_freq = 25_000

checkpoint_callback = CheckpointCallback(
    save_freq=max(save_freq // n_envs, 1),
    save_path=MODEL_DIR,
    name_prefix=MODEL_NAME,
    save_vecnormalize=True,
)

eval_callback = EvalCallback(
    eval_env=env,
    n_eval_episodes=10,
    eval_freq=max(eval_freq // n_envs, 1),
    log_path=LOGS_DIR,
    best_model_save_path=BEST_MODEL_DIR,
)

callback = CallbackList([checkpoint_callback, eval_callback])

print(f"Training model: {MODEL_NAME}")
try:
    model.learn(
        total_timesteps=10_000_000,
        callback=callback,
        tb_log_name=MODEL_NAME,
        reset_num_timesteps=False,
        progress_bar=True,
    )

except KeyboardInterrupt:
    print(f"Training interrupted. Saving model: {MODEL_DIR + MODEL_NAME}")
    model.save(MODEL_DIR + MODEL_NAME)
else:
    print(f"Training finished. Saving model: {MODEL_DIR + MODEL_NAME}")
    model.save(MODEL_DIR + MODEL_NAME)

