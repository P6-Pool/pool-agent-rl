from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy

# from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.ppo.policies import MlpPolicy
import fastfiz_env
from fastfiz_env.utils import CombinedReward
from fastfiz_env.utils.reward_functions.common import *
import os

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

params = {
    "n_steps": N_STEPS,
    "batch_size": BATCH_SIZE,
}


BALLS = 3
ENV_NAME = "BasicRLFastFiz-v0"
MODEL_NAME = f"ppo_recurrent-v{VERSION}-{ENV_NAME.split('FastFiz')[0].lower()}-{BALLS}_balls-{params['n_steps']}_steps-{params['batch_size']}_batch"
TB_LOGS_DIR = "logs/tb_logs/"
LOGS_DIR = f"logs/{MODEL_NAME}"
MODEL_DIR = f"models/{MODEL_NAME}/"
BEST_MODEL_DIR = f"models/{MODEL_NAME}/best/"


rewards_functions = [
    GameWonReward(),
    StepPocketedReward(),
    # DeltaBestTotalDistanceReward(),
    # ImpossibleShotReward(),
    CueBallNotMovedReward(),
    CueBallPocketedReward(),
    ConstantReward(),
]
reward_weights = [
    1,
    0.25,
    # 0.025,
    # 0,
    -1,
    -1,
    -0.025,
]


reward_function = CombinedReward(rewards_functions, reward_weights, short_circuit=True)


def make_env():
    return fastfiz_env.make(
        ENV_NAME,
        reward_function=reward_function,
        num_balls=BALLS,
        max_episode_steps=20,
        disable_env_checker=False,
    )


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

# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(
#     mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
# )

model = RecurrentPPO(
    MlpLstmPolicy,
    env,
    verbose=1,
    tensorboard_log=TB_LOGS_DIR,
    **params,
    policy_kwargs={"enable_critic_lstm": False},
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
