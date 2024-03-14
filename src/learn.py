from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import fastfiz_env
from fastfiz_env.utils.reward_functions.common import *
from fastfiz_env.utils.reward_functions import CombinedReward
import os

# Get next version
if os.path.exists("models/"):
    versions = [int(d.split("-")[1].split("v")[1])
                for d in os.listdir("models/") if d.startswith("ppo")]
    versions.append(0)
    VERSION = max(versions) + 1
else:
    VERSION = 1

BALLS = 2

MODEL_NAME = f"ppo-v{VERSION}-b{BALLS}-2x32"
TB_LOGS_DIR = "logs/tb_logs/"
LOGS_DIR = f"logs/{MODEL_NAME}"
MODEL_DIR = f"models/{MODEL_NAME}/"
BEST_MODEL_DIR = f"models/{MODEL_NAME}/best/"


rewards = [StepPocketedReward(), BestTotalDistanceReward(), GameWonReward(
), ImpossibleShotReward(), CueBallNotMovedReward(), CueBallPocketedReward(), ConstantReward()]
weights = [5, 1, 100, -100, -100, -100, -1]

reward_fn = CombinedReward(rewards, weights)


def make_env():
    return fastfiz_env.make("BaseRLFastFiz-v0", reward_function=reward_fn, num_balls=BALLS, max_episode_steps=100)


net_arch = dict(pi=[32, 32], vf=[32, 32])

env = VecNormalize(make_vec_env(make_env, n_envs=4),
                   training=True, norm_obs=True, norm_reward=True)

model = PPO(MlpPolicy, env, learning_rate=0.0001,
            verbose=1, tensorboard_log=TB_LOGS_DIR,
            policy_kwargs={"net_arch": net_arch},
            use_sde=True)


checkpoint_callback = CheckpointCallback(name_prefix=MODEL_NAME,
                                         save_freq=50_000, save_path=MODEL_DIR, save_vecnormalize=True)


eval_callback = EvalCallback(eval_env=env, best_model_save_path=BEST_MODEL_DIR, n_eval_episodes=10,
                             log_path=LOGS_DIR, eval_freq=10_000, deterministic=True, render=False)

print(f"Training model: {MODEL_NAME}")
try:
    model.learn(total_timesteps=50_000_000, callback=[
        checkpoint_callback, eval_callback], tb_log_name=MODEL_NAME)
except KeyboardInterrupt:
    print(f"Training interrupted. Saving model: {MODEL_DIR + MODEL_NAME}")
    model.save(MODEL_DIR + MODEL_NAME)
else:
    print(f"Training finished. Saving model: {MODEL_DIR + MODEL_NAME}")
    model.save(MODEL_DIR + MODEL_NAME)
