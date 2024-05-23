import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.tensorboard.writer import SummaryWriter

import fastfiz_env
from fastfiz_env.reward_functions.default_reward import DefaultReward
from fastfiz_env.wrappers.action import ActionSpaces

env = fastfiz_env.make_callable_wrapped_env(
    "PocketsFastFiz-v0",
    max_episode_steps=20,
    reward_function=DefaultReward,
    action_space_id=ActionSpaces.VECTOR_3D,
    num_balls=2,
)

env = make_vec_env(env, n_envs=1)

total_timesteps = 8_000_000
eval_freq = 50_000
eval_episodes = 100
total_runs = (total_timesteps // eval_freq) * eval_episodes

writer = SummaryWriter(log_dir="logs/random_policy", comment="-random-policy")

total_success = 0
total_len = 0
total_reward = 0
for episode in range(total_runs):
    obs = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()  # Random policy
        obs, reward, done, info = env.step(np.array(action))
        total_len += 1
        total_reward += reward

    total_success += int(info[0]["is_success"])


success_mean = total_success / total_runs
episode_mean = total_len / total_runs
rew_mean = total_reward / total_runs
print(f"Success rate: {success_mean}")
print(f"Mean episode length: {episode_mean}")
print(f"Mean episode reward: {rew_mean}")
for episode in range(1, total_timesteps + eval_freq - 1):
    if episode % eval_freq == 0:
        writer.add_scalar("eval/success_rate", success_mean, episode)
        writer.add_scalar("eval/mean_reward", rew_mean, episode)
        writer.add_scalar("eval/mean_ep_length", episode_mean, episode)

env.close()
writer.close()
