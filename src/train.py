import argparse
import glob
import json
import os
from fastfiz_env.make import make_callable_wrapped_env
from fastfiz_env.reward_functions import RewardFunction
from typing import Optional
from stable_baselines3 import PPO
from fastfiz_env.reward_functions import DefaultReward, WinningReward
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.env_util import make_vec_env
from fastfiz_env.wrappers.action import ActionSpaces
from hyperparams import params_to_kwargs


def get_latest_run_id(log_path: str, name: str) -> int:
    id = 0
    for path in glob.glob(os.path.join(log_path, name + "_[0-9]*")):
        run_id = path.split("_")[-1]
        path_without_run_id = path[: -len(run_id) - 1]
        if path_without_run_id.endswith(name) and run_id.isdigit() and int(run_id) > id:
            id = int(run_id)
    return id


def get_model_name(env_name: str, balls: int, algo: str = "PPO", action_space_id=ActionSpaces.VECTOR_3D) -> str:
    return f"{env_name.split('FastFiz-v0')[0]}-{balls}_balls-{action_space_id.name}-{algo}".lower()


def train(
    env_id: str,
    num_balls: int,
    max_episode_steps: int = 20,
    n_envs: int = 4,
    model_dir: Optional[str] = None,
    total_timesteps=10_000_000,
    logs_path: str = "logs/",
    models_path: str = "models/",
    reward_function: RewardFunction = DefaultReward,
    action_space_id: ActionSpaces = ActionSpaces.VECTOR_3D,
    callbacks=None,
    params: Optional[dict] = None,
) -> None:
    env = make_vec_env(
        make_callable_wrapped_env(env_id, num_balls, max_episode_steps, reward_function, action_space_id=action_space_id),
        n_envs=n_envs,
    )

    hyperparams = params_to_kwargs(**params) if params else {}
    print(hyperparams)
    model_name = get_model_name(env_id, num_balls, action_space_id=action_space_id)

    if model_dir is None:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_path, **hyperparams)
    else:
        model = PPO.load(model_dir, env=env, verbose=1, tensorboard_log=logs_path)
        pretrained_name = model_dir.split("/")[-1].rsplit(".zip", 1)[0]
        model_name = f"{model_name}-from_{pretrained_name}"

    id = get_latest_run_id(logs_path, model_name) + 1
    model_dir = os.path.join(models_path, f"{model_name}_{id}")
    model_path = os.path.join(model_dir, f"{model_name}_{id}")

    checkpoint_callback = CheckpointCallback(
        save_freq=int(50_000 / n_envs),
        name_prefix=model_name,
        save_path=model_dir,
    )

    eval_callback = EvalCallback(
        eval_env=env,
        n_eval_episodes=100,
        eval_freq=int(50_000 / n_envs),
        best_model_save_path=model_dir,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=model_name,
            progress_bar=True,
        )
        print("Training finished.")
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        model.save(model_path)
        print(f"Model saved: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("-b", "--num_balls", type=int, required=True)
    parser.add_argument("-m", "--max_episode_steps", type=int, default=20)
    parser.add_argument("-n", "--n_time_steps", type=int, default=1_000_000)
    parser.add_argument(
        "-i",
        "--model_path",
        type=str,
        default=None,
        help="Path to pretrained model to continue training on",
    )
    parser.add_argument("-l", "--logs_path", type=str, default="logs/")
    parser.add_argument("-d", "--models_path", type=str, default="models/")
    parser.add_argument(
        "--reward",
        type=str,
        choices=["DefaultReward", "WinningReward"],
        default="DefaultReward",
    )

    # Hyper params
    parser.add_argument(
        "--params",
        type=str,
        help="Path to hyperparameters file (file must have key 'params' with dict of hyperparameters",
    )

    parser.add_argument(
        "-a", "--action_id", type=lambda a: ActionSpaces[a], choices=list(ActionSpaces), default=ActionSpaces.VECTOR_3D
    )

    args = parser.parse_args()

    reward_function = DefaultReward if args.reward == "DefaultReward" else WinningReward

    env_id = args.env
    num_balls = args.num_balls
    assert 1 <= num_balls <= 16, "Number of balls must be between 1 and 16"
    max_episode_steps = args.max_episode_steps
    model_path = args.model_path
    total_timesteps = args.n_time_steps
    logs_path = args.logs_path
    models_path = args.models_path
    reward = args.reward
    params = None
    if args.params:
        params_path = args.params
        assert os.path.exists(logs_path), f"params path does not exist: {logs_path}"
        with open(params_path, "r") as fp:
            params = json.load(fp)
        assert "params" in params, "params file must have key 'params' with dict of hyperparameters"
        params = params["params"]

    print(
        f"Starting training on {env_id} with following settings:\n\
          num_balls: {num_balls}\n\
          max_episode_steps: {max_episode_steps}\n\
          total_timesteps: {total_timesteps}\n\
          model_path: {model_path}\n\
          logs_path: {logs_path}\n\
          models_path: {models_path}\n\
          reward_function: {reward}\n\
          action_space_id: {args.action_id}\n"
    )

    train(
        env_id,
        num_balls,
        max_episode_steps,
        model_dir=model_path,
        total_timesteps=total_timesteps,
        logs_path=logs_path,
        models_path=models_path,
        reward_function=reward_function,
        action_space_id=args.action_id,
        params=params,
    )
