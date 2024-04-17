import argparse
import gymnasium as gym
import optuna
import time
import torch
import torch.nn as nn
from fastfiz_env.make import make_wrapped_env, make_wrapped_vec_env
from fastfiz_env.reward_functions import DefaultReward, WinningReward, RewardFunction
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from typing import Any, Dict


# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py
def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical(
        "n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical(
        "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
    )
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_float("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    # lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch_type]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn_name]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


# https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py
class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(
    trial: optuna.Trial,
    env_id: str,
    num_balls: int,
    max_episode_steps: int,
    reward_function: RewardFunction,
    n_eval_episodes: int,
    eval_freq: int,
    n_timesteps: int,
    start_time: str,
) -> float:
    kwargs = sample_ppo_params(trial)
    model = PPO(
        "MlpPolicy",
        env=make_wrapped_vec_env(
            env_id, num_balls, max_episode_steps, 4, reward_function
        ),
        **kwargs,
        tensorboard_log=f"logs/trials_test",
    )

    eval_env = Monitor(
        make_wrapped_env(env_id, num_balls, max_episode_steps, reward_function)
    )

    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
        deterministic=True,
    )

    nan_encountered = False
    try:
        model.learn(
            n_timesteps,
            callback=eval_callback,
            tb_log_name=f"trial_{trial.number}_run_{start_time}",
        )
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    except Exception as e:
        print(e)
    finally:
        # Free memory.
        model.env.close()  # type: ignore
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials")
    parser.add_argument(
        "--n_startup_trials", type=int, default=5, help="Number of startup trials"
    )
    parser.add_argument(
        "--reward",
        type=str,
        choices=["DefaultReward", "WinningReward"],
        default="DefaultReward",
        help="Reward function",
    )
    parser.add_argument(
        "--n_timesteps", type=int, default=int(5e5), help="Number of timesteps"
    )
    parser.add_argument(
        "--num-balls", type=int, default=2, help="Number of balls in the environment"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=int(5e5 / 3), help="Evaluation frequency"
    )
    parser.add_argument(
        "--n_eval_episodes", type=int, default=100, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="SimpleFastFiz-v0",
        help="Environment ID",
        required=True,
    )
    parser.add_argument("--max_episode_steps", type=int, default=20)
    args = parser.parse_args()

    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)
    n_startup_trials = 5
    sampler = TPESampler(n_startup_trials=n_startup_trials)
    pruner = MedianPruner(n_startup_trials=n_startup_trials)

    print(f"Start optimization with {args.n_trials} trials.")

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    start_time = time.strftime("%Y_%m_%d-%H:%M:%S")

    reward_function = DefaultReward if args.reward == "DefaultReward" else WinningReward

    obj_fn = lambda trial: objective(
        trial,
        args.env_id,
        args.num_balls,
        args.max_episode_steps,
        reward_function,
        args.n_eval_episodes,
        args.eval_freq,
        args.n_timesteps,
        start_time,
    )

    try:
        study.optimize(obj_fn, n_trials=args.n_trials, timeout=3600)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))