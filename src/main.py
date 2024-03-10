import argparse
import enum
import logging
from train import train, create_model
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MultiInputPolicy
from utils import ModelManager
from environment import register_env


class Mode(enum.Enum):
    TRAIN = "train"
    PLAY = "play"


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pool Agent CLI")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    model_manager = ModelManager("ppo", "models/", "logs/")

    train_parser = subparsers.add_parser(
        Mode.TRAIN.value, help="Train a model")
    train_parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        required=True,
        help="Number of timesteps to train for",
    )
    train_parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        required=True,
        help="Number of episodes to train for",
    )
    train_parser.add_argument(
        "-m", "--model", type=str, help="Path to pretrained model to train from"
    )
    train_parser.add_argument(
        "-v",
        "--version",
        type=int,
        help="Train from pretrained model version. WARNING: This will overwrite existing models",
    )
    train_parser.add_argument(
        "-b",
        "--balls",
        type=int,
        help="Amount of balls on table",
    )

    args = parser.parse_args()

    if args.balls:
        if args.balls < 2 and args.balls > 16:
            raise ValueError("Amount of balls must be between 2 and 16")
        balls = args.balls
    else:
        balls = 2
    env = register_env({"render_mode": "human", "n_balls_train": balls})

    # Print options
    match args.mode:
        case _:
            pass
    if args.mode == Mode.TRAIN.value:
        print(
            f'Training mode with {args.timesteps} timestep{"s" if args.timesteps != 1 else ""}, {args.episodes} episode{"s" if args.episodes != 1 else ""}'
        )

        print(f"NEXT VER: {model_manager.next_version()}")
        model_manager.name_handler.version = model_manager.next_version()

        if args.model:
            model = PPO.load(
                args.model, env=env, verbose=1, tensorboard_log=model_manager.logs_dir
            )
        else:
            model = create_model(env, model_manager.logs_dir)

        train(
            args.timesteps,
            args.episodes,
            model,
            args.version,
            model_manager=model_manager,
        )


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    main()
