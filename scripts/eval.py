import argparse
import os
from typing import Optional

import fastfiz as ff
from fastfiz_renderer import GameHandler
from stable_baselines3 import PPO

from fastfiz_env.envs import PocketsFastFiz
from fastfiz_env.envs.utils import game_won, possible_shot
from fastfiz_env.utils.fastfiz import create_random_table_state
from fastfiz_env.wrappers.action import ActionSpaces, FastFizActionWrapper


def play(
    decider: GameHandler.ShotDecider,
    *,
    balls=2,
    episodes=100,
    shot_speed_factor=1.0,
):
    games = [(create_random_table_state(balls), decider) for _ in range(episodes)]
    gh = GameHandler(window_pos=(0, 0), scaling=375)
    gh.play_games(games, shot_speed_factor=shot_speed_factor)


class Agent:
    def __init__(self, model, env) -> None:
        self.model = model
        self.env = env
        self.max_shots = 10

    def decide_shot(self, table_state: ff.TableState) -> Optional[ff.ShotParams]:
        if game_won(table_state):
            print("Agent: Game Won!")
            return None

        for _ in range(self.max_shots):
            obs = self.env.compute_observation(table_state)
            action, _ = self.model.predict(obs, deterministic=True)
            action = self.env.action(action)
            shot = ff.ShotParams(*action)
            if possible_shot(table_state, shot):
                return shot

        print(f"Agent: No possible shot found in {self.max_shots} attempts.")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Path to the model file")
    parser.add_argument("-e", "--episodes", type=int, default=100, help="Number of episodes to play")
    parser.add_argument("-b", "--num-balls", type=int, default=2, help="Number of balls to play with")
    parser.add_argument("-s", "--shot-speed", type=float, default=1.0, help="Shot speed factor")
    parser.add_argument(
        "-a",
        "--action-space",
        choices=list(ActionSpaces),
        type=lambda a: ActionSpaces[a],
        default=ActionSpaces.VECTOR_3D,
        help="Action space id to use for the agent",
    )
    args = parser.parse_args()

    assert args.model is not None and os.path.exists(args.model), f"Model file not found: {args.model}"

    model = PPO.load(args.model)

    action_space = FastFizActionWrapper.SPACES[args.action_space.name]

    assert (
        model.action_space.shape == action_space.shape
    ), f"Model action space size mismatch: {model.action_space.shape} != {action_space.shape}"

    env = FastFizActionWrapper(PocketsFastFiz, args.action_space)
    agent = Agent(model, env)
    play(
        agent.decide_shot,
        balls=args.num_balls,
        episodes=args.episodes,
        shot_speed_factor=args.shot_speed,
    )


if __name__ == "__main__":
    main()
