import fastfiz as ff
import os
from fastfiz_renderer import GameHandler
import numpy as np
from fastfiz_env.utils.fastfiz import (
    create_random_table_state,
    get_ball_positions,
    normalize_ball_positions,
)
from fastfiz_env.envs.utils import game_won, possible_shot
from stable_baselines3 import PPO
from typing import Optional, Callable
import argparse

from fastfiz_env.wrappers.action import FastFizActionWrapper
from fastfiz_env.wrappers.utils import spherical_coordinates


def get_play_config() -> dict:
    return {
        "auto_play": True,
        "shot_speed_factor": 1.0,
    }


def observation(table_state):
    ball_positions = positions(table_state)[:16]
    ball_positions = normalize_ball_positions(ball_positions) * 2 - 1
    observation = np.zeros((16, 3), dtype=np.float32)
    for i, ball_pos in enumerate(ball_positions):
        observation[i] = [*ball_pos, int(table_state.getBall(i).isPocketed())]
    return np.array(observation)


def positions(table_state):
    balls = []
    for i in range(table_state.getNumBalls()):
        # if table_state.getBall(i).isPocketed():
        #     # balls.append((0, 0))
        #     pass
        # else:
        pos = table_state.getBall(i).getPos()
        balls.append((pos.x, pos.y))
    balls = np.array(balls)
    return balls


def play(
    decider: GameHandler.ShotDecider,
    *,
    balls=2,
    episodes=100,
):
    config = get_play_config()

    games = [(create_random_table_state(balls), decider) for _ in range(episodes)]
    gh = GameHandler(window_pos=(0, 0), scaling=375)
    gh.play_games(
        games,
        auto_play=config["auto_play"],
        shot_speed_factor=config["shot_speed_factor"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Path to the model file")
    args = parser.parse_args()

    assert args.model is not None and os.path.exists(
        args.model
    ), f"Model file not found: {args.model}"

    model = PPO.load(args.model)

    def decide_shot(table_state: ff.TableState) -> Optional[ff.ShotParams]:
        if game_won(table_state):
            print("Agent: Game Won!")
            return None
        obs = observation(table_state)
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=True)
            if np.allclose(action, 0):
                shot = ff.ShotParams(*[0, 0, 0, 0, 0])
            else:
                r, theta, phi = spherical_coordinates(action)
                theta = np.interp(theta, (0, 360), (0, 70 - 0.001))
                phi = np.interp(phi, (0, 360), (0, 360))
                velocity = np.interp(r, (0, np.sqrt(3)), (0, 10))
                shot = ff.ShotParams(*[0, 0, theta, phi, velocity])
            if possible_shot(table_state, shot):
                return shot

        print("Agent: No possible shot found in 10 attempts.")
        return None

    play(decide_shot, balls=3, episodes=100)


if __name__ == "__main__":
    main()
