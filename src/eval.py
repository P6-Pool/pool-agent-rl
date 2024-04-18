import fastfiz as ff
import os
from fastfiz_renderer import GameHandler
import numpy as np
import fastfiz_env
from fastfiz_env.envs import FramesFastFiz, SimpleFastFiz, PocketsFastFiz
from fastfiz_env.reward_functions import reward_function
from fastfiz_env.reward_functions.default_reward import DefaultReward
from fastfiz_env.utils.fastfiz import (
    create_random_table_state,
    get_ball_positions,
    normalize_ball_positions,
)
from fastfiz_env.envs.utils import game_won, possible_shot
from stable_baselines3 import PPO
from typing import Optional, Callable
import argparse

from fastfiz_env.wrappers.action import ActionSpaces, FastFizActionWrapper
from fastfiz_env.wrappers.utils import spherical_coordinates


def get_play_config() -> dict:
    return {
        "auto_play": True,
        "shot_speed_factor": 1.0,
    }


def observation(table_state):
    ball_positions = positions(table_state)[:16]
    ball_positions = normalize_ball_positions(ball_positions) * 2 - 1
    observation = np.zeros((16, 2), dtype=np.float32)
    for i, ball_pos in enumerate(ball_positions):
        observation[i] = [*ball_pos]
    return np.array(observation)


def positions(table_state):
    balls = []
    for i in range(table_state.getNumBalls()):
        if table_state.getBall(i).isPocketed():
            # balls.append((0, 0))
            pass
        else:
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


# env = FramesFastFiz()
# env = SimpleFastFiz()


class Agent:
    def __init__(self, model, env) -> None:
        self.prev_ts = None
        self.model = model
        self.env = env
        self.shot = None

    def decide_shot(self, table_state: ff.TableState) -> Optional[ff.ShotParams]:
        if game_won(table_state):
            print("Agent: Game Won!")
            return None

        for _ in range(10):
            if isinstance(self.env, FramesFastFiz):
                if self.prev_ts is None:
                    obs = self.env.compute_observation(
                        table_state, table_state, self.shot
                    )
                else:
                    obs = self.env.compute_observation(
                        self.prev_ts, table_state, self.shot
                    )
            elif isinstance(self.env, PocketsFastFiz):
                obs = self.env.compute_observation(table_state)
            else:
                obs = self.env.compute_observation(table_state)
            action, _ = self.model.predict(obs, deterministic=True)
            action = self.env.action(action)
            shot = ff.ShotParams(*action)
            if possible_shot(table_state, shot):
                self.prev_ts = ff.TableState(table_state)
                ts = ff.TableState(table_state)
                self.shot = ts.executeShot(shot)
                return shot

        print("Agent: No possible shot found in 10 attempts.")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Path to the model file")
    args = parser.parse_args()

    assert args.model is not None and os.path.exists(
        args.model
    ), f"Model file not found: {args.model}"

    model = PPO.load(args.model)

    # env_vec = fastfiz_env.make("SimpleFastFiz-v0", reward_function=DefaultReward)
    # env_vec = FastFizActionWrapper(env_vec, ActionSpaces.NO_OFFSET_3D)
    env = FastFizActionWrapper(PocketsFastFiz, ActionSpaces.NO_OFFSET_3D)
    agent = Agent(model, env)
    play(agent.decide_shot, balls=2, episodes=100)


if __name__ == "__main__":
    main()
