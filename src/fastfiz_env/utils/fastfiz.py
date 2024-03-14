import numpy as np
from typing import Optional
import fastfiz as ff

POCKETS = [
    ff.Table.SW,
    ff.Table.SE,
    ff.Table.W,
    ff.Table.E,
    ff.Table.NW,
    ff.Table.NE,
]


def get_ball_positions(table_state: ff.TableState) -> np.ndarray:
    balls = []
    for i in range(table_state.getNumBalls()):
        pos = table_state.getBall(i).getPos()
        balls.append((pos.x, pos.y))
    balls = np.array(balls)
    return balls


def num_balls_in_play(table_state: ff.TableState) -> int:
    return len([i for i in range(table_state.getNumBalls()) if table_state.getBall(i).isInPlay()])


def num_balls_pocketed(table_state: ff.TableState) -> int:
    return len([i for i in range(table_state.getNumBalls()) if table_state.getBall(i).isPocketed()])


def any_ball_has_moved(prev_ball_positions: np.ndarray, ball_positions: np.ndarray) -> bool:
    return not np.array_equal(prev_ball_positions, ball_positions)


def pocket_centers(table_state: ff.TableState) -> np.ndarray:
    table: ff.Table = table_state.getTable()
    pocket_positions = []
    for pocket in POCKETS:
        pocket_center = table.getPocketCenter(pocket)
        pocket_positions.append((pocket_center.x, pocket_center.y))

    return np.array(pocket_positions)


def distance_to_pocket(ball_position: np.ndarray, pocket: np.ndarray) -> float:
    return np.linalg.norm(pocket - ball_position)


def distance_to_pockets(ball_position: np.ndarray) -> np.ndarray:
    return np.array([distance_to_pocket(ball_position, pocket) for pocket in POCKETS])


def distance_to_closest_pocket(ball_position: np.ndarray) -> float:
    return np.min(distance_to_pockets(ball_position))


def distances_to_closest_pockets(ball_positions: np.ndarray) -> np.ndarray:
    return np.array([distance_to_closest_pocket(ball_position) for ball_position in ball_positions])


def create_table_state(n_balls: int) -> ff.TableState:
    assert 1 <= n_balls <= 15, "Number of balls must be between 1 and 15"

    game_state: ff.GameState = ff.GameState.RackedState(ff.GT_EIGHTBALL)
    table_state: ff.TableState = game_state.tableState()

    # Remove balls from table state
    for i in range(n_balls + 1, 16):
        table_state.setBall(i, ff.Ball.NOTINPLAY, ff.Point(0.0, 0.0))

    return table_state


def create_random_table_state(n_balls: int, seed: Optional[int] = None) -> ff.TableState:
    table_state = create_table_state(n_balls)
    table_state = randomize_table_state(table_state, seed)
    return table_state


def randomize_table_state(table_state: ff.TableState, seed: Optional[int] = None) -> None:
    if seed:
        np.random.seed(seed)

    table: ff.Table = table_state.getTable()
    width: float = table.TABLE_WIDTH
    length: float = table.TABLE_LENGTH

    while True:
        overlap = False
        for i in range(table_state.getNumBalls()):
            ball_i: ff.Ball = table_state.getBall(i)
            if ball_i.isInPlay():
                ball_radius: float = ball_i.getRadius()
                ball_i.setPos(ff.Point(np.random.uniform(
                    0 + ball_radius, width - ball_radius), np.random.uniform(0 + ball_radius, length - ball_radius)))
                table_state.setBall(ball_i)

                # Check overlap
                for j in range(i):
                    ball_j: ff.Ball = table_state.getBall(j)
                    if ball_j.isInPlay() and ball_j != ball_i:
                        overlap = ball_overlaps(ball_i, ball_j)
                        if overlap:
                            break
                if overlap:
                    break
        if not overlap:
            break

    return table_state


def interpolate_action(table_state: ff.TableState, action: np.ndarray) -> np.ndarray:
    a = np.interp(action[0], [0, 0], [0, 0])
    b = np.interp(action[1], [0, 0], [0, 0])
    theta = np.interp(
        action[2], [0, 1], [table_state.MIN_THETA,
                            table_state.MAX_THETA - 0.001]
    )
    phi = np.interp(action[3], [0, 1], [0, 360])
    v = np.interp(action[4], [0, 1], [0, table_state.MAX_VELOCITY])
    return [a, b, theta, phi, v]


def shot_params_from_action(table_state: ff.TableState, action: np.ndarray) -> ff.ShotParams:
    return ff.ShotParams(*interpolate_action(table_state, action))


def ball_overlaps(ball_1: ff.Ball, ball_2: ff.Ball) -> bool:
    epsilon = 1.0e-11  # avoid floating point errors (from FastFiz.cpp)
    dx = ball_1.getPos().x - ball_2.getPos().x
    dy = ball_1.getPos().y - ball_2.getPos().y

    assert ball_1.getRadius() == ball_2.getRadius(), "Balls must have the same radius"
    radius = ball_1.getRadius()

    return 4 * radius * radius - dx * dx - dy * dy > epsilon
