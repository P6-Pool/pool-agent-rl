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
"""
List of pocket positions.
"""


def get_ball_positions(table_state: ff.TableState) -> np.ndarray:
    """
    Get the positions of all the balls on the table.

    Args:
        table_state (ff.TableState): The table state object representing the current state of the pool table.

    Returns:
        np.ndarray: An array containing the positions of all the balls.
    """
    balls = []
    for i in range(table_state.getNumBalls()):
        pos = table_state.getBall(i).getPos()
        balls.append((pos.x, pos.y))
    balls = np.array(balls)
    return balls


def num_balls_in_play(table_state: ff.TableState) -> int:
    """
    Returns the number of balls currently in play on table state.

    Args:
        table_state (ff.TableState): The table state object representing the current state of the pool table.

    Returns:
        int: The number of balls in play.
    """
    return len(
        [
            i
            for i in range(table_state.getNumBalls())
            if table_state.getBall(i).isInPlay()
        ]
    )


def num_balls_pocketed(table_state: ff.TableState) -> int:
    """
    Returns the number of balls pocketed in the given table state.

    Args:
        table_state (ff.TableState): The table state object representing the current state of the pool table.

    Returns:
        int: The number of balls pocketed.
    """
    return len(
        [
            i
            for i in range(table_state.getNumBalls())
            if table_state.getBall(i).isPocketed()
        ]
    )


def any_ball_has_moved(
    prev_ball_positions: np.ndarray, ball_positions: np.ndarray
) -> bool:
    """
    Check if any ball has moved by comparing the previous ball positions with the current ball positions.

    Args:
        prev_ball_positions (np.ndarray): Array of previous ball positions.
        ball_positions (np.ndarray): Array of current ball positions.

    Returns:
        bool: True if any ball has moved, False otherwise.
    """
    return not np.array_equal(prev_ball_positions, ball_positions)


def pocket_centers(table_state: ff.TableState) -> np.ndarray:
    """
    Calculates the positions of the pocket centers on the pool table.

    Args:
        table_state (ff.TableState): The table state object representing the current state of the pool table.

    Returns:
        np.ndarray: An array containing the x and y coordinates of the pocket centers.
    """
    table: ff.Table = table_state.getTable()
    pocket_positions = []
    for pocket in POCKETS:
        pocket_center = table.getPocketCenter(pocket)
        pocket_positions.append((pocket_center.x, pocket_center.y))

    return np.array(pocket_positions)


def distance_to_pocket(ball_position: np.ndarray, pocket: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between the ball position and the pocket.

    Args:
        ball_position (np.ndarray): The position of the ball.
        pocket (np.ndarray): The position of the pocket.

    Returns:
        float: The Euclidean distance between the ball position and the pocket.
    """
    return np.linalg.norm(pocket - ball_position)


def distance_to_pockets(ball_position: np.ndarray, pockets: np.ndarray) -> np.ndarray:
    """
    Calculates the distance from a given ball position to each pocket on the pool table.

    Args:
        ball_position (np.ndarray): The position of the ball on the pool table.

    Returns:
        np.ndarray: An array containing the distance from the ball position to each pocket.
    """
    return np.array([distance_to_pocket(ball_position, pocket) for pocket in pockets])


def distance_to_closest_pocket(ball_position: np.ndarray, pockets: np.ndarray) -> float:
    """
    Calculates the distance from the given ball position to the closest pocket.

    Args:
        ball_position (np.ndarray): The position of the ball.

    Returns:
        float: The distance from the ball position to the closest pocket.
    """
    return np.min(distance_to_pockets(ball_position, pockets))


def distances_to_closest_pocket(
    ball_positions: np.ndarray, pockets: np.ndarray
) -> np.ndarray:
    """
    Calculates the distances from each ball position to the closest pocket.

    Args:
        ball_positions (np.ndarray): An array of ball positions.
        pockets (np.ndarray): An array of pocket positions.

    Returns:
        np.ndarray: An array of distances from each ball position to the closest pocket.
    """
    return np.array(
        [
            distance_to_closest_pocket(ball_position, pockets)
            for ball_position in ball_positions
        ]
    )


def create_table_state(n_balls: int) -> ff.TableState:
    """
    Creates a table state with the specified number of balls.

    Args:
        n_balls (int): The number of balls to include in the table state. Must be between 1 and 16.

    Returns:
        ff.TableState: The created table state.

    Raises:
        AssertionError: If the number of balls is not between 0 and 16.
    """

    assert 0 <= n_balls <= 16, "Number of balls must be between 1 and 15"

    game_state: ff.GameState = ff.GameState.RackedState(ff.GT_EIGHTBALL)
    table_state: ff.TableState = game_state.tableState()

    # Remove balls from table state
    for i in range(n_balls, 16):
        table_state.setBall(i, ff.Ball.NOTINPLAY, ff.Point(0.0, 0.0))

    return table_state


def create_random_table_state(
    n_balls: int, seed: Optional[int] = None
) -> ff.TableState:
    """
    Creates a random table state with the specified number of balls.

    Args:
        n_balls (int): The number of balls to include in the table state. Must be between 0 and 16.
        seed (Optional[int]): The seed value to use for random number generation. If not provided, the random number generator will not be seeded.

    Returns:
        ff.TableState: The randomly generated table state.
    """
    table_state = create_table_state(n_balls)
    table_state = randomize_table_state(table_state, seed)
    return table_state


def randomize_table_state(
    table_state: ff.TableState, seed: Optional[int] = None
) -> None:
    """
    Randomizes the positions of the balls on the pool table within the given table state.

    Args:
        table_state (ff.TableState): The table state object representing the current state of the pool table.
        seed (Optional[int]): The seed value to use for random number generation. If not provided, the random number generator will not be seeded.

    Raises:
        RuntimeError: If the function fails to randomize the table state after 100 attempts.

    Returns:
        ff.TableState: The randomized table state.
    """
    if seed:
        np.random.seed(seed)

    table: ff.Table = table_state.getTable()
    width: float = table.TABLE_WIDTH
    length: float = table.TABLE_LENGTH

    tries = 0
    while True:
        overlap = False
        for i in range(table_state.getNumBalls()):
            ball_i: ff.Ball = table_state.getBall(i)
            if ball_i.isInPlay():
                ball_radius: float = ball_i.getRadius()
                ball_i.setPos(
                    ff.Point(
                        np.random.uniform(0 + ball_radius, width - ball_radius),
                        np.random.uniform(0 + ball_radius, length - ball_radius),
                    )
                )
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
        tries += 1
        if tries > 100:
            raise RuntimeError("Could not randomize table state.")
        if not overlap:
            break

    return table_state


def map_action_to_shot_params(
    table_state: ff.TableState, action: np.ndarray
) -> np.ndarray:
    """
    Maps the given action values to the corresponding shot parameters within the specified ranges.

    Args:
        table_state (ff.TableState): The table state object containing the minimum and maximum values for mapping.
        action (np.ndarray): The action values to be mapped.

    Returns:
        np.ndarray: The mapped shot parameters.

    """
    a = np.interp(action[0], [0, 0], [0, 0])
    b = np.interp(action[1], [0, 0], [0, 0])
    theta = np.interp(
        action[2], [0, 1], [table_state.MIN_THETA, table_state.MAX_THETA - 0.001]
    )
    phi = np.interp(action[3], [0, 1], [0, 360])
    v = np.interp(action[4], [0, 1], [0, table_state.MAX_VELOCITY])
    return [a, b, theta, phi, v]


def shot_params_from_action(
    table_state: ff.TableState, action: np.ndarray
) -> ff.ShotParams:
    """
    Converts an action into shot parameters.

    Args:
        table_state (ff.TableState): The current state of the pool table.
        action (np.ndarray): The action to be converted.

    Returns:
        ff.ShotParams: The shot parameters corresponding to the given action.
    """
    return ff.ShotParams(*map_action_to_shot_params(table_state, action))


def ball_overlaps(ball_a: ff.Ball, ball_b: ff.Ball) -> bool:
    """
    Check if two balls overlap.

    Args:
        ball_a (ff.Ball): The first ball.
        ball_b (ff.Ball): The second ball.

    Returns:
        bool: True if the balls overlap, False otherwise.
    """
    # Replica of the original fastfiz implementation (from FastFiz.cpp)
    epsilon = 1.0e-11  # avoid floating point errors
    dx = ball_a.getPos().x - ball_b.getPos().x
    dy = ball_a.getPos().y - ball_b.getPos().y

    assert ball_a.getRadius() == ball_b.getRadius(), "Balls must have the same radius"
    radius = ball_a.getRadius()

    return 4 * radius * radius - dx * dx - dy * dy > epsilon
