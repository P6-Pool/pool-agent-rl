import numpy as np
from typing import Optional
import fastfiz as ff
from gymnasium import spaces

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


def num_balls_pocketed(
    table_state: ff.TableState,
    *,
    range_start: int = 0,
    range_stop: Optional[int] = None,
) -> int:
    """
    Returns the number of balls pocketed in the given table state.

    Args:
        table_state (ff.TableState): The table state object representing the current state of the pool table.
        range_start (int): The starting index of the range of balls to check. Defaults to 0.
        range_stop (Optional[int]): The stopping index of the range of balls to check. Defaults to 16 (`table_state.getNumBalls()`).

    Returns:
        int: The number of balls pocketed.
    """
    stop = table_state.getNumBalls() if range_stop is None else range_stop
    return len(
        [i for i in range(range_start, stop) if table_state.getBall(i).isPocketed()]
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
        action[2], [-1, 1], [table_state.MIN_THETA, table_state.MAX_THETA - 0.001]
    )
    phi = np.interp(action[3], [-1, 1], [0, 360])
    v = np.interp(action[4], [-1, 1], [0, table_state.MAX_VELOCITY - 0.001])
    return np.array([0, 0, theta, phi, v], dtype=np.float64)


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


def action_to_shot(action: np.ndarray, action_space: spaces.Box) -> ff.ShotParams:
    """
    Maps the given action values to the corresponding action space.
    """
    MIN_OFFSET = 0  # -1?
    MAX_OFFSET = 0  # 1?
    MIN_PHI = 0
    MAX_PHI = 360
    MIN_THETA = ff.TableState.MIN_THETA
    MAX_THETA = ff.TableState.MAX_THETA
    MAX_VELOCITY = ff.TableState.MAX_VELOCITY

    a = np.interp(
        action[0], [action_space.low[0], action_space.high[0]], [MIN_OFFSET, MAX_OFFSET]
    )
    b = np.interp(
        action[1], [action_space.low[1], action_space.high[1]], [MIN_OFFSET, MAX_OFFSET]
    )
    theta = np.interp(
        action[2], [action_space.low[2], action_space.high[2]], [MIN_THETA, MAX_THETA]
    )
    phi = np.interp(
        action[3], [action_space.low[3], action_space.high[3]], [MIN_PHI, MAX_PHI]
    )
    velocity = np.interp(
        action[4], [action_space.low[4], action_space.high[4]], [0, MAX_VELOCITY]
    )

    # print(f"a: {a}, b: {b}, theta: {theta}, phi: {phi}, velocity: {velocity}")

    return ff.ShotParams(a, b, theta, phi, velocity)


def normalize_ball_positions(
    ball_positions: np.ndarray[float, np.dtype[np.float32]]
) -> np.ndarray[float, np.dtype[np.float32]]:
    """
    Normalize the ball positions to be within the range [0, 1].

    Args:
        ball_positions (np.ndarray): The ball positions to be normalized.

    Returns:
        np.ndarray: The normalized ball positions.
    """
    width: float = ff.Table.TABLE_WIDTH
    length: float = ff.Table.TABLE_LENGTH

    return ball_positions / np.array([width, length])


def get_ball_velocity(ball: ff.Ball) -> float:
    """
    Get the velocity of the given ball.

    Args:
        ball (ff.Ball): The ball.

    Returns:
        np.ndarray: The velocity of the ball.
    """
    vel = ball.getVelocity()
    return np.hypot(vel.x, vel.y)


def normalize_ball_velocity(velocity: float) -> np.ndarray:
    scale = 1.580  # Estimated maximum velocity of a ball scale
    max_velocity = ff.TableState.MAX_VELOCITY * scale
    return velocity / max_velocity


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


def get_ball_positions_id(table_state: ff.TableState) -> np.ndarray:
    """
    Get the positions of all the balls on the table.

    Args:
        table_state (ff.TableState): The table state object representing the current state of the pool table.

    Returns:
        np.ndarray: An array containing the positions of all the balls.
    """
    balls = []
    for i in range(table_state.getNumBalls()):
        ball = table_state.getBall(i)
        pos = ball.getPos()
        balls.append([ball.getID(), pos.x, pos.y])

    balls = np.array(balls)
    return balls


def is_pocketed_state(state: int) -> bool:
    """
    Check if the given state represents a pocketed ball.

    Args:
        state (int): The state of the ball.

    Returns:
        bool: True if the ball is pocketed, False otherwise.
    """
    return (
        state == ff.Ball.POCKETED_NE
        or state == ff.Ball.POCKETED_NW
        or state == ff.Ball.POCKETED_E
        or state == ff.Ball.POCKETED_W
        or state == ff.Ball.POCKETED_SE
        or state == ff.Ball.POCKETED_SW
    )


def shotparams_to_string(shot_params: ff.ShotParams, separator: str = ", ") -> str:
    """
    Converts a ShotParams object to a string representation.

    Args:
        shot_params (ff.ShotParams): The ShotParams object to convert.
        separator (str, optional): The separator to use between each parameter. Defaults to ", ".

    Returns:
        str: The string representation of the ShotParams object.
    """
    return separator.join([str(param) for param in shotparams_to_list(shot_params)])


def shotparams_to_list(shot_params: ff.ShotParams) -> list:
    """
    Converts the given ShotParams object into a list.

    Args:
        shot_params (ff.ShotParams): The ShotParams object to be converted.

    Returns:
        list: A list containing the values of the ShotParams object in the following order:
              [a, b, theta, phi, v]
    """
    return [
        shot_params.a,
        shot_params.b,
        shot_params.theta,
        shot_params.phi,
        shot_params.v,
    ]


def table_state_to_string(table_state: ff.TableState) -> str:
    """
    Converts a TableState object to a string representation.

    Args:
        table_state (ff.TableState): The TableState object to convert.

    Returns:
        str: The string representation of the TableState object.
    """

    strs = []
    for i in range(table_state.getNumBalls()):
        ball: ff.Ball = table_state.getBall(i)
        pos = ball.getPos()
        id: str = ball.getIDString()
        state: str = ball.getStateString()
        strs.append(f"{id.upper():8}  ({pos.x:.2f}, {pos.y:.2f})  {state.upper():14}")

    return "\n".join(strs)
