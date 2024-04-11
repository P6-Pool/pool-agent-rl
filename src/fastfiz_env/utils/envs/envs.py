import fastfiz as ff
from gymnasium.wrappers import resize_observation


def game_won(table_state: ff.TableState) -> bool:
    """
    Check if the game is won. The game is won if all balls are pocketed except the cue ball.
    """
    if table_state.getBall(0).isPocketed():
        return False
    for i in range(1, table_state.getNumBalls()):
        if table_state.getBall(i).isInPlay():
            return False
    return True


def terminal_state(table_state: ff.TableState) -> bool:
    """
    Check if the game is in a terminal state.
    """
    if table_state.getBall(0).isPocketed():
        return True
    return game_won(table_state)


def possible_shot(table_state: ff.TableState, shot_params: ff.ShotParams) -> bool:
    """
    Check if the shot is possible.
    """
    return (
        table_state.isPhysicallyPossible(shot_params) == ff.TableState.OK_PRECONDITION
    )
