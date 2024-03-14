import fastfiz as ff
from typing import Callable, TypeAlias

RewardFunction: TypeAlias = Callable[[
    ff.TableState, ff.TableState, bool, ff.ShotParams], float]
"""
Used to define the reward function for the environment.

RewardFunction is a type alias for a function that takes three arguments:
    1st argument: Table state before action
    2nd argument: Table state after action
    3rd argument: If the action was possible
    4th argument: The shot parameters
    
The function returns a float, which is the reward for the action.
"""
