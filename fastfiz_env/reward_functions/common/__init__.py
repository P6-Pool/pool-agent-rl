"""
This module contains implementations of useful reward functions for the FastFiz environment.
"""

from .balls_not_moved_reward import BallsNotMovedReward
from .constant_reward import ConstantReward
from .cue_ball_not_moved_reward import CueBallNotMovedReward
from .cue_ball_pocketed_reward import CueBallPocketedReward
from .delta_best_total_distance_reward import DeltaBestTotalDistanceReward
from .exponential_velocity_reward import ExponentialVelocityReward
from .game_won_reward import GameWonReward
from .impossible_shot_reward import ImpossibleShotReward
from .step_no_balls_pocketed_reward import StepNoBallsPocketedReward
from .step_pocketed_reward import StepPocketedReward
from .total_distance_reward import TotalDistanceReward
from .velocity_reward import VelocityReward
from .weights import (
    ConstantWeight,
    ConstantWeightBalls,
    ConstantWeightCurrentStep,
    ConstantWeightMaxSteps,
    ConstantWeightNumBalls,
    NegativeConstantWeight,
    NegativeConstantWeightBalls,
    NegativeConstantWeightCurrentStep,
    NegativeConstantWeightMaxSteps,
    NegativeConstantWeightNumBalls,
)

__all__ = [
    # Reward functions
    "StepPocketedReward",
    "TotalDistanceReward",
    "DeltaBestTotalDistanceReward",
    "CueBallPocketedReward",
    "CueBallNotMovedReward",
    "GameWonReward",
    "ImpossibleShotReward",
    "ConstantReward",
    "BallsNotMovedReward",
    "VelocityReward",
    "ExponentialVelocityReward",
    "StepNoBallsPocketedReward",
    # Weights
    "ConstantWeight",
    "NegativeConstantWeight",
    "ConstantWeightMaxSteps",
    "NegativeConstantWeightMaxSteps",
    "ConstantWeightNumBalls",
    "NegativeConstantWeightNumBalls",
    "ConstantWeightBalls",
    "NegativeConstantWeightBalls",
    "ConstantWeightCurrentStep",
    "NegativeConstantWeightCurrentStep",
]
