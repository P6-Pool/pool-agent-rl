"""
This module contains implementations of useful reward functions for the FastFiz environment.
"""

from .step_pocketed_reward import StepPocketedReward
from .total_distance_reward import TotalDistanceReward
from .delta_best_total_distance_reward import DeltaBestTotalDistanceReward
from .cue_ball_pocketed_reward import CueBallPocketedReward
from .cue_ball_not_moved_reward import CueBallNotMovedReward
from .game_won_reward import GameWonReward
from .impossible_shot_reward import ImpossibleShotReward
from .constant_reward import ConstantReward
from .balls_not_moved_reward import BallsNotMovedReward
from .velocity_reward import VelocityReward
from .exponential_velocity_reward import ExponentialVelocityReward

from .weights import (
    ConstantWeight,
    NegativeConstantWeight,
    ConstantWeightMaxSteps,
    NegativeConstantWeightMaxSteps,
    ConstantWeightNumBalls,
    NegativeConstantWeightNumBalls,
    ConstantWeightBalls,
    NegativeConstantWeightBalls,
    ConstantWeightCurrentStep,
    NegativeConstantWeightCurrentStep,
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
