from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np


class FastFizFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for FastFiz environment.

    Input observations are an array of shape (16, 3) where each row contains the following information:
    - 0: Ball position x (normalized to [-1, 1])
    - 1: Ball position y (normalized to [-1, 1])
    - 2: Is pocketed (0 or 1)
    """

    def __init__(self, observation_space, features_dim=16):
        super(FastFizFeatureExtractor, self).__init__(observation_space, features_dim)
        # We assume that the observation space is a Box space
        self.observation_space = observation_space
        self.features_dim = features_dim

    def forward(self, observations):
        # Normalize the ball positions to [-1, 1]
        ball_positions = observations[:, :2]
        ball_positions = ball_positions * 2 - 1
        # Concatenate the normalized ball positions and the is pocketed flag
        return np.concatenate((ball_positions, observations[:, 2:]), axis=1)
