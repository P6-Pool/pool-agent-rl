from .utils import cart2sph, sph2deg
from gymnasium import ActionWrapper
from gymnasium import spaces
import numpy as np
from enum import Enum


class ActionSpaces(Enum):
    VECTOR_2D = (0,)
    """
    2D vector representation of cue stick:
    - a: Always 0.
    - b: Always 11.
    - theta: Always 20.
    - phi: Angle between the 2D vector and the x-axis.
    - velocity: Magnitude of the 2D vector.
    """
    VECTOR_3D = (1,)
    """
    3D vector representation of cue stick:
    - a: Always 0.
    - b: Always 0.
    - theta: Derived from the 3D vector spherical coordinates.
    - phi: Derived from the 3D vector spherical coordinates.
    - velocity: Magnitude of the 3D vector.
    """
    NORM_3D = (2,)
    """
    Normalized shot paramaters, 3D representation of cue stick:
    - a: Always 0.
    - b: Always 0.
    - theta: Normalized angle from `MIN_THETA` to `MAX_THETA`.
    - phi: Normalized angle from `MIN_PHI` to `MAX_PHI`.
    - velocity: Normalized velocity from `MIN_VELOCITY` to `MAX_VELOCITY`.
    """
    NORM_5D = (3,)
    """
    Normalized shot paramaters, 5D representation of cue stick:
    - a: Alwyas 0.
    - b: Always 0.
    - theta: Normalized angle from `MIN_THETA` to `MAX_THETA`.
    - phi: Normalized angle from `MIN_PHI` to `MAX_PHI`.
    - velocity: Normalized velocity from `MIN_VELOCITY` to `MAX_VELOCITY`.
    """
    OFFSET_NORM_5D = (4,)
    """
    Normalized shot paramaters, 5D representation of cue stick:
    - a: Normalized value from `MIN_OFFSET` to `MAX_OFFSET`.
    - b: Normalized value from `MIN_OFFSET` to `MAX_OFFSET`.
    - theta: Normalized angle from `MIN_THETA` to `MAX_THETA`.
    - phi: Normalized angle from `MIN_PHI` to `MAX_PHI`.
    - velocity: Normalized velocity from `MIN_VELOCITY` to `MAX_VELOCITY`.
    """
    OUTPUT_5D = (5,)
    """
    5D representation of cue stick, using original FastFiz shot parameter values:
    - a: Offset of the cue ball in the x-coordinate.
    - b: Offset of the cue ball in the y-coordinate.
    - theta: The vertical angle of the shot.
    - phi: The horizontal angle of the shot.
    - velocity: The power of the shot (in m/s).
    """

    def __str__(self):
        return self.name


class FastFizActionWrapper(ActionWrapper):
    MIN_THETA = 0
    MAX_THETA = 70 - 0.001
    MIN_PHI = 0
    MAX_PHI = 360 - 0.001
    MIN_VELOCITY = 0
    MAX_VELOCITY = 10 - 0.001
    MIN_OFFSET = -28
    MAX_OFFSET = 28
    SPACES = {
        "VECTOR_2D": spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            shape=(2,),
            dtype=np.float32,
        ),
        "VECTOR_3D": spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            dtype=np.float32,
        ),
        "NORM_3D": spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            dtype=np.float32,
        ),
        "NORM_5D": spaces.Box(
            low=np.array([0, 0, -1, -1, -1]),
            high=np.array([0, 0, 1, 1, 1]),
            dtype=np.float32,
        ),
        "OFFSET_NORM_5D": spaces.Box(
            low=np.array([-1, -1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1, 1]),
            dtype=np.float32,
        ),
        "OUTPUT": spaces.Box(
            low=np.array([-28, -28, 0, 0, 0]),
            high=np.array([28, 28, 70, 360, 10]),
            dtype=np.float32,
        ),
    }

    def __init__(
        self,
        env,
        action_space_id: ActionSpaces,
    ):
        super().__init__(env)
        self.env = env
        self.action_space_id = action_space_id
        self.action_space = self.SPACES[action_space_id.name]

    def action(self, action: np.ndarray[float, np.dtype[np.float32]]) -> np.ndarray[float, np.dtype[np.float32]]:
        offset_a = 0.0
        offset_b = 0.0

        match self.action_space_id:
            case ActionSpaces.VECTOR_2D:
                theta = 20.0
                phi = float(np.degrees(np.arctan2(action[1], action[0])) % self.MAX_PHI)
                offset_b = 11.0
                velocity = float(
                    np.interp(
                        np.hypot(*action),
                        (0, np.sqrt(2)),
                        (self.MIN_VELOCITY, self.MAX_VELOCITY - 5),
                    )
                )
            case ActionSpaces.VECTOR_3D:
                x, y, z = action
                r, inclination, azimuth = sph2deg(*cart2sph(x, y, z))
                phi = float(np.interp(azimuth, (0, 360), (self.MIN_PHI, self.MAX_PHI)))
                theta = float(np.interp(inclination, (0, 180), (self.MIN_THETA, self.MAX_THETA)))
                velocity = float(np.interp(r, (0, np.sqrt(3)), (self.MIN_VELOCITY, self.MAX_VELOCITY)))
            case ActionSpaces.NORM_3D:
                theta = float(np.interp(action[0], (-1, 1), (self.MIN_THETA, self.MAX_THETA)))
                phi = float(np.interp(action[1], (-1, 1), (self.MIN_PHI, self.MAX_PHI)))
                velocity = float(np.interp(action[2], (-1, 1), (self.MIN_VELOCITY, self.MAX_VELOCITY)))
            case ActionSpaces.NORM_5D:
                theta = float(np.interp(action[2], (-1, 1), (self.MIN_THETA, self.MAX_THETA)))
                phi = float(np.interp(action[3], (-1, 1), (self.MIN_PHI, self.MAX_PHI)))
                velocity = float(np.interp(action[4], (-1, 1), (self.MIN_VELOCITY, self.MAX_VELOCITY)))
            case ActionSpaces.OFFSET_NORM_5D:
                offset_a = float(np.interp(action[0], (-1, 1), (self.MIN_OFFSET, self.MAX_OFFSET)))
                offset_b = float(np.interp(action[1], (-1, 1), (self.MIN_OFFSET, self.MAX_OFFSET)))
                theta = float(np.interp(action[2], (-1, 1), (self.MIN_THETA, self.MAX_THETA)))
                phi = float(np.interp(action[3], (-1, 1), (self.MIN_PHI, self.MAX_PHI)))
                velocity = float(np.interp(action[4], (-1, 1), (self.MIN_VELOCITY, self.MAX_VELOCITY)))

        action = np.array([offset_a, offset_b, theta, phi, velocity])
        return action
