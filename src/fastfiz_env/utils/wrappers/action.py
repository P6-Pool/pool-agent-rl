from dataclasses import dataclass
from .utils import (
    deg_to_vec,
    vec_to_abs_deg,
    vec_length,
    vec_normalize,
    spherical_coordinates,
    vec_magnitude,
)
from gymnasium import ActionWrapper
from gymnasium import spaces
import numpy as np
from enum import Enum


class ActionSpaces(Enum):
    NO_OFFSET_5D = (0,)
    """No a and b offset, 5D representation of cue stick:
    - a: 0
    - b: 0
    - theta: The angle of the shot in the yz-plane (0th and 1st element).
    - phi: The angle of the in the xz-plane (2nd and 3rd element).
    - velocity: 5th element.
    """
    NO_OFFSET_4D = (1,)
    """No a and b offset, 4D representation of cue stick:
    - a: 0
    - b: 0
    - theta: The angle of the shot in the yz-plane (0th and 1st element).
    - phi: The angle of the in the xz-plane (2nd and 3rd element).
    - velocity: Derived from the unit vector of theta + phi.
    """

    NO_OFFSET_3D = (2,)
    """
    No a and b offset, 3D representation of cue stick:
    - a: 0
    - b: 0
    - theta: The angle of the shot in the yz-plane (0th and 1st element).
    - phi: The angle of the in the xz-plane (1st and 2nd element).
    - velocity: Derived from the 3D vector.
    """

    NORM_PARAMS_5D = (3,)
    """
    Normalized shot paramaters, 5D representation of cue stick:
    - a: The offset of the cue ball in the x-coordinate. (Always 0)
    - b: The offset of the cue ball in the y-coordinate. (Always 0)
    - theta: The angle of the shot in the yz-plane.
    - phi: The angle of the in the xz-plane.
    - velocity: The velocity of the shot.
    """

    NO_OFFSET_NORM_PARAMS_3D = (4,)
    """
    Normalized shot paramaters, 5D representation of cue stick:
    - a: 0
    - b: 0
    - theta: The angle of the shot in the yz-plane.
    - phi: The angle of the in the xz-plane.
    - velocity: The velocity of the shot.
    """

    OUTPUT = (5,)
    """
    Output of FastFizActionWrapper.
    """


class FastFizActionWrapper(ActionWrapper):
    MIN_THETA = 0
    MAX_THETA = 70
    MIN_PHI = 0
    MAX_PHI = 360
    MIN_VELOCITY = 0
    MAX_VELOCITY = 10

    def __init__(
        self,
        env,
        action_space_id: ActionSpaces,
    ):
        super().__init__(env)
        self.env = env
        self.action_space_id = action_space_id

    def action(self, action):

        # Offset a and b are always 0
        offset_a = 0
        offset_b = 0

        match self.action_space_id:
            case ActionSpaces.NO_OFFSET_4D:
                if np.allclose(action, 0):
                    return np.array([offset_a, offset_b, 0, 0, 0])
                vec_theta = action[:2]
                theta = vec_to_abs_deg(vec_theta)
                theta = np.interp(theta, (0, 360), (self.MIN_THETA, self.MAX_THETA))

                vec_phi = action[2:4]
                phi = vec_to_abs_deg(vec_phi)

                vec_velocity = vec_length(vec_theta + vec_phi)
                velocity = np.interp(
                    vec_velocity, (0, 2), (self.MIN_VELOCITY, self.MAX_VELOCITY)
                )

            case ActionSpaces.NO_OFFSET_3D:
                if np.allclose(action, 0):
                    return np.array([offset_a, offset_b, 0, 0, 0])
                r, theta, phi = spherical_coordinates(action)
                theta = np.interp(
                    theta, (0, 360), (self.MIN_THETA, self.MAX_THETA - 0.001)
                )
                phi = np.interp(phi, (0, 360), (self.MIN_PHI, self.MAX_PHI))
                velocity = np.interp(
                    r, (0, np.sqrt(3)), (self.MIN_VELOCITY, self.MAX_VELOCITY)
                )

            case ActionSpaces.NO_OFFSET_5D:
                if np.allclose(action, 0):
                    return np.array([offset_a, offset_b, 0, 0, 0])
                vec_theta = action[:2]
                theta = vec_to_abs_deg(vec_theta)
                theta = np.interp(theta, (0, 360), (self.MIN_THETA, self.MAX_THETA))

                vec_phi = action[2:4]
                phi = vec_to_abs_deg(vec_phi)

                velocity = np.interp(
                    action[4], (-1, 1), (self.MIN_VELOCITY, self.MAX_VELOCITY)
                )
            case ActionSpaces.NORM_PARAMS_5D:
                theta = np.interp(action[2], (-1, 1), (self.MIN_THETA, self.MAX_THETA))
                phi = np.interp(action[3], (-1, 1), (self.MIN_PHI, self.MAX_PHI))
                velocity = np.interp(
                    action[4], (-1, 1), (self.MIN_VELOCITY, self.MAX_VELOCITY)
                )
            case ActionSpaces.NO_OFFSET_NORM_PARAMS_3D:
                theta = np.interp(action[0], (-1, 1), (self.MIN_THETA, self.MAX_THETA))
                phi = np.interp(action[1], (-1, 1), (self.MIN_PHI, self.MAX_PHI))
                velocity = np.interp(
                    action[2], (-1, 1), (self.MIN_VELOCITY, self.MAX_VELOCITY)
                )

        action = np.array([offset_a, offset_b, theta, phi, velocity])
        return action

    @staticmethod
    def get_action_space(action_space: ActionSpaces) -> spaces.Space:
        match action_space:
            case ActionSpaces.NO_OFFSET_5D:
                return spaces.Box(
                    low=np.array([-1, -1, -1, -1, -1]),
                    high=np.array([1, 1, 1, 1, 1]),
                    dtype=np.float32,
                )
            case ActionSpaces.NO_OFFSET_4D:
                return spaces.Box(
                    low=np.array([-1, -1, -1, -1]),
                    high=np.array([1, 1, 1, 1]),
                    dtype=np.float32,
                )
            case ActionSpaces.NO_OFFSET_3D:
                return spaces.Box(
                    low=np.array([-1, -1, -1]),
                    high=np.array([1, 1, 1]),
                    dtype=np.float32,
                )
            case ActionSpaces.OUTPUT:
                return spaces.Box(
                    low=np.array([-1, -1, 0, 0, 0]),
                    high=np.array([1, 1, 70, 360, 10]),
                    dtype=np.float32,
                )
            case ActionSpaces.NORM_PARAMS_5D:
                return spaces.Box(
                    low=np.array([0, 0, -1, -1, -1]),
                    high=np.array([0, 0, 1, 1, 1]),
                    dtype=np.float32,
                )
            case ActionSpaces.NO_OFFSET_NORM_PARAMS_3D:
                return spaces.Box(
                    low=np.array([-1, -1, -1]),
                    high=np.array([1, 1, 1]),
                    dtype=np.float32,
                )
            case _:
                raise ValueError("Invalid action space")
