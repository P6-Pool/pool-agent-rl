from dataclasses import dataclass
from .utils import deg_to_vec, vec_to_abs_deg
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

    def action(self, action):

        # thata = self._compute_angle(
        #     action, self.theta_index, self.MIN_THETA, self.MAX_THETA
        # )
        # phi = self._compute_angle(action, self.phi_index, self.MIN_PHI, self.MAX_PHI)

        return action

    @staticmethod
    def _compute_angle(
        action, index: int | tuple[int, int], min_angle: int, max_angle: int
    ):
        if isinstance(index, int):
            theta = action[index]
            # convert from -1 to 1, to 0 to 70
            theta = (theta + 1) * 35

        elif isinstance(index, tuple):
            theta_x = action[index[0]]
            theta_y = action[index[1]]

            max_vec = deg_to_vec(max_angle)
            min_vec = deg_to_vec(min_angle)

            interp_vec = (
                min_vec + (max_vec - min_vec) * (np.array([theta_x, theta_y]) + 1) / 2
            )

            theta = vec_to_abs_deg(interp_vec)

        assert min_angle <= theta <= max_angle, f"Theta out of bounds: {theta}"

    @staticmethod
    def get_action_space(action_space: ActionSpaces):
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
            case _:
                raise ValueError("Invalid action space")
