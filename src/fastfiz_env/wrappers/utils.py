import numpy as np


def deg_to_vec(deg: float) -> np.ndarray:
    """
    Gets the vector of an angle.
    """
    rad = np.deg2rad(deg)
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)


def vec_to_deg(vec: np.ndarray) -> float:
    """
    Gets the angle of a vector.
    """
    return np.rad2deg(np.arctan2(vec[1], vec[0]))


def vec_to_abs_deg(vec: np.ndarray) -> float:
    """
    Gets the absolute angle of a vector.
    """
    return vec_to_deg(vec) % 360


def vec_length(vec: np.ndarray) -> float:
    """
    Gets the length of a vector.
    """
    return np.linalg.norm(vec)


def vec_normalize(vec: np.ndarray) -> np.ndarray:
    """
    Gets the unit vector of a vector.
    """
    return vec / vec_length(vec)


def spherical_coordinates(vector: np.ndarray) -> tuple[float, float, float]:
    """
    Converts a vector to spherical coordinates.

    Returns:
        r: float - The magnitude of the vector.
        theta: float - The angle from the z-axis.
        phi: float - The angle in the xy-plane.
    """

    assert len(vector) == 3, "Vector must have excatly 3 components."
    Vx, Vy, Vz = vector

    theta = np.arccos(Vz / np.linalg.norm(vector))

    phi = np.arctan2(Vy, Vx)  # Using arctan2 to get correct quadrant
    phi = (phi + 2 * np.pi) % (2 * np.pi)

    r = np.linalg.norm(vector)

    return r, np.degrees(theta), np.degrees(phi)


def vec_magnitude(vector):
    return np.linalg.norm(vector)
