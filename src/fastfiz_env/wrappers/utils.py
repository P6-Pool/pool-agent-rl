import numpy as np


def cart2sph(x: float, y: float, z: float) -> tuple[float, float, float]:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.

    Returns:
        tuple[float, float, float]: A tuple containing radius (magnitude), inclination angle (theta, in radians), and azimuth angle (phi, in radians).
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    inclination = np.arccos(z / r)
    azimuth = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)  # Using arctan2 to get correct quadrant

    return r, inclination, azimuth


def sph2deg(r: float, inclination: float, azimuth: float) -> tuple[float, float, float]:
    """
    Convert spherical coordinates to degrees.

    Args:
        r (float): Radius.
        inclination (float): Inclination angle in radians.
        azimuth (float): Azimuth angle in radians.

    Returns:
        tuple[float, float, float]: A tuple containing radius (magnitude), inclination angle (theta, in degrees), and azimuth angle (phi, in degrees).
    """
    inclination_deg: float = np.rad2deg(inclination)
    azimuth_deg: float = np.rad2deg(azimuth)
    return r, inclination_deg, azimuth_deg
