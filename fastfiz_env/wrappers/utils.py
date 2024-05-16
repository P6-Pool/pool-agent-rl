import numpy as np


def cart2sph(x: float, y: float, z: float) -> tuple[float, float, float]:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.

    Returns:
        tuple[float, float, float]: A tuple containing radius (magnitude), elevation angle (theta, in radians), and azimuth angle (phi, in radians).
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    el = np.arccos(z / r)
    az = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)  # Using arctan2 to get correct quadrant

    return r, el, az


def sph2deg(r: float, el: float, az: float) -> tuple[float, float, float]:
    """
    Convert spherical coordinates to degrees.

    Args:
        r (float): Radius.
        el (float): Elevation angle in radians.
        az (float): Azimuth angle in radians.

    Returns:
        tuple[float, float, float]: A tuple containing radius (magnitude), elevation angle (theta, in degrees), and azimuth angle (phi, in degrees).
    """
    el_deg: float = np.rad2deg(el)
    az_deg: float = np.rad2deg(az)
    return r, el_deg, az_deg
