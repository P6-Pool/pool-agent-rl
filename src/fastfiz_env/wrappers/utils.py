import numpy as np


def cart2sph(x: float, y: float, z: float) -> tuple[float, float, float]:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.

    Returns:
        tuple[float, float, float]: A tuple containing azimuth angle (in degrees), elevation angle (in degrees), and radius.
    """
    hxy: float = np.hypot(x, y)
    r: float = np.hypot(hxy, z)
    el: float = np.arctan2(z, hxy)
    az: float = np.arctan2(y, x)
    return az, el, r


def sph2deg(az: float, el: float, r: float) -> tuple[float, float, float]:
    """
    Convert spherical coordinates to degrees.

    Args:
        az (float): Azimuth angle in radians.
        el (float): Elevation angle in radians.
        r (float): Radius.

    Returns:
        tuple[float, float, float]: A tuple containing azimuth angle (phi, in degrees), elevation angle (theta, in degrees), and radius.
    """
    phi: float = np.rad2deg(az % (2 * np.pi))
    theta: float = np.rad2deg(el % np.pi)
    return phi, theta, r
