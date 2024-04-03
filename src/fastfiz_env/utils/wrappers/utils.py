import numpy as np


def deg_to_vec(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    print("rad: ", rad)
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)


def vec_to_deg(vec: np.ndarray) -> float:
    return np.rad2deg(np.arctan2(vec[1], vec[0]))


def vec_to_abs_deg(vec: np.ndarray) -> float:
    return vec_to_deg(vec) % 360
