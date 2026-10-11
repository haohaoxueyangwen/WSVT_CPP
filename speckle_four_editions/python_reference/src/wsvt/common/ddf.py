from __future__ import annotations

import math

import numpy as np


def covariance_from_abc(
    a: float, b: float, c: float
) -> tuple[np.ndarray, float, float, float]:
    """Return covariance (x, y), major/minor widths, and major angle in degrees."""
    precision = np.asarray([[2.0 * c, b], [b, 2.0 * a]], dtype=np.float64)
    if not np.isfinite(precision).all():
        return np.full((2, 2), np.nan), math.nan, math.nan, math.nan
    determinant = float(np.linalg.det(precision))
    if determinant <= 0.0 or not math.isfinite(determinant):
        return np.full((2, 2), np.nan), math.nan, math.nan, math.nan
    covariance = np.linalg.inv(precision)
    values, vectors = np.linalg.eigh(covariance)
    if values[0] < 0.0 or not np.isfinite(values).all():
        return covariance, math.nan, math.nan, math.nan
    major_vector = vectors[:, 1]
    angle = float(np.degrees(np.arctan2(major_vector[1], major_vector[0])) % 180.0)
    return covariance, float(np.sqrt(values[1])), float(np.sqrt(values[0])), angle
