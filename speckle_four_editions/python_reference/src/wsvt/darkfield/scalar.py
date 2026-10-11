"""Fixed-displacement scalar transmission–dark-field solver.

This module implements the UMPA modulation-attenuation model independently of
the author's directional UMPA-DDF fitter. Given a WSVT/WXST displacement, it
returns to raw intensities and estimates (T, D) by weighted linear least
squares. It does not invent a new scattering model.

Model, at one analysis window:

    s_i = T [ m_i + D (r_i - m_i) ] + eps_i

with a = T D, b = T (1 - D), so s_i = a r_i + b m_i + eps_i.
T_hat = a + b, D_hat = a / (a + b). D is the retained speckle modulation
fraction, not a scattering angle.

The analysis window is not the WSVT/WXST search window. Default field radius
is 7 (15 x 15 Hamming). Public displacement samples the reference at
(y - u_y, x - u_x). Solutions are unconstrained; physical use is gated by
identifiability flags. D is never clipped into [0, 1].
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from wsvt.common.hashing import file_sha256 as sha256
from wsvt.common.json_io import write_json

DEFAULT_ANALYSIS_RADIUS = 7
RELATIVE_DELTA_TOLERANCE = 1.0e-12
TRANSMISSION_EPSILON = 1.0e-12
METHOD_ID = "scalar_modeldf_fixed_u_linear"
LOOKUP_CONVENTION = "reference_at_(y-u_y, x-u_x)"


@dataclass(frozen=True)
class UmpaStatistics:
    l1: float
    l2: float
    l3: float
    l4: float
    l5: float
    l6: float
    weight_sum: float
    observation_count: int


@dataclass(frozen=True)
class UmpaFit:
    statistics: UmpaStatistics
    alpha: float
    beta: float
    transmission: float
    visibility: float
    cost: float
    delta: float
    condition: float
    numerical_valid: bool
    physical_valid: bool


@dataclass(frozen=True)
class ScalarFixture:
    id: str
    sample: np.ndarray
    reference: np.ndarray
    reference_mean: np.ndarray
    weights: np.ndarray
    expected_transmission: float | None
    expected_visibility: float | None
    expect_numerical_valid: bool
    expect_physical_valid: bool


def normalized_hamming_window_2d(radius: int) -> np.ndarray:
    """Return a separable, normalized NumPy-compatible Hamming window."""
    if radius < 0:
        raise ValueError("analysis radius must be non-negative")
    width = 2 * radius + 1
    one_d = np.hamming(width) if width > 1 else np.ones(1, dtype=np.float64)
    window = np.multiply.outer(one_d, one_d).astype(np.float64, copy=False)
    window /= float(window.sum(dtype=np.float64))
    return window


def _as_float64_vector(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return np.ascontiguousarray(array)


def accumulate_statistics(
    sample: Iterable[float],
    reference: Iterable[float],
    reference_mean: Iterable[float],
    weights: Iterable[float] | None = None,
) -> UmpaStatistics:
    """Accumulate the six UMPA ModelDF sufficient statistics.

    Active observations must be finite. A zero weight acts as an explicit mask
    and may cover non-finite sample/reference values.
    """
    sample_array = _as_float64_vector(sample, "sample")
    reference_array = _as_float64_vector(reference, "reference")
    mean_array = _as_float64_vector(reference_mean, "reference_mean")
    if not (sample_array.size == reference_array.size == mean_array.size):
        raise ValueError("sample/reference/reference_mean size mismatch")
    if sample_array.size == 0:
        raise ValueError("at least one observation is required")
    if weights is None:
        weight_array = np.ones(sample_array.size, dtype=np.float64)
    else:
        weight_array = _as_float64_vector(weights, "weights")
        if weight_array.size != sample_array.size:
            raise ValueError("weights size mismatch")
    if np.any(~np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("weights must be finite and non-negative")
    active = weight_array > 0.0
    if not np.any(active):
        raise ValueError("at least one positive weight is required")
    if (
        np.any(~np.isfinite(sample_array[active]))
        or np.any(~np.isfinite(reference_array[active]))
        or np.any(~np.isfinite(mean_array[active]))
    ):
        raise ValueError("active observations must be finite")

    w = weight_array[active]
    s = sample_array[active]
    r = reference_array[active]
    mu = mean_array[active]
    return UmpaStatistics(
        l1=float(np.sum(w * s * s, dtype=np.float64)),
        l2=float(np.sum(w * mu * mu, dtype=np.float64)),
        l3=float(np.sum(w * r * r, dtype=np.float64)),
        l4=float(np.sum(w * mu * s, dtype=np.float64)),
        l5=float(np.sum(w * r * s, dtype=np.float64)),
        l6=float(np.sum(w * r * mu, dtype=np.float64)),
        weight_sum=float(np.sum(w, dtype=np.float64)),
        observation_count=int(np.count_nonzero(active)),
    )


def solve_statistics(
    statistics: UmpaStatistics,
    *,
    relative_delta_tolerance: float = RELATIVE_DELTA_TOLERANCE,
    transmission_epsilon: float = TRANSMISSION_EPSILON,
) -> UmpaFit:
    """Solve alpha=T*D and beta=T*(1-D) without clipping the result."""
    if relative_delta_tolerance < 0.0 or transmission_epsilon < 0.0:
        raise ValueError("tolerances must be non-negative")
    values = np.asarray(
        [
            statistics.l1,
            statistics.l2,
            statistics.l3,
            statistics.l4,
            statistics.l5,
            statistics.l6,
            statistics.weight_sum,
        ],
        dtype=np.float64,
    )
    delta = statistics.l3 * statistics.l2 - statistics.l6 * statistics.l6
    trace = statistics.l3 + statistics.l2
    discriminant = math.hypot(
        statistics.l3 - statistics.l2, 2.0 * statistics.l6
    )
    eigen_max = 0.5 * (trace + discriminant)
    eigen_min = 0.5 * (trace - discriminant)
    condition = (
        eigen_max / eigen_min
        if eigen_min > 0.0 and math.isfinite(eigen_max)
        else math.inf
    )
    scale = max(
        abs(statistics.l3 * statistics.l2),
        abs(statistics.l6 * statistics.l6),
        np.finfo(np.float64).tiny,
    )
    valid = bool(
        np.all(np.isfinite(values))
        and statistics.weight_sum > 0.0
        and math.isfinite(delta)
        and delta > relative_delta_tolerance * scale
    )
    if not valid:
        return UmpaFit(
            statistics=statistics,
            alpha=math.nan,
            beta=math.nan,
            transmission=math.nan,
            visibility=math.nan,
            cost=math.nan,
            delta=float(delta),
            condition=float(condition),
            numerical_valid=False,
            physical_valid=False,
        )

    alpha = (
        statistics.l2 * statistics.l5 - statistics.l4 * statistics.l6
    ) / delta
    beta = (
        statistics.l3 * statistics.l4 - statistics.l5 * statistics.l6
    ) / delta
    transmission = alpha + beta
    numerical_valid = bool(
        math.isfinite(alpha)
        and math.isfinite(beta)
        and math.isfinite(transmission)
        and abs(transmission) > transmission_epsilon
    )
    visibility = alpha / transmission if numerical_valid else math.nan
    cost = (
        statistics.l1
        + beta * beta * statistics.l2
        + alpha * alpha * statistics.l3
        - 2.0 * beta * statistics.l4
        - 2.0 * alpha * statistics.l5
        + 2.0 * beta * alpha * statistics.l6
    ) / statistics.weight_sum
    numerical_valid = bool(
        numerical_valid and math.isfinite(visibility) and math.isfinite(cost)
    )
    physical_slack = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(visibility))
    physical_valid = bool(
        numerical_valid
        and transmission > 0.0
        and visibility >= -physical_slack
        and visibility <= 1.0 + physical_slack
    )
    return UmpaFit(
        statistics=statistics,
        alpha=float(alpha),
        beta=float(beta),
        transmission=float(transmission),
        visibility=float(visibility),
        cost=float(cost),
        delta=float(delta),
        condition=float(condition),
        numerical_valid=numerical_valid,
        physical_valid=physical_valid,
    )


def fit_physical(
    sample: Iterable[float],
    reference: Iterable[float],
    reference_mean: Iterable[float],
    weights: Iterable[float] | None = None,
    *,
    relative_delta_tolerance: float = RELATIVE_DELTA_TOLERANCE,
    transmission_epsilon: float = TRANSMISSION_EPSILON,
) -> UmpaFit:
    """Fit raw intensities and recompute a manifestly non-negative cost."""
    sample_array = _as_float64_vector(sample, "sample")
    reference_array = _as_float64_vector(reference, "reference")
    mean_array = _as_float64_vector(reference_mean, "reference_mean")
    weight_array = (
        np.ones(sample_array.size, dtype=np.float64)
        if weights is None
        else _as_float64_vector(weights, "weights")
    )
    statistics = accumulate_statistics(
        sample_array, reference_array, mean_array, weight_array
    )
    fit = solve_statistics(
        statistics,
        relative_delta_tolerance=relative_delta_tolerance,
        transmission_epsilon=transmission_epsilon,
    )
    if not fit.numerical_valid:
        return fit
    active = weight_array > 0.0
    residual = (
        sample_array[active]
        - fit.alpha * reference_array[active]
        - fit.beta * mean_array[active]
    )
    cost = float(
        np.sum(weight_array[active] * residual * residual, dtype=np.float64)
        / statistics.weight_sum
    )
    return UmpaFit(**{**asdict(fit), "statistics": statistics, "cost": cost})


def sample_array(
    image: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    """Sample a 2-D image at (y, x). order 0 is nearest; order 1 is bilinear."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("image must be 2-D")
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.shape != x.shape:
        raise ValueError("y and x must have the same shape")
    height, width = image.shape
    out = np.full(y.shape, np.nan, dtype=np.float64)
    if order == 0:
        yi = np.rint(y).astype(np.int64)
        xi = np.rint(x).astype(np.int64)
        ok = (yi >= 0) & (yi < height) & (xi >= 0) & (xi < width)
        out[ok] = image[yi[ok], xi[ok]]
        return out
    if order != 1:
        raise ValueError("order must be 0 or 1")
    y0f = np.floor(y)
    x0f = np.floor(x)
    wy = y - y0f
    wx = x - x0f
    y0 = y0f.astype(np.int64)
    x0 = x0f.astype(np.int64)
    need_y1 = wy > 0.0
    need_x1 = wx > 0.0
    ok = (
        (y0 >= 0)
        & (y0 < height)
        & (x0 >= 0)
        & (x0 < width)
        & (~need_y1 | (y0 + 1 < height))
        & (~need_x1 | (x0 + 1 < width))
    )
    if not np.any(ok):
        return out
    y1 = np.where(need_y1, y0 + 1, y0)
    x1 = np.where(need_x1, x0 + 1, x0)
    v00 = image[y0[ok], x0[ok]]
    v01 = image[y0[ok], x1[ok]]
    v10 = image[y1[ok], x0[ok]]
    v11 = image[y1[ok], x1[ok]]
    wy_ok = wy[ok]
    wx_ok = wx[ok]
    out[ok] = (
        (1.0 - wy_ok) * (1.0 - wx_ok) * v00
        + (1.0 - wy_ok) * wx_ok * v01
        + wy_ok * (1.0 - wx_ok) * v10
        + wy_ok * wx_ok * v11
    )
    return out


def _as_stack(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 2:
        return array[None]
    if array.ndim == 3:
        return array
    raise ValueError(f"{name} must have shape (H,W) or (F,H,W)")


def fit_scalar_field(
    sample: np.ndarray,
    reference: np.ndarray,
    displace_y: np.ndarray,
    displace_x: np.ndarray,
    *,
    analysis_radius: int = DEFAULT_ANALYSIS_RADIUS,
    origin_y: float = 0.0,
    origin_x: float = 0.0,
    relative_delta_tolerance: float = RELATIVE_DELTA_TOLERANCE,
    transmission_epsilon: float = TRANSMISSION_EPSILON,
) -> dict[str, np.ndarray]:
    """Estimate T and D on the displacement grid from raw intensities.

    ``displace_y`` / ``displace_x`` are public WSVT/WXST fields. Pixel (i, j)
    of those fields corresponds to intensity coordinate
    (origin_y + i, origin_x + j). The analysis window is a Hamming disk of
    the given radius; it is not the tracker search window.

    Multi-frame stacks share one displacement field. The local reference mean
    is computed per frame over the analysis window, then all frames enter the
    same linear system. A one-pixel-per-frame mean is rejected by construction.
    """
    if analysis_radius < 0:
        raise ValueError("analysis radius must be non-negative")
    sample_stack = _as_stack(sample, "sample")
    reference_stack = _as_stack(reference, "reference")
    if sample_stack.shape != reference_stack.shape:
        raise ValueError("sample and reference shapes must match")
    uy = np.asarray(displace_y, dtype=np.float64)
    ux = np.asarray(displace_x, dtype=np.float64)
    if uy.ndim != 2 or ux.shape != uy.shape:
        raise ValueError("displace_y and displace_x must be identical 2-D fields")
    height, width = uy.shape
    window = normalized_hamming_window_2d(analysis_radius)
    ii, jj = np.indices((height, width), dtype=np.float64)
    cy = origin_y + ii
    cx = origin_x + jj
    l1 = np.zeros((height, width), dtype=np.float64)
    l2 = np.zeros((height, width), dtype=np.float64)
    l3 = np.zeros((height, width), dtype=np.float64)
    l4 = np.zeros((height, width), dtype=np.float64)
    l5 = np.zeros((height, width), dtype=np.float64)
    l6 = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)
    n_frames = sample_stack.shape[0]
    for frame in range(n_frames):
        sample_img = sample_stack[frame]
        reference_img = reference_stack[frame]
        r_acc = np.zeros((height, width), dtype=np.float64)
        w_acc = np.zeros((height, width), dtype=np.float64)
        patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for row, offset_y in enumerate(range(-analysis_radius, analysis_radius + 1)):
            for col, offset_x in enumerate(range(-analysis_radius, analysis_radius + 1)):
                wt = float(window[row, col])
                if wt <= 0.0:
                    continue
                sy = cy + offset_y
                sx = cx + offset_x
                s = sample_array(sample_img, sy, sx, order=0)
                r = sample_array(reference_img, sy - uy, sx - ux, order=1)
                ok = np.isfinite(s) & np.isfinite(r)
                wpix = np.where(ok, wt, 0.0)
                s = np.where(ok, s, 0.0)
                r = np.where(ok, r, 0.0)
                r_acc += wpix * r
                w_acc += wpix
                patches.append((wpix, s, r))
        mu = np.divide(r_acc, w_acc, out=np.zeros_like(r_acc), where=w_acc > 0.0)
        for wpix, s, r in patches:
            l1 += wpix * s * s
            l2 += wpix * mu * mu
            l3 += wpix * r * r
            l4 += wpix * mu * s
            l5 += wpix * r * s
            l6 += wpix * r * mu
            weight_sum += wpix
    delta = l3 * l2 - l6 * l6
    scale = np.maximum(
        np.maximum(np.abs(l3 * l2), np.abs(l6 * l6)),
        np.finfo(np.float64).tiny,
    )
    numerical = (
        (weight_sum > 0.0)
        & np.isfinite(delta)
        & (delta > relative_delta_tolerance * scale)
    )
    alpha = np.divide(
        l2 * l5 - l4 * l6, delta, out=np.full_like(delta, np.nan), where=numerical
    )
    beta = np.divide(
        l3 * l4 - l5 * l6, delta, out=np.full_like(delta, np.nan), where=numerical
    )
    transmission = alpha + beta
    numerical &= (
        np.isfinite(alpha)
        & np.isfinite(beta)
        & np.isfinite(transmission)
        & (np.abs(transmission) > transmission_epsilon)
    )
    visibility = np.divide(
        alpha, transmission, out=np.full_like(alpha, np.nan), where=numerical
    )
    cost = (
        l1
        + beta * beta * l2
        + alpha * alpha * l3
        - 2.0 * beta * l4
        - 2.0 * alpha * l5
        + 2.0 * beta * alpha * l6
    )
    cost = np.divide(cost, weight_sum, out=np.full_like(cost, np.nan), where=numerical)
    numerical &= np.isfinite(visibility) & np.isfinite(cost)
    slack = 64.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(visibility))
    physical = (
        numerical
        & (transmission > 0.0)
        & (visibility >= -slack)
        & (visibility <= 1.0 + slack)
    )
    ln_d = np.full_like(visibility, np.nan)
    ln_ok = physical & (visibility > 1.0e-12) & (visibility <= 1.0)
    ln_d[ln_ok] = -np.log(visibility[ln_ok])
    constraint_active = np.zeros((height, width), dtype=bool)
    return {
        "T": transmission,
        "D": visibility,
        "a": alpha,
        "b": beta,
        "cost": cost,
        "delta": delta,
        "weight_sum": weight_sum,
        "numerical_valid": numerical,
        "physical_valid": physical,
        "lnD": ln_d,
        "constraint_active": constraint_active,
    }


def synthesize_speckle(height: int, width: int, *, seed: int = 0) -> np.ndarray:
    """Deterministic zero-mean texture plus a positive mean, for solver tests."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((height, width), dtype=np.float64)
    texture = np.sin(0.35 * yy + 0.21 * xx) + 0.4 * np.cos(0.17 * yy - 0.29 * xx)
    texture += 0.15 * rng.standard_normal((height, width))
    texture -= float(texture.mean())
    return 100.0 + 20.0 * texture


def deterministic_speckle(height: int, width: int) -> np.ndarray:
    """C++-matched speckle (no RNG) for Python/C++ field parity."""
    yy, xx = np.indices((height, width), dtype=np.float64)
    texture = np.sin(0.35 * yy + 0.21 * xx) + 0.4 * np.cos(0.17 * yy - 0.29 * xx)
    texture -= float(texture.mean())
    return (100.0 + 20.0 * texture).astype(np.float64)


def apply_scalar_model(
    reference: np.ndarray,
    *,
    transmission: float,
    visibility: float,
    shift_y: float = 0.0,
    shift_x: float = 0.0,
) -> np.ndarray:
    """Pixelwise forward sample using one global reference mean.

    This matches the textbook example that separates T from D. Field recovery
    tests that must match the solver's local-mean definition should use
    ``forward_scalar_field`` instead.
    """
    reference = np.asarray(reference, dtype=np.float64)
    height, width = reference.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    shifted = sample_array(reference, yy - shift_y, xx - shift_x, order=1)
    mean = float(np.nanmean(shifted))
    return transmission * (mean + visibility * (shifted - mean))


def forward_scalar_field(
    reference: np.ndarray,
    displace_y: np.ndarray,
    displace_x: np.ndarray,
    *,
    transmission: float,
    visibility: float,
    analysis_radius: int = DEFAULT_ANALYSIS_RADIUS,
    origin_y: float = 0.0,
    origin_x: float = 0.0,
) -> np.ndarray:
    """Synthesize a sample field with the same local mean as ``fit_scalar_field``."""
    reference_img = np.asarray(reference, dtype=np.float64)
    if reference_img.ndim != 2:
        raise ValueError("forward_scalar_field expects a 2-D reference")
    uy = np.asarray(displace_y, dtype=np.float64)
    ux = np.asarray(displace_x, dtype=np.float64)
    height, width = uy.shape
    window = normalized_hamming_window_2d(analysis_radius)
    ii, jj = np.indices((height, width), dtype=np.float64)
    cy = origin_y + ii
    cx = origin_x + jj
    r_acc = np.zeros((height, width), dtype=np.float64)
    w_acc = np.zeros((height, width), dtype=np.float64)
    r_center = sample_array(reference_img, cy - uy, cx - ux, order=1)
    for row, offset_y in enumerate(range(-analysis_radius, analysis_radius + 1)):
        for col, offset_x in enumerate(range(-analysis_radius, analysis_radius + 1)):
            wt = float(window[row, col])
            if wt <= 0.0:
                continue
            r = sample_array(
                reference_img,
                cy + offset_y - uy,
                cx + offset_x - ux,
                order=1,
            )
            ok = np.isfinite(r)
            wpix = np.where(ok, wt, 0.0)
            r_acc += wpix * np.where(ok, r, 0.0)
            w_acc += wpix
    mu = np.divide(r_acc, w_acc, out=np.full_like(r_acc, np.nan), where=w_acc > 0.0)
    return transmission * (mu + visibility * (r_center - mu))


def _fixture_arrays(
    transmission: float,
    visibility: float,
    *,
    masked: bool = False,
    singular: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    window = normalized_hamming_window_2d(2).reshape(-1)
    sample_parts: list[np.ndarray] = []
    reference_parts: list[np.ndarray] = []
    mean_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for frame in range(4):
        yy, xx = np.indices((5, 5), dtype=np.float64)
        if singular:
            reference_patch = np.full((5, 5), 80.0 + 3.0 * frame)
        else:
            texture = np.mod((frame + 1) * (yy + 2.0 * xx), 7.0)
            reference_patch = (
                80.0 + 3.0 * frame + 0.5 * yy + 0.25 * xx + texture
            )
        reference_flat = reference_patch.reshape(-1)
        mean = (
            float(reference_flat[0])
            if singular
            else float(np.sum(window * reference_flat, dtype=np.float64))
        )
        alpha = transmission * visibility
        beta = transmission * (1.0 - visibility)
        sample_flat = alpha * reference_flat + beta * mean
        weights = window.copy()
        if masked:
            mask = ((np.arange(25) + frame) % 7) == 0
            weights[mask] = 0.0
            sample_flat = sample_flat.copy()
            reference_flat = reference_flat.copy()
            sample_flat[mask] = np.nan
            reference_flat[mask] = np.nan
        sample_parts.append(sample_flat)
        reference_parts.append(reference_flat)
        mean_parts.append(np.full(25, mean, dtype=np.float64))
        weight_parts.append(weights)
    return tuple(
        np.ascontiguousarray(np.concatenate(parts), dtype=np.float64)
        for parts in (sample_parts, reference_parts, mean_parts, weight_parts)
    )


def build_scalar_fixtures() -> list[ScalarFixture]:
    specs = [
        ("identity_T1_D1", 1.0, 1.0, False, False, True, True),
        ("known_T082_D063", 0.82, 0.63, False, False, True, True),
        ("masked_known_T091_D074", 0.91, 0.74, True, False, True, True),
        ("unphysical_T090_D120", 0.90, 1.20, False, False, True, False),
        ("near_singular_reference", 0.85, 0.70, False, True, False, False),
    ]
    fixtures = []
    for (
        case_id,
        transmission,
        visibility,
        masked,
        singular,
        numerical,
        physical,
    ) in specs:
        sample, reference, reference_mean, weights = _fixture_arrays(
            transmission, visibility, masked=masked, singular=singular
        )
        fixtures.append(
            ScalarFixture(
                id=case_id,
                sample=sample,
                reference=reference,
                reference_mean=reference_mean,
                weights=weights,
                expected_transmission=transmission if numerical else None,
                expected_visibility=visibility if numerical else None,
                expect_numerical_valid=numerical,
                expect_physical_valid=physical,
            )
        )
    return fixtures


def _fit_payload(case_id: str, fit: UmpaFit) -> dict[str, object]:
    payload = asdict(fit)
    statistics = payload.pop("statistics")
    combined = {"id": case_id, **statistics, **payload}
    for key, value in tuple(combined.items()):
        if isinstance(value, float) and not math.isfinite(value):
            combined[key] = None
    return combined


def python_fixture_report() -> dict[str, object]:
    cases = []
    for fixture in build_scalar_fixtures():
        fit = fit_physical(
            fixture.sample,
            fixture.reference,
            fixture.reference_mean,
            fixture.weights,
        )
        cases.append(_fit_payload(fixture.id, fit))
    return {
        "schema_version": 1,
        "implementation": "python_scalar_float64",
        "analysis_radius": 2,
        "cases": cases,
    }


def compare_reports(
    python_report: dict[str, object],
    cpp_report: dict[str, object],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    numeric_fields = [
        "l1", "l2", "l3", "l4", "l5", "l6", "weight_sum",
        "alpha", "beta", "transmission", "visibility", "cost", "delta",
        "condition",
    ]
    boolean_fields = ["numerical_valid", "physical_valid"]
    py_cases = {case["id"]: case for case in python_report["cases"]}
    cpp_cases = {case["id"]: case for case in cpp_report["cases"]}
    if py_cases.keys() != cpp_cases.keys():
        raise RuntimeError("Python/C++ fixture identifiers differ")
    comparisons = []
    passed = True
    maximum_absolute_difference = 0.0
    for case_id in py_cases:
        field_results = {}
        for field in numeric_fields:
            py_raw = py_cases[case_id][field]
            cpp_raw = cpp_cases[case_id][field]
            if py_raw is None and cpp_raw is None:
                difference = 0.0
                field_passed = True
            else:
                try:
                    py_value = float(py_raw)
                    cpp_value = float(cpp_raw)
                except (TypeError, ValueError):
                    difference = math.inf
                    field_passed = False
                else:
                    difference = abs(py_value - cpp_value)
                    limit = absolute_tolerance + relative_tolerance * abs(py_value)
                    field_passed = math.isfinite(difference) and difference <= limit
            maximum_absolute_difference = max(maximum_absolute_difference, difference)
            field_results[field] = {
                "python": py_raw,
                "cpp": cpp_raw,
                "absolute_difference": difference,
                "passed": field_passed,
            }
            passed = passed and field_passed
        for field in boolean_fields:
            field_passed = bool(py_cases[case_id][field]) == bool(cpp_cases[case_id][field])
            field_results[field] = {
                "python": bool(py_cases[case_id][field]),
                "cpp": bool(cpp_cases[case_id][field]),
                "passed": field_passed,
            }
            passed = passed and field_passed
        comparisons.append({"id": case_id, "fields": field_results})
    return {
        "schema_version": 1,
        "passed": passed,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "case_count": len(comparisons),
        "numeric_field_count": len(numeric_fields),
        "boolean_field_count": len(boolean_fields),
        "maximum_absolute_difference": maximum_absolute_difference,
        "comparisons": comparisons,
        "python": python_report,
        "cpp": cpp_report,
    }


def load_intensity_pair(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = path.resolve()
    if path.suffix.lower() != ".npz":
        raise ValueError("intensity input must be NPZ")
    with np.load(path, allow_pickle=False) as archive:
        sample_key = "sample"
        reference_key = "reference" if "reference" in archive else "ref"
        if sample_key not in archive or reference_key not in archive:
            raise ValueError("input NPZ must contain sample and reference/ref")
        sample = np.asarray(archive[sample_key], dtype=np.float64)
        reference = np.asarray(archive[reference_key], dtype=np.float64)
    return sample, reference, {
        "path": str(path),
        "sha256": sha256(path),
        "sample_key": sample_key,
        "reference_key": reference_key,
        "sample_shape": list(sample.shape),
    }


def load_displacement_pair(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = path.resolve()
    if path.suffix.lower() != ".npz":
        raise ValueError("displacement input must be NPZ")
    with np.load(path, allow_pickle=False) as archive:
        if "displace_x" not in archive or "displace_y" not in archive:
            raise ValueError("displacement NPZ must contain displace_x and displace_y")
        dx = np.asarray(archive["displace_x"], dtype=np.float64)
        dy = np.asarray(archive["displace_y"], dtype=np.float64)
    return dy, dx, {
        "path": str(path),
        "sha256": sha256(path),
        "shape": list(dy.shape),
    }


def parser() -> argparse.ArgumentParser:
    handle = argparse.ArgumentParser(description=__doc__)
    handle.add_argument("--input", type=Path, required=True, help="NPZ with sample/reference")
    handle.add_argument("--wsvt-result", type=Path, required=True, help="NPZ with displace_x/y")
    handle.add_argument("--output", type=Path, required=True)
    handle.add_argument("--analysis-radius", type=int, default=DEFAULT_ANALYSIS_RADIUS)
    handle.add_argument("--origin-yx", type=float, nargs=2, default=(0.0, 0.0))
    return handle


def run_scalar_solver(
    *,
    input_path: Path,
    displacement_path: Path,
    output_path: Path,
    analysis_radius: int = DEFAULT_ANALYSIS_RADIUS,
    origin_yx: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    sample, reference, input_meta = load_intensity_pair(input_path)
    displace_y, displace_x, disp_meta = load_displacement_pair(displacement_path)
    fields = fit_scalar_field(
        sample,
        reference,
        displace_y,
        displace_x,
        analysis_radius=analysis_radius,
        origin_y=float(origin_yx[0]),
        origin_x=float(origin_yx[1]),
    )
    output_path = output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path / "fields.npz", **fields)
    report = {
        "schema_version": 1,
        "method": METHOD_ID,
        "lookup_convention": LOOKUP_CONVENTION,
        "analysis_radius": int(analysis_radius),
        "analysis_window_is_not_search_window": True,
        "constrained_fit": False,
        "input": input_meta,
        "displacement": disp_meta,
        "field_shape": list(displace_y.shape),
        "numerical_valid_fraction": float(np.mean(fields["numerical_valid"])),
        "physical_valid_fraction": float(np.mean(fields["physical_valid"])),
    }
    write_json(output_path / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = run_scalar_solver(
        input_path=args.input,
        displacement_path=args.wsvt_result,
        output_path=args.output,
        analysis_radius=args.analysis_radius,
        origin_yx=(float(args.origin_yx[0]), float(args.origin_yx[1])),
    )
    print(json.dumps({"output": str(args.output.resolve()), **{
        k: report[k] for k in ("method", "numerical_valid_fraction", "physical_valid_fraction")
    }}))
    return 0


def _fixture_main() -> int:
    """Keep the historical umpa_physical_reference CLI when used as that script."""
    import subprocess
    from pathlib import Path as PathType

    parser_local = argparse.ArgumentParser(
        description="Scalar UMPA ModelDF reference fixtures."
    )
    parser_local.add_argument("--cpp-exe", type=PathType)
    parser_local.add_argument("--output", type=PathType, required=True)
    parser_local.add_argument("--absolute-tolerance", type=float, default=1.0e-10)
    parser_local.add_argument("--relative-tolerance", type=float, default=1.0e-10)
    args = parser_local.parse_args()
    python_report = python_fixture_report()
    if args.cpp_exe is None:
        report: dict[str, object] = python_report
    else:
        completed = subprocess.run(
            [str(args.cpp_exe.resolve())],
            check=True,
            capture_output=True,
            text=True,
        )
        cpp_report = json.loads(completed.stdout)
        report = compare_reports(
            python_report,
            cpp_report,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    if args.cpp_exe is not None and not report.get("passed", True):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
