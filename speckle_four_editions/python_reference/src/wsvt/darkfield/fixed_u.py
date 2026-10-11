"""Fixed-external-u directional dark-field (single-pair mainline).

Tracker-agnostic: displacement may come from WXST, WSVT, or truth.
Uses author-style C++ Nelder–Mead at rounded integer lookup.
Reports continuous covariance AND discrete kernel moments separately.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from wsvt.common.ddf import covariance_from_abc
from wsvt.darkfield.cpp_fit import NativeAuthorFit, DEFAULT_LIBRARY


def c_round(x: float) -> int:
    """libc round: halfway away from zero."""
    return int(np.copysign(np.floor(abs(float(x)) + 0.5), float(x)))


def discrete_moments_from_abc(a: float, b: float, c: float, radius: int = 8) -> dict[str, float]:
    q = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    ker = np.exp(-a * yy * yy - b * yy * xx - c * xx * xx)
    ker = ker / ker.sum()
    s = float(ker.sum())
    mx = float((ker * xx).sum() / s)
    my = float((ker * yy).sum() / s)
    cxx = float((ker * (xx - mx) ** 2).sum() / s)
    cxy = float((ker * (xx - mx) * (yy - my)).sum() / s)
    cyy = float((ker * (yy - my) ** 2).sum() / s)
    return {
        "kernel_sum": s,
        "mean_x": mx,
        "mean_y": my,
        "cov_xx_px2": cxx,
        "cov_xy_px2": cxy,
        "cov_yy_px2": cyy,
        "kernel_radius": int(radius),
        "kernel_size": 2 * int(radius) + 1,
        **principal_from_cov(cxx, cxy, cyy),
    }


def principal_from_cov(cxx: float, cxy: float, cyy: float) -> dict[str, float]:
    cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float64)
    if not np.isfinite(cov).all():
        return {
            "sigma_major_px": float("nan"),
            "sigma_minor_px": float("nan"),
            "angle_deg": float("nan"),
            "trace_px2": float("nan"),
        }
    values, vectors = np.linalg.eigh(cov)
    if not np.isfinite(values).all():
        return {
            "sigma_major_px": float("nan"),
            "sigma_minor_px": float("nan"),
            "angle_deg": float("nan"),
            "trace_px2": float(cxx + cyy),
        }
    major_vector = vectors[:, 1]
    angle = float(np.degrees(np.arctan2(major_vector[1], major_vector[0])) % 180.0)
    return {
        "sigma_major_px": float(math.sqrt(max(values[1], 0.0))),
        "sigma_minor_px": float(math.sqrt(max(values[0], 0.0))),
        "angle_deg": angle,
        "trace_px2": float(values[0] + values[1]),
    }


def kernel_support_diagnostics(a: float, b: float, c: float, radius: int) -> dict[str, float]:
    """Edge-mass diagnostics on the finite kernel; not a physical scatter claim."""
    q = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    ker = np.exp(-a * yy * yy - b * yy * xx - c * xx * xx)
    total = float(ker.sum())
    if total <= 0.0 or not math.isfinite(total):
        return {
            "edge_mass_frac": float("nan"),
            "boundary_to_peak": float("nan"),
            "kernel_peak": float("nan"),
            "kernel_min": float("nan"),
        }
    kn = ker / total
    ring = (np.abs(yy) == radius) | (np.abs(xx) == radius)
    peak = float(kn.max())
    edge_max = float(kn[ring].max()) if np.any(ring) else float("nan")
    return {
        "edge_mass_frac": float(kn[ring].sum()) if np.any(ring) else float("nan"),
        "boundary_to_peak": (edge_max / peak) if peak > 0.0 else float("nan"),
        "kernel_peak": peak,
        "kernel_min": float(kn.min()),
    }


def predicted_window_stats(
    sample: np.ndarray,
    reference: np.ndarray,
    y: int,
    x: int,
    public_uy: float,
    public_ux: float,
    a: float,
    b: float,
    c: float,
    transmission: float,
    *,
    window_half: int,
    kernel_radius: int,
) -> dict[str, float]:
    """Integer-lattice predicted intensity on the analysis window (no pad/wrap)."""
    from scipy.signal import fftconvolve

    sample = np.asarray(sample, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if sample.ndim == 2:
        sample = sample[None]
    if reference.ndim == 2:
        reference = reference[None]
    lookup_y = c_round(-float(public_uy))
    lookup_x = c_round(-float(public_ux))
    ah, kr = int(window_half), int(kernel_radius)
    q = np.arange(-kr, kr + 1, dtype=np.float64)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    ker = np.exp(-a * yy * yy - b * yy * xx - c * xx * xx)
    ker = ker / ker.sum()
    n = 2 * ah + 1
    w1 = np.hamming(n)
    weights = np.outer(w1, w1)
    weights = weights / weights.sum()
    pred_stack = []
    samp_stack = []
    for fi in range(sample.shape[0]):
        cy = y + lookup_y
        cx = x + lookup_x
        y0, y1 = cy - ah - kr, cy + ah + kr + 1
        x0, x1 = cx - ah - kr, cx + ah + kr + 1
        if y0 < 0 or x0 < 0 or y1 > reference.shape[1] or x1 > reference.shape[2]:
            return {
                "sample_mean": float("nan"),
                "pred_mean": float("nan"),
                "e_rel": float("nan"),
                "weighted_resid_rms": float("nan"),
                "status": "HALO_OOB",
            }
        patch = reference[fi, y0:y1, x0:x1]
        blurred = fftconvolve(patch, ker, mode="valid")
        samp = sample[fi, y - ah : y + ah + 1, x - ah : x + ah + 1]
        pred_stack.append(float(transmission) * blurred)
        samp_stack.append(samp)
    pred = np.stack(pred_stack, axis=0)
    samp = np.stack(samp_stack, axis=0)
    w = weights[None, ...]
    n_frames = float(samp.shape[0])
    sbar = float(np.sum(w * samp) / n_frames)
    num = float(np.sum(w * (samp - pred) ** 2))
    den = float(np.sum(w * (samp - sbar) ** 2))
    return {
        "sample_mean": float(np.mean(samp)),
        "pred_mean": float(np.mean(pred)),
        "e_rel": (num / den) if den > 0.0 else float("nan"),
        "weighted_resid_rms": float(math.sqrt(max(num, 0.0))),
        "status": "OK",
    }


def pack_quality(
    *,
    numerical_valid: bool,
    converged: bool,
    sigma_major: float | None,
    sigma_minor: float | None,
    aniso_ratio_min: float = 1e-3,
) -> dict[str, Any]:
    aniso = None
    if (
        sigma_major is not None
        and sigma_minor is not None
        and math.isfinite(sigma_major)
        and math.isfinite(sigma_minor)
        and (sigma_major + sigma_minor) > 0
    ):
        aniso = (sigma_major - sigma_minor) / (sigma_major + sigma_minor)
    direction_identifiable = bool(
        numerical_valid
        and aniso is not None
        and aniso > aniso_ratio_min
        and math.isfinite(sigma_major)
        and sigma_major < 50.0  # reject continuous-explosion displays
    )
    # Convergence is intentionally NOT part of the display screen.
    review_candidate = bool(numerical_valid and converged and direction_identifiable)
    return {
        "numerical_valid": bool(numerical_valid),
        "optimizer_converged": bool(converged),
        "direction_identifiable": direction_identifiable,
        "direction_display_screen": direction_identifiable,
        "review_candidate": review_candidate,
        "anisotropy": aniso,
        "aniso_ratio_threshold": aniso_ratio_min,
        "note": (
            "direction_identifiable is a display screen only: numerical validity, "
            "anisotropy ratio, and continuous sigma_major<50. It does NOT require "
            "optimizer convergence and is not a confirmed orientation. "
            "review_candidate adds convergence and is still only a review subset."
        ),
    }


def fit_fixed_u_point(
    sample: np.ndarray,
    reference: np.ndarray,
    y: int,
    x: int,
    public_uy: float,
    public_ux: float,
    *,
    library: Path | None = None,
    window_half: int = 8,
    kernel_radius: int = 8,
    max_shift: int | None = None,
    initial: tuple[float, float, float] = (0.0, 0.0, 0.4),
    blur_extra: float = 0.05,
    max_sigma: float = 10.0,
    maxiter: int = 500,
    native_fit: NativeAuthorFit | None = None,
    objective_atol: float | None = None,
    use_direct_residual: bool = False,
) -> dict[str, Any]:
    """Fit author directional kernel at fixed external public displacement.

    public_u = WXST/WSVT sample motion. Author lookup = -public_u, then rounded.
    ``window_half`` is the analysis window (author 8 → 17×17).
    ``kernel_radius`` is the DF kernel support, independent of the analysis window.
    Author-alignment baseline is kernel_radius=8; larger values are application versions.
    """
    kr = int(kernel_radius)
    if kr < 1:
        raise ValueError("kernel_radius must be >= 1")
    sample = np.asarray(sample, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if sample.ndim == 2:
        sample = sample[None]
    if reference.ndim == 2:
        reference = reference[None]
    if sample.shape != reference.shape:
        raise ValueError("sample/reference shape mismatch")

    lookup_y = -float(public_uy)
    lookup_x = -float(public_ux)
    ry, rx = c_round(lookup_y), c_round(lookup_x)
    need = max(abs(ry), abs(rx)) + 2
    ms = int(max_shift) if max_shift is not None else max(16, need)
    if abs(ry) >= ms or abs(rx) >= ms:
        return {
            "status": "REJECT_LOOKUP_OUTSIDE_OPEN_INTERVAL",
            "public_u_yx": [public_uy, public_ux],
            "lookup_yx": [lookup_y, lookup_x],
            "rounded_lookup_yx": [ry, rx],
            "max_shift": ms,
            "window_half": window_half,
            "kernel_radius": kr,
            "numerical_valid": False,
            "optimizer_converged": False,
            "direction_identifiable": False,
        }

    native = native_fit if native_fit is not None else NativeAuthorFit(library or DEFAULT_LIBRARY)
    try:
        r = native.fit(
            sample, reference, y, x, lookup_y, lookup_x,
            window_half=window_half, max_shift=ms,
            initial=initial, blur_extra=blur_extra,
            max_sigma=max_sigma, maxiter=maxiter,
            kernel_radius=kr, objective_atol=objective_atol,
            use_direct_residual=use_direct_residual,
        )
    except Exception as exc:
        return {
            "status": "FAIL",
            "error": str(exc)[:400],
            "public_u_yx": [public_uy, public_ux],
            "lookup_yx": [lookup_y, lookup_x],
            "rounded_lookup_yx": [ry, rx],
            "max_shift": ms,
            "window_half": window_half,
            "kernel_radius": kr,
            "numerical_valid": False,
            "optimizer_converged": False,
            "direction_identifiable": False,
        }

    a, b, c = float(r["a"]), float(r["b"]), float(r["c"])
    cov, maj, minor, angle = covariance_from_abc(a, b, c)
    disc = discrete_moments_from_abc(a, b, c, radius=kr)
    edge = kernel_support_diagnostics(a, b, c, kr)
    pred = predicted_window_stats(
        sample, reference, y, x, public_uy, public_ux, a, b, c, float(r["transmission"]),
        window_half=window_half, kernel_radius=kr,
    )
    numerical = bool(
        math.isfinite(r["transmission"])
        and math.isfinite(r["cost"])
        and math.isfinite(a)
        and math.isfinite(c)
    )
    q = pack_quality(
        numerical_valid=numerical,
        converged=bool(r["converged"]),
        sigma_major=maj,
        sigma_minor=minor,
    )
    angle_display = angle if q["direction_identifiable"] else float("nan")
    return {
        "status": "OK",
        "path": "author_fixed_integer_lookup",
        "public_u_yx": [public_uy, public_ux],
        "lookup_yx": [lookup_y, lookup_x],
        "rounded_lookup_yx": [ry, rx],
        "max_shift": ms,
        "window_half": window_half,
        "kernel_radius": kr,
        "kernel_size": 2 * kr + 1,
        "transform": [r["sx"], r["sy"], r["sigma"]],
        "a": a, "b": b, "c": c,
        "penalty": r["penalty"],
        "transmission": r["transmission"],
        "cost": r["cost"],
        "residual_cost": r["residual_cost"],
        "objective": r.get("objective"),
        "direct_plus_penalty": r.get("direct_plus_penalty"),
        "data_scale": r.get("data_scale"),
        "objective_atol": r.get("objective_atol"),
        "final_dx": r.get("final_dx"),
        "final_df": r.get("final_df"),
        "author_converged": r.get("author_converged"),
        "iterations": r["iterations"],
        "evaluations": r["evaluations"],
        "continuous_cov_xx_px2": float(cov[0, 0]),
        "continuous_cov_xy_px2": float(cov[0, 1]),
        "continuous_cov_yy_px2": float(cov[1, 1]),
        "continuous_sigma_major_px": maj,
        "continuous_sigma_minor_px": minor,
        "continuous_angle_deg": angle,
        "angle_deg_display": angle_display,
        "discrete_moments": disc,
        "kernel_edge": edge,
        "predicted": pred,
        **q,
    }
