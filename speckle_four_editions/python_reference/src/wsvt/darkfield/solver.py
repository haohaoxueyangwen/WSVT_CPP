#!/usr/bin/env python3
"""Fixed-WSVT-displacement UMPA-DDF raw-intensity solver.

The WSVT displacement is supplied to the official UMPA++ ModelDFKernel
objective. Dark-field fitting is performed by the author's
``solver_at_resolution.optimise_pixel`` implementation. This module does not
use the project's Fourier directional-dark-field estimator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from wsvt.common.ddf import covariance_from_abc
from wsvt.common.hashing import file_sha256 as sha256
from wsvt.common.json_io import finite_json, write_json
from wsvt.common.paths import project_root

ROOT = project_root()
DEFAULT_S1 = ROOT / "bench_results/umpa_ddf_source_20260915"
DEFAULT_S2 = ROOT / "bench_results/umpa_s2_zenodo_6984740/src/optimato-UMPA-337c086"


def load_stack(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a frame-major sample/reference pair without changing values."""
    path = path.resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            sample_key = "sample"
            reference_key = "reference" if "reference" in archive else "ref"
            if sample_key not in archive or reference_key not in archive:
                raise ValueError("input NPZ must contain sample and reference/ref")
            sample = np.asarray(archive[sample_key])
            reference = np.asarray(archive[reference_key])
    elif path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py
        with h5py.File(path, "r") as handle:
            sample = np.asarray(handle["sample"])
            reference_key = "reference" if "reference" in handle else "ref"
            reference = np.asarray(handle[reference_key])
    else:
        raise ValueError("input must be NPZ, H5, or HDF5")
    if sample.ndim != 3 or reference.shape != sample.shape:
        raise ValueError(f"sample/reference must have identical (F,H,W) shape, got {sample.shape} and {reference.shape}")
    if not np.isfinite(sample).all() or not np.isfinite(reference).all():
        raise ValueError("sample/reference contain NaN or Inf")
    sample = np.ascontiguousarray(sample, dtype=np.float64)
    reference = np.ascontiguousarray(reference, dtype=np.float64)
    return sample, reference, {
        "path": str(path), "sha256": sha256(path), "sample_key": "sample",
        "reference_key": reference_key, "shape": list(sample.shape), "input_dtype": str(np.asarray(sample).dtype),
    }


def load_displacement(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = path.resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "displace_x" not in archive or "displace_y" not in archive:
                raise ValueError("displacement NPZ must contain displace_x and displace_y")
            dx = np.asarray(archive["displace_x"])
            dy = np.asarray(archive["displace_y"])
    elif path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py
        with h5py.File(path, "r") as handle:
            dx = np.asarray(handle["displace_x"])
            dy = np.asarray(handle["displace_y"])
    else:
        raise ValueError("displacement input must be NPZ, H5, or HDF5")
    if dx.ndim != 2 or dy.shape != dx.shape:
        raise ValueError("displace_x and displace_y must be identical 2D fields")
    if not np.isfinite(dx).all() or not np.isfinite(dy).all():
        raise ValueError("displacement field contains NaN/Inf")
    return np.ascontiguousarray(dx, dtype=np.float64), np.ascontiguousarray(dy, dtype=np.float64), {
        "path": str(path), "sha256": sha256(path), "shape": list(dx.shape),
    }


def make_roi(y0: int, y1: int, x0: int, x1: int, padding: int, step: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if not (0 <= y0 <= y1 and 0 <= x0 <= x1):
        raise ValueError("ROI must satisfy 0 <= start <= end")
    if y0 < padding or x0 < padding:
        raise ValueError("ROI starts inside the UMPA padded boundary")
    return ((y0 - padding, y1 - padding + 1, step),
            (x0 - padding, x1 - padding + 1, step))


def import_official(s1_dir: Path, s2_dir: Path):
    sys.path.insert(0, str(s2_dir.resolve()))
    sys.path.insert(1, str(s1_dir.resolve()))
    import UMPA  # noqa: F401
    from UMPA_directional_dark_field.single_model import solver_at_resolution
    from UMPA_directional_dark_field.utils import abc_from_transform, abc_from_transform_c_notc
    return solver_at_resolution, abc_from_transform, abc_from_transform_c_notc


def run_solver(args: argparse.Namespace) -> dict[str, Any]:
    sample, reference, input_info = load_stack(args.input)
    dx, dy, displacement_info = load_displacement(args.wsvt_result)
    frames, height, width = sample.shape
    oy, ox = args.displacement_origin_yx
    if oy < 0 or ox < 0:
        raise ValueError("displacement origin must be non-negative")
    solver_at_resolution, abc_from_transform, abc_from_transform_c_notc = import_official(args.s1_dir, args.s2_dir)
    padding = args.max_shift + args.window_half + 8
    if args.roi_yx is None:
        y0, y1, x0, x1 = padding, height - padding - 1, padding, width - padding - 1
    else:
        y0, y1, x0, x1 = args.roi_yx
    roi = make_roi(y0, y1, x0, x1, padding, args.step)

    # ModelDFKernel remains the official dark-field model. Its displacement
    # search is skipped: `model.result` receives WSVT's registered field.
    model = solver_at_resolution(
        [np.ascontiguousarray(frame, dtype=np.float64) for frame in sample],
        [np.ascontiguousarray(frame, dtype=np.float64) for frame in reference],
        step=args.step, Nw=args.window_half, max_shift=args.max_shift,
        max_sig=args.max_sig, blur_extra=args.blur_extra, ROI=roi)
    model.PM.assign_coordinates = "sam"
    sh = tuple(int(v) for v in model.sh)
    raw_y = padding + np.arange(y0 - padding, y1 - padding + 1, args.step, dtype=np.int64)
    raw_x = padding + np.arange(x0 - padding, x1 - padding + 1, args.step, dtype=np.int64)
    raw_y, raw_x = raw_y[:sh[0]], raw_x[:sh[1]]
    if raw_y.size != sh[0] or raw_x.size != sh[1]:
        raise RuntimeError(f"raw coordinate map {raw_y.shape, raw_x.shape} disagrees with UMPA output {sh}")
    iy = raw_y[:, None] - oy
    ix = raw_x[None, :] - ox
    if iy.min() < 0 or ix.min() < 0 or iy.max() >= dx.shape[0] or ix.max() >= dx.shape[1]:
        raise ValueError("requested ROI is outside the displacement field")
    # WSVT's output is sample displacement; UMPA `sam` consumes reference
    # lookup displacement, so the sign conversion is explicit and recorded.
    lookup_ux, lookup_uy = -dx[iy, ix], -dy[iy, ix]
    if not np.isfinite(lookup_ux).all() or not np.isfinite(lookup_uy).all():
        raise ValueError("requested WSVT displacement contains NaN/Inf")
    model.result = {"dx": np.ascontiguousarray(lookup_ux, dtype=np.float64),
                    "dy": np.ascontiguousarray(lookup_uy, dtype=np.float64),
                    "T": np.ones(sh, dtype=np.float64)}
    model.initial_vals = np.zeros(sh + (3,), dtype=np.float64)
    model.initial_vals[:, :, 2] = args.initial_sigma

    displacement_mode = "fixed_wsvt"
    if args.refine_displacement:
        # This is the author's own integer-cost/subpixel minimization path,
        # initialized by WSVT. The returned UMPA displacement is then used by
        # the same official kernel fit below.
        initial_abc = np.ascontiguousarray(abc_from_transform(model.initial_vals, args.blur_extra), dtype=np.float64)
        refined = model.PM.match(
            step=None, abc=initial_abc,
            dxdy=np.stack([lookup_ux, lookup_uy], axis=0), ROI=roi,
            num_threads=args.match_threads, quiet=True)
        if refined["dx"].shape != sh or refined["dy"].shape != sh:
            raise RuntimeError("official UMPA refinement returned an unexpected displacement shape")
        model.result = refined
        lookup_ux = np.ascontiguousarray(refined["dx"], dtype=np.float64)
        lookup_uy = np.ascontiguousarray(refined["dy"], dtype=np.float64)
        displacement_mode = "umpa_refined_from_wsvt"

    started = time.perf_counter()
    model.optimise_image(mode=args.mode, maxiter=args.maxiter, tol=args.tolerance,
                         transmission_threshold=args.transmission_threshold)
    elapsed = time.perf_counter() - started
    transforms = np.asarray(model.final_vals, dtype=np.float64)
    sx, sy, sigma = (transforms[:, :, i] for i in range(3))
    map_names = ("a", "b", "c", "transmission", "cost", "cov_xx_px2", "cov_xy_px2",
                 "cov_yy_px2", "sigma_major_px", "sigma_minor_px", "angle_deg",
                 "penalty", "rounded_lookup_x", "rounded_lookup_y", "fit_valid")
    maps = {name: np.full(sh, np.nan, dtype=np.float64) for name in map_names}
    fit_count = 0
    for row in range(sh[0]):
        for col in range(sh[1]):
            aa, bb, cc, penalty = abc_from_transform_c_notc(
                float(sx[row, col]), float(sy[row, col]), float(sigma[row, col]),
                t=args.blur_extra, max_sig=args.max_sig)
            covariance, major, minor, angle = covariance_from_abc(float(aa), float(bb), float(cc))
            rounded_y = int(np.floor(float(lookup_uy[row, col]) + 0.5))
            rounded_x = int(np.floor(float(lookup_ux[row, col]) + 0.5))
            cost, transmission = model.PM.cost(int(raw_y[row]), int(raw_x[col]), rounded_y, rounded_x,
                                                float(aa), float(bb), float(cc))
            values = (aa, bb, cc, transmission, cost, covariance[0, 0], covariance[0, 1], covariance[1, 1],
                      major, minor, angle, penalty, rounded_x, rounded_y)
            for name, value in zip(map_names[:-1], values):
                maps[name][row, col] = value
            valid = bool(np.isfinite(values).all())
            maps["fit_valid"][row, col] = valid
            fit_count += int(valid)

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    np.savez_compressed(output / "wsvt_umpa_ddf.npz", **maps, sx=sx, sy=sy, sigma=sigma,
                        lookup_ux=lookup_ux, lookup_uy=lookup_uy, raw_y=raw_y, raw_x=raw_x)
    field_names = ["raw_y", "raw_x", "lookup_uy", "lookup_ux", "sx", "sy", "sigma", *maps.keys()]
    with (output / "wsvt_umpa_ddf.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        writer.writeheader()
        for row in range(sh[0]):
            for col in range(sh[1]):
                record = {"raw_y": int(raw_y[row]), "raw_x": int(raw_x[col]),
                          "lookup_uy": float(lookup_uy[row, col]), "lookup_ux": float(lookup_ux[row, col]),
                          "sx": float(sx[row, col]), "sy": float(sy[row, col]), "sigma": float(sigma[row, col])}
                record.update({name: float(maps[name][row, col]) for name in maps})
                writer.writerow(finite_json(record))
    source_files = [Path(__file__), args.input, args.wsvt_result]
    write_json(output / "params.json", {
        "method": "official UMPA-DDF ModelDFKernel with fixed WSVT registration",
        "fourier_directional_darkfield_used": False, "input": input_info,
        "wsvt_result": displacement_info, "displacement_origin_yx": [oy, ox],
        "wsvt_to_umpa_sign": {"lookup_ux": "-displace_x", "lookup_uy": "-displace_y"},
        "frames": frames, "raw_shape": [height, width], "raw_centers_y": raw_y.tolist(),
        "raw_centers_x": raw_x.tolist(), "window_half": args.window_half,
        "window_size": 2 * args.window_half + 1, "max_shift": args.max_shift,
        "kernel_support": "official ModelDFKernel 17x17 discrete normalized Gaussian",
        "blur_extra": args.blur_extra, "max_sig": args.max_sig, "step": args.step,
        "mode": args.mode, "maxiter": args.maxiter, "tolerance": args.tolerance,
        "assign_coordinates": "sam", "displacement_mode": displacement_mode,
        "match_threads": args.match_threads,
        "physical_units": {"covariance": "pixel^2", "angle": "degrees modulo 180"},
        "source_paths": {"s1": str(args.s1_dir.resolve()), "s2": str(args.s2_dir.resolve())},
        "source_sha256": {str(path): sha256(path) for path in source_files},
    })
    report = {"status": "completed", "sampled_points": int(np.prod(sh)), "fit_valid_count": fit_count,
              "fit_valid_fraction": fit_count / float(np.prod(sh)), "solver_time_s": elapsed,
              "formal_timing": False, "physical_accuracy_claim": False,
              "production_pipeline_changed": False, "output": str(output)}
    write_json(output / "bench_report.json", report)
    write_json(output / "artifact_index.json", {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                                                 for path in output.iterdir() if path.is_file()})
    return report


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="NPZ/HDF5 with sample and reference stacks")
    ap.add_argument("--wsvt-result", type=Path, required=True, help="NPZ/HDF5 with displace_x/displace_y")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--displacement-origin-yx", type=int, nargs=2, default=(0, 0))
    ap.add_argument("--window-half", type=int, default=8)
    ap.add_argument("--max-shift", type=int, default=16)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--roi-yx", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--initial-sigma", type=float, default=0.4)
    ap.add_argument("--blur-extra", type=float, default=0.05)
    ap.add_argument("--max-sig", type=float, default=10.0)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--tolerance", type=float, default=1e-15)
    ap.add_argument("--transmission-threshold", type=float, default=5.99)
    ap.add_argument("--mode", choices=("coordinate_descent", "gradient_descent"), default="gradient_descent")
    ap.add_argument("--refine-displacement", action="store_true",
                    help="initialize the author's UMPA displacement minimization from WSVT")
    ap.add_argument("--match-threads", type=int, default=1)
    ap.add_argument("--s1-dir", type=Path, default=DEFAULT_S1)
    ap.add_argument("--s2-dir", type=Path, default=DEFAULT_S2)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.window_half < 0 or args.max_shift <= 0 or args.step <= 0:
        raise SystemExit("window-half, max-shift and step must be positive/non-negative")
    if args.maxiter <= 0 or args.max_sig <= 0 or args.initial_sigma < 0 or args.match_threads <= 0:
        raise SystemExit("maxiter/max-sig must be positive and initial-sigma non-negative")
    if args.roi_yx is not None:
        args.roi_yx = tuple(args.roi_yx)
    args.displacement_origin_yx = tuple(args.displacement_origin_yx)
    print(json.dumps(run_solver(args), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
