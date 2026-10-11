#!/usr/bin/env python3
"""Generate a deterministic, lightweight Python-WSVT stage fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pywt
import scipy


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
PYTHON_REFERENCE_DIR = PROJECT_ROOT / "speckle_retriver"
sys.path.insert(0, str(PYTHON_REFERENCE_DIR))

from WSVT import WSVT  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def make_inputs(seed: int, frames: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    envelope = 25.0 + 0.03 * yy + 0.05 * xx
    ref = np.empty((frames, size, size), dtype=np.float64)
    for frame in range(frames):
        speckle = rng.normal(0.0, 1.0, size=(size, size))
        ref[frame] = envelope + 0.4 * frame + speckle
    img = np.roll(ref, shift=(1, -1), axis=(1, 2)) * 0.985 + 0.25
    return img, ref


def build_solver(img: np.ndarray, ref: np.ndarray, args: argparse.Namespace) -> WSVT:
    return WSVT(
        img.copy(),
        ref.copy(),
        M_image=args.size,
        cal_half_window=args.cal_half_window,
        N_s_extend=args.n_s_extend,
        n_cores=1,
        n_group=1,
        energy=14e3,
        p_x=0.65e-6,
        z=0.5,
        wavelet_level_cut=args.wavelet_level_cut,
        pyramid_level=args.pyramid_level,
        n_iter=1,
        use_wavelet=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--cal-half-window", type=int, default=2)
    parser.add_argument("--n-s-extend", type=int, default=1)
    parser.add_argument("--pyramid-level", type=int, default=1)
    parser.add_argument("--wavelet-level-cut", type=int, default=1)
    args = parser.parse_args()

    if args.frames <= 1 or args.size <= 2 * args.cal_half_window:
        parser.error("fixture dimensions are incompatible with the manual window")

    img, ref = make_inputs(args.seed, args.frames, args.size)
    raw_ref, raw_img = build_solver(img, ref, args).pyramid_data()
    descriptor_ref, descriptor_img = build_solver(img, ref, args).wavelet_data()
    displacement, dpc, phase, transmission = build_solver(img, ref, args).solver()

    arrays: dict[str, np.ndarray] = {
        "input_img_chw": img,
        "input_ref_chw": ref,
        "displace_y_hw": np.asarray(displacement[0]),
        "displace_x_hw": np.asarray(displacement[1]),
        "dpc_y_hw": np.asarray(dpc[0]),
        "dpc_x_hw": np.asarray(dpc[1]),
        "phase_hw": np.asarray(phase),
        "transmission_hw": np.asarray(transmission),
    }
    for level, (ref_level, img_level) in enumerate(zip(raw_ref, raw_img)):
        arrays[f"raw_ref_chw_l{level}"] = np.asarray(ref_level)
        arrays[f"raw_img_chw_l{level}"] = np.asarray(img_level)
    for level, (ref_level, img_level) in enumerate(
        zip(descriptor_ref, descriptor_img)
    ):
        arrays[f"descriptor_ref_hwd_l{level}"] = np.asarray(ref_level)
        arrays[f"descriptor_img_hwd_l{level}"] = np.asarray(img_level)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)

    source_paths = [
        SCRIPT_PATH,
        PYTHON_REFERENCE_DIR / "WSVT.py",
        PYTHON_REFERENCE_DIR / "func.py",
    ]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_command": (
            "python WSVT_CPP/tests/reference/generate_wsvt_reference.py "
            f"--output {args.output.as_posix()}"
        ),
        "semantics_profile": "python_reference_v1",
        "window_policy": "manual_fixed",
        "parameters": {
            "seed": args.seed,
            "frames": args.frames,
            "size": args.size,
            "cal_half_window": args.cal_half_window,
            "n_s_extend": args.n_s_extend,
            "n_template": 0,
            "pyramid_level": args.pyramid_level,
            "wavelet_level_cut": args.wavelet_level_cut,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pywavelets": pywt.__version__,
        },
        "sources": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in source_paths
        },
        "fixture": {
            "path": args.output.name,
            "sha256": sha256_file(args.output),
            "arrays": {
                name: {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "axis_order": (
                        "CHW" if "chw" in name else "HWD" if "hwd" in name else "HW"
                    ),
                    "sha256_c_order": sha256_array(array),
                }
                for name, array in arrays.items()
            },
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
