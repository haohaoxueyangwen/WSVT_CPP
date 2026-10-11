#!/usr/bin/env python3
"""Compare C++ debug-stage exports against the lightweight Python fixture."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | list[int]]:
    actual64 = np.asarray(actual, dtype=np.float64)
    expected64 = np.asarray(expected, dtype=np.float64)
    if actual64.shape != expected64.shape:
        raise ValueError(f"shape mismatch: {actual64.shape} != {expected64.shape}")
    delta = actual64 - expected64
    rmse = float(np.sqrt(np.mean(delta * delta)))
    scale = float(np.sqrt(np.mean(expected64 * expected64)))
    centered_actual = actual64.ravel() - float(np.mean(actual64))
    centered_expected = expected64.ravel() - float(np.mean(expected64))
    denominator = float(
        np.linalg.norm(centered_actual) * np.linalg.norm(centered_expected)
    )
    correlation = (
        float(np.dot(centered_actual, centered_expected) / denominator)
        if denominator > 0.0
        else 1.0
    )
    return {
        "shape": list(actual64.shape),
        "rmse": rmse,
        "relative_rmse": rmse / max(scale, np.finfo(np.float64).eps),
        "max_abs_error": float(np.max(np.abs(delta))),
        "correlation": correlation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--pyramid-level", type=int, default=1)
    parser.add_argument("--max-relative-rmse", type=float, default=2.0e-5)
    parser.add_argument("--min-correlation", type=float, default=0.999999)
    args = parser.parse_args()

    fixture = np.load(args.fixture)
    temporary = Path(tempfile.mkdtemp(prefix="wsvt-stage-compare."))
    try:
        img_h5 = temporary / "img.h5"
        ref_h5 = temporary / "ref.h5"
        with h5py.File(img_h5, "w") as handle:
            handle.create_dataset("stack", data=fixture["input_img_chw"].astype(np.float32))
        with h5py.File(ref_h5, "w") as handle:
            handle.create_dataset("stack", data=fixture["input_ref_chw"].astype(np.float32))

        output_dir = temporary / "cpp"
        command = [
            str(args.executable),
            "wsvt_stage",
            str(img_h5),
            "stack",
            str(ref_h5),
            "stack",
            str(output_dir),
            "--crop",
            "0",
            "--cal_half_window",
            "2",
            "--n_template",
            "0",
            "--n_s_extend",
            "1",
            "--n_cores",
            "1",
            "--pyramid_level",
            str(args.pyramid_level),
            "--wavelet_level_cut",
            "1",
            "--h5_deflate",
            "0",
        ]
        completed = subprocess.run(
            command, check=True, text=True, capture_output=True
        )

        comparisons: dict[str, dict[str, float | list[int] | bool]] = {}
        with h5py.File(output_dir / "WSVT_stage.hdf5", "r") as cpp:
            for role in ("img", "ref"):
                for level in range(args.pyramid_level + 1):
                    cases = {
                        f"prewavelet_{role}_hwd_l{level}": np.moveaxis(
                            fixture[f"raw_{role}_chw_l{level}"], 0, -1
                        ),
                        f"descriptor_{role}_hwd_l{level}": fixture[
                            f"descriptor_{role}_hwd_l{level}"
                        ],
                    }
                    for key, expected in cases.items():
                        result = metrics(cpp[key][...], expected)
                        result["pass"] = bool(
                            result["relative_rmse"] <= args.max_relative_rmse
                            and result["correlation"] >= args.min_correlation
                        )
                        comparisons[key] = result

            for key in (
                "displace_y_hw",
                "displace_x_hw",
                "dpc_y_hw",
                "dpc_x_hw",
                "phase_hw",
                "transmission_hw",
            ):
                actual = cpp[key][...]
                expected = fixture[key]
                result = metrics(actual, expected)
                # Final fields also include spline/subpixel/FFTW float32 effects;
                # keep the screening threshold explicit and separate from the
                # tighter prewavelet/descriptor gate.
                result["pass"] = bool(
                    result["relative_rmse"] <= 5.0e-2
                    and result["correlation"] >= 0.99
                )
                if actual.ndim == 2:
                    result["masked"] = {
                        str(margin): metrics(
                            actual[margin:-margin, margin:-margin],
                            expected[margin:-margin, margin:-margin],
                        )
                        for margin in (1, 2, 4, 6)
                        if 2 * margin < min(actual.shape)
                    }
                if key.startswith("displace_"):
                    limit = 2.0
                    result["cpp_search_limit_fraction"] = float(
                        np.mean(np.abs(actual) >= limit - 1.0e-6)
                    )
                    result["python_search_limit_fraction"] = float(
                        np.mean(np.abs(expected) >= limit - 1.0e-6)
                    )
                comparisons[key] = result

        report = {
            "schema_version": 1,
            "fixture": str(args.fixture),
            "executable": str(args.executable),
            "command": command,
            "stdout": completed.stdout,
            "thresholds": {
                "max_relative_rmse": args.max_relative_rmse,
                "min_correlation": args.min_correlation,
            },
            "comparisons": comparisons,
            "pass": all(bool(item["pass"]) for item in comparisons.values()),
        }
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if report["pass"] else 1
    finally:
        shutil.rmtree(temporary)


if __name__ == "__main__":
    raise SystemExit(main())
