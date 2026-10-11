"""C-solver load and wavelet-backend choice for FastXST.

Solver math is unchanged. This module is the single copy of the ctypes
bootstrap that used to live in both ``WSVT.py`` and ``WXST.py``.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from numpy.ctypeslib import ndpointer

from wsvt.common.paths import project_root

c_lib = None
USE_C_SOLVER = False
_LOADED = False


def _library_name() -> str:
    return "c_solver.dll" if sys.platform == "win32" else "c_solver.so"


def _log(message: str, color: str) -> None:
    try:
        from .func import prColor

        prColor(message, color)
    except Exception:
        print(message)


def _candidate_paths(name: str) -> list[Path]:
    env = os.environ.get("WSVT_FASTXST_CSOLVER")
    here = Path(__file__).resolve().parent
    root = project_root()
    paths = []
    if env:
        paths.append(Path(env))
    paths.extend(
        [
            here / name,
            here / "Release" / name,
            here / "Debug" / name,
            root / "FastXST-main" / "WaveletSBI" / name,
        ]
    )
    return paths


def load_c_solver():
    """Load ``c_solver`` via ctypes. Missing library falls back to Python/Numba."""
    global c_lib, USE_C_SOLVER, _LOADED
    if _LOADED:
        return c_lib, USE_C_SOLVER
    _LOADED = True
    name = _library_name()
    last_error: Exception | None = None
    for path in _candidate_paths(name):
        if not path.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(path))
            lib.calculate_displacement_block.argtypes = [
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ]
            lib.calculate_displacement_block.restype = None
            lib.wavelet_transform_db2.argtypes = [
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                ctypes.c_int,
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ]
            lib.wavelet_transform_db2.restype = None
            c_lib = lib
            USE_C_SOLVER = True
            _log(f"C solver loaded successfully ({path})", "green")
            return lib, True
        except Exception as exc:
            last_error = exc
    message = f"Failed to load C solver: {last_error}" if last_error else (
        f"Failed to load C solver: {name} not found"
    )
    _log(message, "red")
    c_lib = None
    USE_C_SOLVER = False
    return None, False


def choose_c_wavelet_backend(
    wavelet_backend: str,
    use_c: bool,
    wavelet_method: str,
    use_c_solver: bool,
) -> bool:
    """Same selection rule previously inlined in WSVT.py and WXST.py."""
    if wavelet_backend == "c":
        use_c_backend = wavelet_method == "db2" and use_c_solver
        if not use_c_backend:
            _log(
                "wavelet_backend=c requested but C db2 solver unavailable, fallback to pywt",
                "yellow",
            )
        return use_c_backend
    if wavelet_backend == "pywt":
        return False
    return bool(use_c and wavelet_method == "db2" and use_c_solver)
