"""FastXST (WaveletSBI) C/GPU comparison path. Solver math is unchanged."""

from __future__ import annotations

from typing import Any

__all__ = ["WSVT", "WXST", "USE_C_SOLVER", "c_lib", "main_wsvt", "main_wxst"]


def __getattr__(name: str) -> Any:
    if name == "WSVT":
        from .WSVT import WSVT

        return WSVT
    if name == "WXST":
        from .WXST import WXST

        return WXST
    if name in {"USE_C_SOLVER", "c_lib"}:
        from .backends import load_c_solver

        lib, loaded = load_c_solver()
        return loaded if name == "USE_C_SOLVER" else lib
    if name == "main_wsvt":
        from .WSVT import main

        return main
    if name == "main_wxst":
        from .WXST import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
