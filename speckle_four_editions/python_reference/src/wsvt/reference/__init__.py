"""Python-reference WSVT/WXST baseline (moved from ``speckle_retriver/``).

Solver math is unchanged. Prefer these imports over the deprecated
``speckle_retriver`` shim. Submodules stay lazy so ``import wsvt.reference``
does not pull PyWavelets/Numba until a solver symbol is used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "WSVT",
    "WXST",
    "dist_numba",
    "find_disp",
    "frankotchellappa",
    "image_preprocess",
    "image_roi",
    "load_image",
    "load_images",
    "Wavelet_transform",
    "main_wsvt",
    "main_wxst",
]


def __getattr__(name: str) -> Any:
    if name == "WSVT":
        from .WSVT import WSVT

        return WSVT
    if name == "WXST":
        from .WXST import WXST

        return WXST
    if name == "dist_numba":
        from .euclidean_dist import dist_numba

        return dist_numba
    if name in {
        "find_disp",
        "frankotchellappa",
        "image_preprocess",
        "image_roi",
        "load_image",
        "load_images",
        "Wavelet_transform",
    }:
        from . import func as _func

        return getattr(_func, name)
    if name == "main_wsvt":
        from .WSVT import main

        return main
    if name == "main_wxst":
        from .WXST import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
