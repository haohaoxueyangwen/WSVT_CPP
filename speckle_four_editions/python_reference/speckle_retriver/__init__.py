"""Compatibility shim. Prefer ``wsvt.reference`` for WSVT/WXST/func.

``directional_darkfield`` remains in this directory and is imported as a
submodule. Solver symbols are loaded lazily so dark-field-only imports do not
pull PyWavelets/Numba.
"""

from __future__ import annotations

import warnings
from typing import Any

from . import _bootstrap  # noqa: F401

warnings.warn(
    "speckle_retriver is a compatibility shim; import wsvt.reference instead",
    DeprecationWarning,
    stacklevel=2,
)

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
]


def __getattr__(name: str) -> Any:
    if name == "WSVT":
        from wsvt.reference.WSVT import WSVT

        return WSVT
    if name == "WXST":
        from wsvt.reference.WXST import WXST

        return WXST
    if name == "dist_numba":
        from wsvt.reference.euclidean_dist import dist_numba

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
        from wsvt.reference import func as _func

        return getattr(_func, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
