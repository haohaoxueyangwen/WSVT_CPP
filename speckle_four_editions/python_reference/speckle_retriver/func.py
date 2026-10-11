"""Compatibility shim. Prefer ``wsvt.reference.func``."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
_SRC_S = str(_SRC)
if _SRC.is_dir():
    if _SRC_S not in sys.path:
        sys.path.insert(0, _SRC_S)
    parts = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    if _SRC_S not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([_SRC_S, *parts]) if parts else _SRC_S

warnings.warn(
    "Importing speckle_retriver.func is deprecated; use wsvt.reference.func",
    DeprecationWarning,
    stacklevel=2,
)

from wsvt.reference.func import *  # noqa: E402,F401,F403
