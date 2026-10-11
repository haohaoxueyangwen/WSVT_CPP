"""Put ``src/`` on ``sys.path`` and ``PYTHONPATH`` so shims can import ``wsvt``.

Spawned ``ProcessPoolExecutor`` workers re-import ``wsvt.reference.WSVT``.
Updating ``PYTHONPATH`` keeps that working when the parent only mutated
``sys.path`` (Windows spawn; Linux fork already inherits ``sys.path``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
_SRC_S = str(_SRC)
if _SRC.is_dir():
    if _SRC_S not in sys.path:
        sys.path.insert(0, _SRC_S)
    parts = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    if _SRC_S not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([_SRC_S, *parts]) if parts else _SRC_S
