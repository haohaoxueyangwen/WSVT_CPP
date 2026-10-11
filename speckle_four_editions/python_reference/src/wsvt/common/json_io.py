from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def finite_json(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


read_json = load_json


def write_json(path: Path | str, payload: Any, *, finite: bool = True) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = finite_json(payload) if finite else payload
    target.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
