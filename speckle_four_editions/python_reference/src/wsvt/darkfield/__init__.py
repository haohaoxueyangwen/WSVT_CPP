"""Dark-field solvers.

Scalar T,D (paper Chapter 2): ``wsvt.darkfield.scalar``.
Directional UMPA-DDF author fit: ``cpp_fit`` / ``solver``. These are not the
same parameter set.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "NativeAuthorFit",
    "DEFAULT_LIBRARY",
    "fit_fixed_u_point",
    "load_stack",
    "load_displacement",
    "parser",
    "run_cpp_solver",
    "run_solver",
    "fit_physical",
    "fit_scalar_field",
]


def __getattr__(name: str) -> Any:
    if name in {"NativeAuthorFit", "DEFAULT_LIBRARY", "run_cpp_solver"}:
        from . import cpp_fit

        return getattr(cpp_fit, name)
    if name == "fit_fixed_u_point":
        from . import fixed_u

        return fixed_u.fit_fixed_u_point
    if name in {"load_stack", "load_displacement", "parser", "run_solver"}:
        from . import solver

        return getattr(solver, name)
    if name in {"fit_physical", "fit_scalar_field"}:
        from . import scalar

        return getattr(scalar, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
