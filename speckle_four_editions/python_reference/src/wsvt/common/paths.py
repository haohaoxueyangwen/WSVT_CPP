from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("WSVT_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    # src/wsvt/common/paths.py -> repository root
    candidate = here.parents[3]
    if (candidate / "scripts").is_dir() or (candidate / "docs").is_dir():
        return candidate
    return Path.cwd().resolve()


def cpp_build_dir() -> Path:
    env = os.environ.get("WSVT_CPP_BUILD_DIR")
    if env:
        return Path(env).resolve()
    root = project_root()
    candidates = [
        root / "WSVT_CPP" / "build-ddf-cpp",
        root / "WSVT_CPP" / "build" / "ci",
        root / "WSVT_CPP" / "build" / "dev",
        root / "WSVT_CPP" / "build" / "release-lto",
        root / "WSVT_CPP" / "build" / "release",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def ddf_library() -> Path:
    env = os.environ.get("WSVT_DDF_LIBRARY")
    if env:
        return Path(env).resolve()
    build = cpp_build_dir()
    names = (
        "libwsvt_ddf_author.so",
        "wsvt_ddf_author.dll",
        "libwsvt_ddf_author.dylib",
    )
    for name in names:
        for path in (build / name, build / "Release" / name, build / "Debug" / name):
            if path.is_file():
                return path
    return build / names[0]


def cpp_tool(*parts: str) -> Path:
    """Resolve a built C++ binary under the active build directory."""
    return cpp_build_dir().joinpath(*parts)
