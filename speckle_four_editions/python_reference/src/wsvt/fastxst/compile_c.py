"""Compile FastXST ``c_solver.c`` next to this package (and the legacy tree)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from wsvt.common.paths import project_root


def compile_c_solver() -> bool:
    current_dir = Path(__file__).resolve().parent
    c_file = current_dir / "c_solver.c"
    if sys.platform == "win32":
        output_file = current_dir / "c_solver.dll"
        commands = [
            ["gcc", "-shared", "-o", str(output_file), str(c_file), "-O3"],
            ["cl", "/LD", str(c_file), "/Fe" + str(output_file), "/O2"],
        ]
    else:
        output_file = current_dir / "c_solver.so"
        commands = [
            ["gcc", "-shared", "-fPIC", "-o", str(output_file), str(c_file), "-O3"],
        ]

    compiled = False
    for command in commands:
        try:
            subprocess.check_call(command)
            print(f"Successfully compiled {output_file} using {command[0]}")
            compiled = True
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{command[0]} failed or not found")

    if not compiled:
        print(
            "Compilation failed. Please ensure you have a C compiler "
            "(gcc or MSVC) installed and in your PATH."
        )
        return False

    legacy = project_root() / "FastXST-main" / "WaveletSBI" / output_file.name
    try:
        if legacy.parent.is_dir() and legacy.resolve() != output_file.resolve():
            shutil.copy2(output_file, legacy)
            print(f"Copied compiled library to {legacy}")
    except OSError as exc:
        print(f"Warning: could not copy library to legacy tree: {exc}")
    return True


if __name__ == "__main__":
    if not compile_c_solver():
        sys.exit(1)
