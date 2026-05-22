import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Parameter schema: name -> (type, default)
# Used for validation and config normalization.
# ---------------------------------------------------------------------------
PARAM_SCHEMA = {
    "crop": (int, 512),
    "m_image": (int, 512),
    "n_s": (int, 5),
    "cal_half_window": (int, 20),
    "n_template": (int, 0),
    "n_s_extend": (int, 4),
    "n_cores": (int, 4),
    "n_group": (int, 4),
    "energy": (float, 14000.0),
    "p_x": (float, 6.5e-7),
    "mag_factor": (float, 1.0),
    "z": (float, 0.5),
    "wavelet_level_cut": (int, 2),
    "pyramid_level": (int, 2),
    "n_iter": (int, 1),
    "use_estimate": (bool, False),
    "use_wavelet": (bool, True),
    "use_gpu": (bool, False),
    "save_img": (bool, False),
    "cleansave": (bool, False),
}

MODE_RESULT_FILE = {
    "wxst": "WXST_result.hdf5",
    "wsvt": "WSVT_result.hdf5",
    "wxst_dir": "WXST_result.hdf5",
    "wsvt_dir": "WSVT_result.hdf5",
}

VALID_MODES = ("demo", "wxst", "wsvt", "wxst_dir", "wsvt_dir")


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def check_export_deps():
    missing = []
    for mod in ("numpy", "h5py"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append("Pillow")
    try:
        import tifffile  # noqa: F401
    except ImportError:
        missing.append("tifffile")
    if missing:
        raise ImportError(
            f"export deps missing: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _normalize_float_to_uint16(arr):
    import numpy as np
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint16)
    safe = np.where(finite, arr, 0.0)
    vmin = float(np.min(safe[finite]))
    vmax = float(np.max(safe[finite]))
    if vmax <= vmin:
        return np.zeros(arr.shape, dtype=np.uint16)
    scaled = (safe - vmin) / (vmax - vmin)
    return np.clip(np.rint(scaled * 65535.0), 0, 65535).astype(np.uint16)


def _save_array_png(arr, out_path):
    from PIL import Image
    Image.fromarray(arr).save(str(out_path))


def _save_array_tiff(arr, out_path):
    import tifffile
    tifffile.imwrite(str(out_path), arr)


def export_results(out_dir, result_file, fmt_set, *,
                   log: Callable[[str], None] = print):
    import numpy as np
    import h5py

    p = Path(out_dir).expanduser().resolve()
    rf = p / result_file
    if not rf.exists():
        log(f"[export] result file not found, skip: {rf}")
        return []

    saved = []
    skipped = []
    with h5py.File(str(rf), "r") as f:
        for key in f.keys():
            data = np.asarray(f[key], dtype=np.float32)
            if data.ndim != 2:
                skipped.append((key, data.ndim, data.shape))
                continue
            if "png" in fmt_set:
                dst = p / f"{key}.png"
                _save_array_png(_normalize_float_to_uint16(data), dst)
                saved.append(str(dst))
            if "tiff" in fmt_set:
                dst = p / f"{key}.tiff"
                _save_array_tiff(data, dst)
                saved.append(str(dst))

    if saved:
        log(f"[export] exported {len(saved)} file(s) to {p}:")
        for s in saved:
            log(f"  - {s}")
    else:
        log(f"[export] no 2D dataset found in: {rf}")

    if skipped:
        for key, ndim, shape in skipped:
            log(f"[export] skipped non-2D dataset: {key} (ndim={ndim}, shape={shape})")

    return saved


# ---------------------------------------------------------------------------
# Config & parameter helpers
# ---------------------------------------------------------------------------

def load_config(path):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a JSON object")
    return cfg


def resolve_path(value, base_dir=None):
    p = Path(value).expanduser()
    if not p.is_absolute() and base_dir:
        p = Path(base_dir) / p
    return p.resolve()


def coerce_param(key, value):
    """Coerce a string value to the expected type per PARAM_SCHEMA."""
    if key not in PARAM_SCHEMA:
        return value
    expected_type, _ = PARAM_SCHEMA[key]
    if expected_type is bool:
        v = str(value).strip().lower()
        if v in ("1", "true", "on", "yes"):
            return True
        if v in ("0", "false", "off", "no"):
            return False
        raise ValueError(f"invalid bool for {key}: {value!r}")
    try:
        return expected_type(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"invalid value for {key}: expected {expected_type.__name__}, got {value!r}") from e


def parse_set_items(items):
    """Parse --set key=value items into [--key, value] list with validation."""
    out = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"invalid --set item: {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"invalid --set key: {item}")
        coerce_param(k, v)  # validate before passing to C++
        out.extend([f"--{k}", v])
    return out


def validate_config(cfg: dict):
    """Validate a resolved config dict. Raises ValueError on problems."""
    mode = cfg.get("mode", "")
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got: {mode!r}")
    if mode == "demo":
        return
    if mode in ("wxst_dir", "wsvt_dir"):
        if not cfg.get("img_dir"):
            raise ValueError("directory mode requires img_dir")
        if not cfg.get("ref_dir"):
            raise ValueError("directory mode requires ref_dir")
        if not cfg.get("out_dir"):
            raise ValueError("directory mode requires out_dir")
    elif mode in ("wxst", "wsvt"):
        if not cfg.get("img_h5"):
            raise ValueError("hdf5 mode requires img_h5")
        if not cfg.get("ref_h5"):
            raise ValueError("hdf5 mode requires ref_h5")
        cfg.setdefault("img_key", "img")
        cfg.setdefault("ref_key", "ref")
    if not cfg.get("out_dir"):
        raise ValueError("out_dir is required")

    for k, v in cfg.get("set", {}).items():
        coerce_param(k, v)


# ---------------------------------------------------------------------------
# Resolve executable
# ---------------------------------------------------------------------------

def resolve_exe(user_exe=None, base_dir=None):
    if user_exe:
        p = resolve_path(user_exe, base_dir)
        if not p.exists():
            raise FileNotFoundError(f"executable not found: {p}")
        return p
    root = Path(__file__).resolve().parent
    names = ["wsvt_cli.exe"] if os.name == "nt" else ["wsvt_cli", "wsvt_cli.exe"]
    candidates = []
    for name in names:
        candidates.extend([
            root / "build" / name,
            root / "build" / "Release" / name,
            root / "build" / "Debug" / name,
            root / "cmake-build-release" / name,
            root / "cmake-build-debug" / name,
            root / "build-opencv" / name,
        ])
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("cannot find wsvt_cli, please pass --exe")


# ---------------------------------------------------------------------------
# Build command line
# ---------------------------------------------------------------------------

def build_command(cfg: dict) -> list[str]:
    """Build the wsvt_cli command from a resolved config dict."""
    base_dir = cfg.get("_config_dir")
    exe = resolve_exe(cfg.get("exe"), base_dir)
    mode = cfg["mode"]

    if mode == "demo":
        return [str(exe), "demo"]

    set_args = []
    for k, v in cfg.get("set", {}).items():
        if isinstance(v, bool):
            v = "true" if v else "false"
        set_args.extend([f"--{k}", str(v)])

    if mode in ("wxst_dir", "wsvt_dir"):
        cmd = [
            str(exe), mode,
            str(resolve_path(cfg["img_dir"], base_dir)),
            str(resolve_path(cfg["ref_dir"], base_dir)),
            str(resolve_path(cfg["out_dir"], base_dir)),
        ]
    else:
        cmd = [
            str(exe), mode,
            str(resolve_path(cfg["img_h5"], base_dir)),
            cfg.get("img_key", "img"),
            str(resolve_path(cfg["ref_h5"], base_dir)),
            cfg.get("ref_key", "ref"),
            str(resolve_path(cfg["out_dir"], base_dir)),
        ]
    cmd.extend(set_args)
    return cmd


# ---------------------------------------------------------------------------
# Core pipeline runner — the public API for GUI / script integration
# ---------------------------------------------------------------------------

def run_pipeline(
    cfg: dict,
    *,
    dry_run: bool = False,
    export: bool = True,
    export_fmt: set[str] | None = None,
    log: Callable[[str], None] = print,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run the WSVT pipeline from a config dict.

    Args:
        cfg: Resolved config dict (mode, paths, set params). See validate_config.
        dry_run: If True, only build and print the command without executing.
        export: Whether to export HDF5 results to PNG/TIFF after run.
        export_fmt: Set of formats to export, e.g. {"png", "tiff"}. Default both.
        log: Logging callable for normal messages.
        progress_callback: Called with each line of C++ subprocess output
            in real time. Useful for GUI progress display.

    Returns:
        dict with keys:
            - returncode: subprocess exit code (0 = success)
            - command: the full command list that was run
            - exported: list of exported file paths (empty if no export)
    """
    validate_config(cfg)
    cmd = build_command(cfg)
    log("command: " + " ".join(shlex.quote(c) for c in cmd))

    if dry_run:
        return {"returncode": 0, "command": cmd, "exported": []}

    child_env = os.environ.copy()
    if os.name != "nt":
        conda_prefix = child_env.get("CONDA_PREFIX", "").strip()
        if conda_prefix:
            lib_dir = str(Path(conda_prefix) / "lib")
            ld_library_path = child_env.get("LD_LIBRARY_PATH", "")
            if ld_library_path:
                existing = ld_library_path.split(":")
                if lib_dir not in existing:
                    child_env["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_library_path}"
            else:
                child_env["LD_LIBRARY_PATH"] = lib_dir

    # Stream C++ output line-by-line for real-time feedback
    proc = subprocess.Popen(
        cmd, env=child_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    output_lines = []

    def _reader():
        for line in proc.stdout:
            line = line.rstrip("\n")
            output_lines.append(line)
            log(line)
            if progress_callback:
                progress_callback(line)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()
    proc.wait()
    reader_thread.join(timeout=5)

    result = {
        "returncode": proc.returncode,
        "command": cmd,
        "exported": [],
        "output": output_lines,
    }

    # Export results on success
    if proc.returncode == 0 and export and cfg["mode"] in MODE_RESULT_FILE:
        if export_fmt is None:
            export_fmt = {"png", "tiff"}
        base_dir = cfg.get("_config_dir")
        out_dir = str(resolve_path(cfg["out_dir"], base_dir))
        result["exported"] = export_results(
            out_dir, MODE_RESULT_FILE[cfg["mode"]], export_fmt, log=log,
        )

    return result


# ---------------------------------------------------------------------------
# Config resolution: merge CLI args + config file into a single dict
# ---------------------------------------------------------------------------

def resolve_config(args) -> dict:
    """Merge CLI arguments and config file into a unified config dict."""
    cfg = {}

    if args.config:
        file_cfg = load_config(args.config)
        cfg.update(file_cfg)
        cfg["_config_dir"] = str(Path(args.config).expanduser().resolve().parent)

    # CLI overrides config file values
    if args.mode:
        cfg["mode"] = args.mode
    if getattr(args, "img_dir", None):
        cfg["img_dir"] = args.img_dir
    if getattr(args, "ref_dir", None):
        cfg["ref_dir"] = args.ref_dir
    if getattr(args, "img_h5", None):
        cfg["img_h5"] = args.img_h5
    if getattr(args, "img_key", None):
        cfg["img_key"] = args.img_key
    if getattr(args, "ref_h5", None):
        cfg["ref_h5"] = args.ref_h5
    if getattr(args, "ref_key", None):
        cfg["ref_key"] = args.ref_key
    if getattr(args, "out_dir", None):
        cfg["out_dir"] = args.out_dir
    if getattr(args, "exe", None):
        cfg["exe"] = args.exe

    # Merge --set items into cfg["set"]
    cli_set_items = getattr(args, "set_items", None) or []
    if cli_set_items:
        existing_set = cfg.get("set", {})
        if not isinstance(existing_set, dict):
            existing_set = {}
        for item in cli_set_items:
            if "=" not in item:
                raise ValueError(f"invalid --set item: {item}")
            k, v = item.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                raise ValueError(f"invalid --set key: {item}")
            existing_set[k] = coerce_param(k, v)
        cfg["set"] = existing_set

    # Apply defaults from PARAM_SCHEMA for missing set params
    resolved_set = dict(cfg.get("set", {}))
    for k, (_, default) in PARAM_SCHEMA.items():
        resolved_set.setdefault(k, default)
    cfg["set"] = resolved_set

    return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_parser():
    parser = argparse.ArgumentParser(
        prog="launch.py",
        description="WSVT_CPP launcher — run wsvt_cli via config file or CLI args",
    )
    parser.add_argument("--config", default="", help="path to JSON config file")
    parser.add_argument("--exe", default="", help="path to wsvt_cli executable")
    parser.add_argument("--dry-run", action="store_true", help="print command without running")
    parser.add_argument("--no-export", action="store_true", default=False,
                        help="skip PNG/TIFF export after C++ run")
    parser.add_argument("--show-config", action="store_true",
                        help="print resolved config and exit")
    parser.add_argument("--mode", choices=VALID_MODES, default="",
                        help="processing mode")
    parser.add_argument("--img-dir", default="", help="sample image directory")
    parser.add_argument("--ref-dir", default="", help="reference image directory")
    parser.add_argument("--img-h5", default="", help="sample HDF5 file")
    parser.add_argument("--ref-h5", default="", help="reference HDF5 file")
    parser.add_argument("--img-key", default="", help="dataset key in sample HDF5")
    parser.add_argument("--ref-key", default="", help="dataset key in reference HDF5")
    parser.add_argument("--out-dir", default="", help="output directory")
    parser.add_argument("--set", dest="set_items", action="append", default=[],
                        help="override param, e.g. --set n_cores=16 (repeatable)")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    if not args.config and not args.mode:
        parser.print_help()
        print("\nError: --config or --mode is required")
        return 2

    cfg = resolve_config(args)

    if args.show_config:
        # Strip internal keys for display
        display = {k: v for k, v in cfg.items() if not k.startswith("_")}
        print(json.dumps(display, indent=2, default=str))
        return 0

    result = run_pipeline(
        cfg,
        dry_run=args.dry_run,
        export=not args.no_export,
    )
    return result["returncode"]


if __name__ == "__main__":
    sys.exit(main())
