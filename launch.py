import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


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


def export_results(out_dir, result_file, fmt_set):
    import numpy as np
    import h5py
    p = Path(out_dir).expanduser().resolve()
    rf = p / result_file
    if not rf.exists():
        print(f"[export] result file not found, skip: {rf}")
        return []
    saved = []
    with h5py.File(str(rf), "r") as f:
        for key in f.keys():
            data = np.asarray(f[key], dtype=np.float32)
            if data.ndim != 2:
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
        print(f"[export] exported {len(saved)} file(s) to {p}:")
        for s in saved:
            print(f"  - {s}")
    else:
        print(f"[export] no 2D dataset found in: {rf}")
    return saved


MODE_RESULT_FILE = {
    "wxst": "WXST_result.hdf5",
    "wsvt": "WSVT_result.hdf5",
    "wxst_dir": "WXST_result.hdf5",
    "wsvt_dir": "WSVT_result.hdf5",
}


def parse_set_items(items):
    out = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"invalid --set item: {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"invalid --set key: {item}")
        out.extend([f"--{k}", v])
    return out


def parse_set_map(items):
    out = []
    for k, v in items.items():
        key = str(k).strip()
        if not key:
            raise ValueError("invalid empty key in config.set")
        out.extend([f"--{key}", str(v)])
    return out


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


def ensure_inputs(args):
    if args.mode == "demo":
        return
    if args.mode in ("wxst_dir", "wsvt_dir"):
        if not args.img_dir or not args.ref_dir:
            raise ValueError("directory mode requires img_dir and ref_dir")
        if not args.out_dir:
            raise ValueError("missing required field: out_dir")
        return
    if args.mode in ("wxst", "wsvt"):
        if not args.img_h5 or not args.ref_h5:
            raise ValueError("hdf5 mode requires img_h5 and ref_h5")
        if not args.img_key or not args.ref_key:
            raise ValueError("hdf5 mode requires img_key and ref_key")
        if not args.out_dir:
            raise ValueError("missing required field: out_dir")
        return
    raise ValueError(f"unsupported mode: {args.mode}")


def resolve_exe(user_exe, base_dir=None):
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


def build_command(args):
    base_dir = getattr(args, "config_dir", None)
    exe = resolve_exe(args.exe, base_dir)
    if args.mode == "demo":
        return [str(exe), "demo"]
    if args.mode in ("wxst_dir", "wsvt_dir"):
        cmd = [
            str(exe),
            args.mode,
            str(resolve_path(args.img_dir, base_dir)),
            str(resolve_path(args.ref_dir, base_dir)),
            str(resolve_path(args.out_dir, base_dir)),
        ]
    else:
        cmd = [
            str(exe),
            args.mode,
            str(resolve_path(args.img_h5, base_dir)),
            args.img_key,
            str(resolve_path(args.ref_h5, base_dir)),
            args.ref_key,
            str(resolve_path(args.out_dir, base_dir)),
        ]
    cmd.extend(parse_set_items(args.set_items))
    return cmd


def apply_config(args):
    if not args.config:
        if not args.mode:
            raise ValueError("mode is required")
        if args.mode in ("wxst", "wsvt"):
            args.img_key = args.img_key or "img"
            args.ref_key = args.ref_key or "ref"
        return args

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    args.config_dir = config_path.parent
    if not args.mode:
        args.mode = str(cfg.get("mode", ""))
    if not args.mode:
        raise ValueError("mode is required in --config or CLI")
    if args.mode not in ("demo", "wxst", "wsvt", "wxst_dir", "wsvt_dir"):
        raise ValueError("mode must be one of: demo, wxst, wsvt, wxst_dir, wsvt_dir")

    for k in ("img_h5", "img_key", "ref_h5", "ref_key", "out_dir", "img_dir", "ref_dir"):
        if not getattr(args, k):
            setattr(args, k, str(cfg.get(k, "")))
    if args.mode in ("wxst", "wsvt"):
        args.img_key = args.img_key or "img"
        args.ref_key = args.ref_key or "ref"
    if not args.exe and cfg.get("exe"):
        args.exe = str(cfg.get("exe"))
    if not args.export and cfg.get("export", True):
        args.export = True

    cfg_set = cfg.get("set", {})
    if isinstance(cfg_set, dict):
        cfg_set_items = [f"{k}={v}" for k, v in cfg_set.items()]
    elif isinstance(cfg_set, list):
        cfg_set_items = [str(x) for x in cfg_set]
    else:
        raise ValueError("config.set must be object or array")
    args.set_items = cfg_set_items + list(args.set_items or [])
    return args


def make_parser():
    parser = argparse.ArgumentParser(prog="launch.py")
    parser.set_defaults(
        mode="",
        img_h5="",
        img_key="",
        ref_h5="",
        ref_key="",
        out_dir="",
        img_dir="",
        ref_dir="",
        set_items=None,
    )
    parser.add_argument("--exe", default="", help="path to wsvt_cli")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-export", dest="export", action="store_false",
                        help="skip PNG/TIFF export after C++ run")
    parser.add_argument("--config", default="", help="path to JSON config")
    parser.add_argument("--export-dir", default="",
                        help="export to a separate directory (default: same as out_dir)")
    sub = parser.add_subparsers(dest="mode", required=False)

    p_demo = sub.add_parser("demo")
    p_demo.set_defaults(mode="demo")

    for mode in ("wxst", "wsvt"):
        p = sub.add_parser(mode)
        p.add_argument("img_h5", nargs="?")
        p.add_argument("img_key", nargs="?")
        p.add_argument("ref_h5", nargs="?")
        p.add_argument("ref_key", nargs="?")
        p.add_argument("out_dir", nargs="?")
        p.add_argument("--set", dest="set_items", action="append")

    for mode in ("wxst_dir", "wsvt_dir"):
        p = sub.add_parser(mode)
        p.add_argument("img_dir", nargs="?")
        p.add_argument("ref_dir", nargs="?")
        p.add_argument("out_dir", nargs="?")
        p.add_argument("--set", dest="set_items", action="append")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    args = apply_config(args)
    ensure_inputs(args)
    cmd = build_command(args)
    print("command:", " ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return 0
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
    proc = subprocess.run(cmd, env=child_env)
    if proc.returncode == 0 and args.export and args.mode in MODE_RESULT_FILE:
        out_dir = str(resolve_path(args.out_dir, getattr(args, "config_dir", None)))
        export_results(out_dir, MODE_RESULT_FILE[args.mode], {"png", "tiff"})
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
