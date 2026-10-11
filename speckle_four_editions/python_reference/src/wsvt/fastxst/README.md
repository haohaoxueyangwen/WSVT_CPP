# `wsvt.fastxst`

FastXST / WaveletSBI modules moved from `FastXST-main/WaveletSBI/` in R1.5.
Packaging move only. Solver math, windows, search, defaults, float32 C path,
and GPU kernels are unchanged.

Canonical imports:

```python
from wsvt.fastxst.WSVT import WSVT
from wsvt.fastxst.WXST import WXST
from wsvt.fastxst.backends import USE_C_SOLVER, choose_c_wavelet_backend
```

CLI:

```text
python -m wsvt.fastxst.single_run_WSVT --help
python -m wsvt.fastxst.single_run_WXST --help
python FastXST-main/WaveletSBI/WSVT.py --help
```

C solver still loads through ctypes (`c_solver.so` / `c_solver.dll`). Search
order: `WSVT_FASTXST_CSOLVER`, this package directory, then
`FastXST-main/WaveletSBI/`. Build with `python -m wsvt.fastxst.compile_c`
(same gcc/MSVC commands as the original `compile_c.py`). Optional setuptools
Extension is not required for `pip install -e .` so Windows editable installs
do not fail without a compiler.

Old `from WSVT import WSVT` paths still work when `FastXST-main/WaveletSBI` is
on `sys.path` (bench_full workers). `gui_func` is in the package because
`WSVT.py` imports `crop_gui`.

## Extracted, not rewritten

- C library load block: one copy in `backends.py`.
- wavelet_backend `auto|c|pywt` choice: `choose_c_wavelet_backend`.
- `WaveletTransformGPU` stays in `func.py` (already a single class).

## Deferred

- Inlined GPU cost-volume / subpixel blocks inside `WSVT`/`WXST` methods.
- `func_PostProcess.py`, `seidel2.py`, `diff_process.py`, tomo/multi-run
  launchers stay under `FastXST-main/WaveletSBI/`.
- Hardcoded example launchers `single_run_WSVT.py` / `single_run_WXST.py` in
  WaveletSBI still run the original os.system recipes; the package modules of
  the same name are the argparse CLIs.
