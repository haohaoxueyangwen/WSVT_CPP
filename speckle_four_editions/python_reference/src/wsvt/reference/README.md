# `wsvt.reference`

Python-reference WSVT/WXST modules moved from `speckle_retriver/` in R1.4.
This is a packaging move. Solver math, windows, search, defaults, DPC scale,
wavelet defaults, and numeric dtypes are unchanged.

Canonical imports:

```python
from wsvt.reference.WSVT import WSVT
from wsvt.reference.WXST import WXST
from wsvt.reference.func import frankotchellappa, Wavelet_transform, find_disp
from wsvt.reference.euclidean_dist import dist_numba
```

Old `from speckle_retriver...` / `from WSVT import WSVT` paths still work through
shims in `speckle_retriver/`. `python speckle_retriver/WSVT.py` still runs the
same CLI. Thin wrappers also live in `tools/run_wsvt_reference.py` and
`tools/run_wxst_reference.py`.

## Shadowed `func.py` definitions (R1.4)

`speckle_retriver/func.py` defined four names twice. Python kept the **later**
body. The earlier copies were deleted:

| Name | Deleted (unused at runtime) | Kept |
|---|---|---|
| `convert_to_grayscale` | incomplete: returned only when `ndim == 2` | full RGB/RGBA conversion |
| `zernike_radial` | `(n, m, r)` | `(rho, n, m)` |
| `zernike` | `(n, m, r, theta)` | `(rho, phi, n, m)` |
| `noll_to_nm` | simple triangular index | later Noll conversion |

`zernike_analysis_phase` already called `zernike(r, theta, n, m)`, which matches
the kept signature. Call sites were **not** rewritten.

The two `zernike_radial` bodies also differ in `|m|` handling and factorial
terms. Deleting the first copy does not change runtime behavior because that
copy was already shadowed.

## Image I/O not merged

Three `load_images` / `load_image` / `image_preprocess` copies remain in
`WSVT.py`, `WXST.py`, and `func.py`. They were **not** folded into
`wsvt.reference.io` this round:

- `WSVT.load_images` matches `func.load_images` aside from comments.
- `WXST.load_image` wraps `np.array(...)` twice; `func.load_image` returns the
  first array. Equivalent for valid files.
- `WSVT`/`WXST.image_preprocess` **return** the corrected array.
  `func.image_preprocess` applies the same arithmetic and then falls off the
  function (**implicit `None`**). Merging would either add a return (behavior
  change for `func.image_preprocess`, including the GUI import) or drop the
  return (break WSVT/WXST CLI preprocess).

Leave the three copies until that return mismatch is an explicit, tested change.

## Deferred

- GUI stays at `speckle_retriver/speckle_tracker_gui.py` (not installable).
  Moving it to `research/gui/` is R1.4 remainder.
- `speckle_retriver/directional_darkfield.py` stays; it is not part of the
  WSVT/WXST algorithm move.
- FastXST (`wsvt.fastxst`) is R1.5; see `src/wsvt/fastxst/README.md`.
