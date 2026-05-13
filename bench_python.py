"""Benchmark WXSVT_v2 Python version — same data/params as C++."""
import sys
sys.path.insert(0, '/home/yrc/vscodespace/WXSVT_v2')
import os
os.chdir('/home/yrc/vscodespace/WXSVT_v2')

# Mock gui_func before importing WSVT
import types
sys.modules['gui_func'] = types.ModuleType('gui_func')
sys.modules['gui_func'].crop_gui = lambda *a, **kw: (None, None)

import time
import numpy as np
from PIL import Image
import glob

# Load same test data
img_files = sorted(glob.glob('/home/yrc/vscodespace/wsvtdata/sample/*.tif'))
ref_files = sorted(glob.glob('/home/yrc/vscodespace/wsvtdata/ref/*.tif'))

print(f"Loading {len(img_files)} sample + {len(ref_files)} ref images...")
t0 = time.time()
img_stack = np.array([np.array(Image.open(f), dtype=np.float32) for f in img_files])
ref_stack = np.array([np.array(Image.open(f), dtype=np.float32) for f in ref_files])
t1 = time.time()
print(f"Load time: {t1-t0:.2f}s, shape: img={img_stack.shape}, ref={ref_stack.shape}")

import io
import contextlib

from WSVT import WSVT

t2 = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    solver = WSVT(
        img_stack, ref_stack,
        crop=2048,
        cal_half_window=30,
        N_template=0,
        N_s_extend=4,
        n_cores=128,
        n_group=2,
        energy=14000,
        p_x=6.5e-7,
        z=0.425,
        wavelet_level_cut=1,
        pyramid_level=1,
        n_iter=1,
        use_estimate=False,
        use_wavelet=True,
        use_GPU=0,
    )
    out = solver.run()
t3 = time.time()
print(f"\nPython WSVT total time: {t3-t2:.3f} s")
print(f"  (including load: {t3-t0:.3f} s)")
