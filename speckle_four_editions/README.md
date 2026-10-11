# Four practical speckle-tracking editions — 0.2.0

This independent release preserves the Python and CPU C++ implementations and adds native C++/CUDA/cuFFT GPU sessions. Both GPU editions share the same numerical kernels; their input-memory and build presets differ. Previous `edge_speckle`0.1.0 and `edge_speckle_optimized`0.1.1 sources remain unchanged. No manuscripts were modified.

| Edition | Directory | Recommended role |
|---|---|---|
| Python reference | `python_reference/` | Algorithm inspection, reference calculations |
| CPU C++ | `cpu_cpp/source/` | CPU deployment with OpenMP and threaded FFTW |
| Discrete GPU C++ | `gpu_discrete/`, `gpu_common/` | Ada workstation: pinned host input/output, device workspace |
| Jetson GPU C++ | `gpu_jetson/`, `gpu_common/` | Orin: mapped WSVT inputs, cached device workspace in shared DRAM |

`speed` and `low-memory` are WSVT workspace policies within each GPU edition. They are not additional implementations. WXST uses fused sample-template/wavelet/search processing; WSVT retains the original normalization → spatial pyramid → temporal wavelet order. Subpixel fitting is unchanged.

## Measured defaults

Use `gpu_discrete/run.sh` or `gpu_jetson/run.sh`. These shell launchers invoke the native executable directly. Jetson selects mapped input for WSVT and copy input for WXST; the measured WXST mapping difference was negligible. Override workspace/input policy explicitly when needed. The executable itself has conservative `speed/copy` defaults.

Build GPU (CUDA, cuFFT, CMake, C++ compiler required; no Python/PyTorch runtime):

```bash
# Ada server
cmake -S gpu_discrete -B build_desktop -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8
cmake --build build_desktop -j2
# Orin; replace desktop preset and toolkit above
cmake -S gpu_jetson -B build_jetson -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake --build build_jetson -j2
```

Both builds retain `--fmad=false`, SM89/SM87 respectively. They were built and tested with CUDA12.8/GCC12 and CUDA11.4/GCC9.4. `tools/build_native.py` is an optional build/receipt helper; Python is not needed to execute the resulting binaries.

```bash
# Shared syntax for WSVT and WXST. Always provide calibrated geometry.
bash gpu_jetson/run.sh --mode wsvt --sample sample.npy --reference reference.npy \
  --out NEW_RESULTS --pixel-m 1.625e-6 --distance-m .4845 --energy-ev 12400 \
  --warmup 1 --repeats 5
# Low-memory configuration (same input and output contract)
bash gpu_jetson/run.sh --mode wsvt --sample sample.npy --reference reference.npy \
  --out NEW_MEMORY_RESULTS --pixel-m 1.625e-6 --distance-m .4845 --energy-ev 12400 \
  --workspace low-memory --chunk 16
```

The current GPU preset supports contiguous little-endian NPY: WSVT100×H×W uint16 or float32, WXST H×W float32; H/W>28 with explicit implementation size checks. Matching is fixed to total half10, coarse5, fine4, two pyramid layers, one iteration, cut1, WXST7×7, WSVT no spatial template, dark-field off. These fixed settings make this a validated BB11 deployment preset, not a general replacement for every Python/CPU parameter combination. Float32 WSVT was checked on small data; full-size measurements use uint16.

`tools/convert_inputs.py` converts TIFF or NPZ once to NPY (NumPy; TIFF additionally needs tifffile). Native execution then performs file reading, scheduling, buffer management, search, spline interpolation, phase FFT and output writing in C++. No TIFF/Python decoding is hidden inside native solve time. For continuous acquisition, feed owned host buffers through `gpu_common/include/speckle_native.h` and reuse the session.

## CPU and Python

CPU C++ requires CMake≥3.21, C++20, OpenMP, single-precision FFTW and its threaded library. Optional HDF5/OpenCV provide file IO. Build the preserved source with your dependency paths:

```bash
cmake -S cpu_cpp/source -B build_cpu -DCMAKE_BUILD_TYPE=Release \
  -DWSVT_WITH_OPENCV=OFF -DWSVT_WITH_HDF5=ON
cmake --build build_cpu --target wsvt_cli -j2
build_cpu/wsvt_cli wsvt sample.h5 sample reference.h5 reference NEW_RESULTS \
  --crop 2048 --cal_half_window 10 --n_template 0 --n_s_extend 4 --wavelet_level_cut 1 \
  --pyramid_level 2 --n_iter 1 --n_cores 64 --phase_cores 4 \
  --energy 12400 --p_x 1.625e-6 --z .4845 --calc_darkfield false
```

Verify CMake reports FFTW enabled; its brute DFT fallback is unsuitable for large data. For WXST use command `wxst`, `--m_image 2048`, `--n_s 3` and main32/phase4 on the measured server. CPU threading is hardware dependent. Packaging builds on both hosts passed; no new full-size CPU speed claim is inferred from the build/demo. Orin packaging check disables external IO libraries and exercises the library/demo; applications can supply arrays through its public C++ constructors, or enable HDF5/OpenCV for the original CLI. `tools/build_cpu_verified.py` records the exact isolated dependency paths used in this experiment. The53 frozen CPU files and added packaging template are identified in the provenance receipts.

Python≥3.10 reference installation and explicit NPY runner:

```bash
python -m pip install -r python_reference/requirements.txt
python python_reference/run_reference.py --mode wxst --sample sample.npy \
  --reference reference.npy --out NEW_RESULTS --pixel-m 1.625e-6 \
  --distance-m .4845 --energy-ev 12400 --cores 4 --groups 4
```

The new runner retains original output/normalization rules. It returns six fields, without a fabricated matching-score field. Both Python modes passed128² invocation/finite-output checks on the server. This is packaging verification, not a new physical accuracy study.

## Results and evidence

See [benchmark report](benchmarks/RESULTS.md), full raw trial CSVs/JSONs, sampled-memory traces, smoke receipts and fixed-ROI phase/DPC figures in `benchmarks/results/{server,orin}`. Both hosts used one GPU without competing compute tasks, one warmup and five measurements, and rebuilt reference features each pair. Solver time, ready-array API time, file IO and initialization are separate. Native CLI API time uses reusable pinned outputs; the optional NumPy API allocates new outputs per call. Transport/ownership policy changes are explicitly reported rather than attributed solely to C++.

All six nonphase fields and available descriptors matched the preserved GPU implementation exactly. Full-image phase differences remained below1.53e-4rad maximum; the FFT/reduction difference and predeclared acceptance thresholds are recorded. Center1000² and the user's fixed600×500 ROI were also checked. This establishes numerical preservation against the old GPU implementation, not independent physical ground truth.

Source/binary/compiler hashes are retained in `docs/` and benchmark reports. Originals were checked against164 provenance hashes. Build directories, full input datasets, full NPY outputs and credentials are excluded from GitHub. Full outputs remain in each host's independent `native_four_editions_20261011` experiment directory.
