# WSVT_CPP Dependencies

## Required

| Dependency | Version | Notes |
|---|---|---|
| C++20 compiler | g++ ≥ 12, clang ≥ 16 | `-std=c++20` |
| OpenMP | ≥ 4.5 | Multi-threading (pyramid/wavelet/displace) |
| CMake | ≥ 3.21 | Build system |

## Recommended (features disabled if missing)

| Dependency | CMake flag | Notes |
|---|---|---|
| HDF5 | `WSVT_WITH_HDF5=ON` | HDF5 result I/O (`write_h5`/`read_h5`) |
| FFTW3 (single precision) | `WSVT_WITH_FFTW=ON` | Fast Frankot-Chellappa phase recovery |
| OpenCV 4 | `WSVT_WITH_OPENCV=ON` | `use_estimate=true` slope tracking init |

If a recommended dependency is not found, the build succeeds but the
corresponding feature throws at runtime.

## Conda environments

| Environment | Purpose | Key packages |
|---|---|---|
| `wsvt_cpp` | C++ build + runtime | cmake, gxx, hdf5, fftw, opencv, libtiff |
| `speckle` | Python benchmark harness | python 3.11, numpy, scipy, numba, pywavelets, h5py, Pillow |

## Build from clean checkout

```bash
cd WSVT_CPP

# Configure (conda env wsvt_cpp activated)
cmake --preset release-lto \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DHDF5_C_COMPILER_EXECUTABLE=$CONDA_PREFIX/bin/h5cc \
    -DOpenCV_DIR=$CONDA_PREFIX/lib/cmake/opencv4

# Build
cmake --build build/release-lto --target wsvt_cli --parallel 8

# Smoke test
./build/release-lto/wsvt_cli demo
```

## Runtime library paths

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Required for HDF5, FFTW, OpenCV, and OpenMP runtime libraries to be found.

## Troubleshooting

### `libhdf5.so` not found
Ensure `hdf5` is installed in the `wsvt_cpp` conda environment and
`LD_LIBRARY_PATH` includes `$CONDA_PREFIX/lib`.

### `libfftw3f.so` not found
Install `fftw` in the `wsvt_cpp` environment.  The C++ solver falls back
to an O(n⁴) brute-force DFT if FFTW is unavailable (very slow).

### `wsvt_cli: command not found`
The binary is at `build/release-lto/wsvt_cli` (or `build/release/`).
Use an absolute path or add to `PATH`.

### OpenMP nested parallelism warnings
The solver uses nested OpenMP sections for ref/img pyramid computation.
Set `OMP_MAX_ACTIVE_LEVELS=2` (or higher) if you see warnings.

### Wrong number of threads
The solver respects `OMP_NUM_THREADS` unless overridden by `--n_cores`.
For benchmarking, set `OMP_PROC_BIND=close` and `OMP_PLACES=cores`.
