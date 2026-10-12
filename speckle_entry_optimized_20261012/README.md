# Read-only CPU input and native GPU validation optimization

This independent edition preserves the 2026-10-11 owning CPU constructors,
native GPU implementation, old wrapper, and their measured benchmark records.
The original code is in the neighboring `speckle_four_editions_20261011` release.

## Changes

* CPU C++: a synchronous borrowed `std::span<const float>` constructor avoids
  creating raw sample/reference vector copies. Input storage must remain alive
  and unchanged until the call returns. WSVT still needs a private normalized
  stack; it is allocated uninitialized and fully filled directly from input.
  WXST reads its fine-level templates directly from the input view, then owns
  coarse-level working buffers. Original owning constructors remain available.
* The borrowed path supports the uncropped dense CPU profile, without initial
  displacement estimation, dark field or experimental adaptive/raw-fit stages.
  Unsupported modes are rejected rather than silently using empty owned data.
* Native GPU: float input finite checks use an alias-safe IEEE exponent reduction
  compiled separately as host C++, enabling SIMD without temporary arrays,
  extra host threads or GPU synchronization. NaN/Inf rejection is retained.
* SSD, subpixel fitting, wavelet arithmetic/order, spline, FFT and output
  ownership are unchanged. Required workspaces remain; no hidden reference cache.

## Build

CPU: configure `cpu_cpp` in Release with GCC12, C++20, OpenMP and FFTW; disable
HDF5/OpenCV for the array-entry benchmark. Build target `cpu_driver`.
GPU: configure `gpu_common` as in the original release (SM89 server or SM87
Orin). CUDA retains `--fmad=false`; only `src/finite.cpp` has native host SIMD
flags. Build `speckle_native` and `speckle_gpu`.

`workflow/matched_entry.py` compares retained original binaries and optimized
ones using the exact matched-array boundary documented in `autoresearch.md`.
`workflow/finite_checks.py` tests the finite check and reference/sample API;
`evidence` holds immutable reports, raw timings and comparison metrics.
Server measurements do not establish Orin performance; the Orin build must be
rebuilt on-device rather than using an x86 native binary.

Results and exact timing boundaries: [matched retest report](docs/BENCHMARK_20261012.md).
For the optimized array entry, build the `cpu_cpp` parent CMake target
`cpu_driver`; `tools/build_cpu_verified.py` builds the preserved owning CLI.
`workflow/implement.py` is the one-off creation receipt and must not be rerun
against an already modified source tree.
