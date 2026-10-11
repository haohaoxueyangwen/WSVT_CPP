# Orin WSVT memory/output-copy benchmark, 2026-10-11

## Measured result

| Profile | Pair solver (s) | Seven-field API (s) | Persistent buffers (GiB) | CUDA allocated peak (GiB) | Peak reserved (GiB) | Lifetime RSS peak (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| original | 1.388753 +/- 0.001194 | 1.428376 +/- 0.002450 | 6.4775 | 6.5766 | 6.6035 | 2.9290 |
| speed | 1.389637 +/- 0.001327 | 1.422335 +/- 0.002958 | 6.4775 | 6.5766 | 6.6035 | 3.0465 |
| low_memory | 1.602849 +/- 0.001940 | 1.633655 +/- 0.003282 | 4.2574 | 4.3562 | 4.3848 | 2.9851 |

One warmup and five measured pairs per independent process, means +/- sample standard deviations. BB11 lens, 100 x 2048 x 2048 uint16. Every timed pair rebuilds the reference. Original0.1.0 and optimized0.1.1 use the same subpixel, SSD, spline, phase and temporal/spatial operator order.

Speed configuration (materialized workspace, independent host tensor outputs without an extra NumPy copy) preserves solver latency, with a small 6.041 ms reduction in API latency. It is not a substantive solver acceleration.

Low-memory configuration (streamed workspace, spatial channel chunk16) reduces persistent device buffers by 34.27% and CUDA allocated peak by 33.76%, while pair solver latency increases 15.42%. On this Orin, use speed mode when latency is the priority; use low-memory mode when DRAM headroom matters. This benchmark does not change either preset to conceal the tradeoff.

## Stage evidence

| Profile | Reference preparation (s) | Sample feature GPU event (s) | Coarse match GPU event (s) | Fine match GPU event (s) |
|---|---:|---:|---:|---:|
| original | 0.383762 | 0.288459 | 0.121035 | 0.456863 |
| speed | 0.383995 | 0.288307 | 0.121229 | 0.457336 |
| low_memory | 0.489546 | 0.393719 | 0.121216 | 0.457715 |

The extra low-memory time appears primarily in reference/sample feature generation. Coarse/fine matching timings remain close. Kernel occupancy, instruction throughput and bandwidth were not profiled, so these measurements do not establish a particular hardware cause.

## Memory and timing scopes

CUDA allocator peaks were reset after warmup and measured across five complete pairs. Persistent buffers are the session's explicit device tensors; peaks also include transient FFT/pointwise workspace. Reserved peak is the PyTorch allocator reservation, not an additional allocation to sum with allocated peak.

Linux ru_maxrss is the process lifetime RSS through the end of formal solves: imports, input decoding, session initialization, warmup and measured runs, excluding subsequent descriptor hashing and file saving. Host RSS does not show a reduction: values span about 2.93-3.05 GiB. Device allocations and host RSS share physical DRAM on Jetson and must not be summed as independent hardware memories.

25 ms sampled process/system RAM and 200 ms tegrastats logs are included. Sampled peaks are lower bounds, and system MemTotal-MemAvailable includes other processes and page-cache effects. The system-level snapshot is not a precise per-process memory attribution.

Solver time: reference preparation + sample solve, including uploads and final CUDA synchronization, excluding validation/download, session construction and disk I/O. API time: already-decoded contiguous host arrays to seven independent writable NumPy arrays, including input validation, reference rebuild and download; excluding construction and disk I/O. Read/decode and output-save times are recorded separately, not mixed into API time.

## Numerical and visual comparison

All seven full-field outputs are exactly equal to the original GPU outputs in both optimized presets. Fine/coarse reference and sample descriptors have identical SHA256 values. Odd/even crops, uint16/float32, constant-zero-standard-deviation and ownership-after-reuse/close checks also passed. These checks establish preservation of the original GPU implementation, not independent physical accuracy.

Figures show all three phases over the full valid2028 x2028 field, central1000 x1000 phase/DPCx images with identical scales, differences, and fixed detector-y1024 profiles. 1-99% clipping is only for display; full comparisons use all pixels.

## Device/build/provenance

Jetson AGX Orin, shared ~61 GiB CPU/GPU DRAM, CC8.7/16 SM, CUDA11.4, GCC9.4, PyTorch2.0.0+nv23.05. MAXN mode0; existing power/clock policy preserved. No jetson_clocks/nvpmodel mutation. Actual native flags include -O3 --std=c++17 --fmad=false -arch=sm_87. CMake3.16 ignores the generic CMAKE_CUDA_ARCHITECTURES variable; the explicit -arch=sm_87 target flag is present and was checked.

Sample/reference canonical uint16 hashes match the server TIFF inputs. Source archive SHA256:8577c1006d9b9579e66c8c77c2bd476c8da21cdbe44a868377cfa58f9c93b1c9. Native binaries were rebuilt for ARM/SM87 in this independent directory; baseline binaries were copied without rebuilding. Original source/native hashes remain unchanged after the runs.

Original: /home/xray/Documents/WSVT_GPU_orin_20261010/edge_speckle
New deployment/results: /home/xray/Documents/WSVT_GPU_orin_20261010/memory_opt_20261011
Speed/low-memory preset examples: package/configs/wsvt_speed.json and wsvt_memory.json.

Full .npy arrays remain on Orin. Lightweight evidence bundle excludes input arrays, reconstruction arrays, source archives, build objects and credentials.

## Reproduction

Review autoresearch.md and source/build receipts. Preserve existing results. In a fresh independent result directory, deploy the same package, then run python3 -u run_orin.py batch and python3 analyze_orin.py. The runner intentionally refuses to overwrite original/speed/low_memory result folders. No test was run concurrently with a competing user GPU job in the inspected state.
