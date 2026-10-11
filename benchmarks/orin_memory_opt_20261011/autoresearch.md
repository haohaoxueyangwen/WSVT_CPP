# Orin WSVT memory and output-copy comparison

User-authorized deployment and benchmark of original0.1.0, optimized speed, optimized low-memory configurations. Preserve original source/binaries/input/results. No algorithm/subpixel/window changes.

Use BB11 uint16 100x2048x2048, half10/coarse5/fine4, two spatial levels, I1/cut1, no spatial template/darkfield; original temporal/spatial operator order. GPU spline, host2, SM87 CUDA11.4 GCC9.4, --fmad=false. Reference rebuilt for every pair; one warmup/five formal per independent process; report solver and seven-field API separately, exclude disk and construction.

Verify source archive/baseline/native/data provenance; compare canonical uint16 array hashes with current server TIFFs. Small real odd/even crop, constant-zero-std and output ownership checks before full runs. Require exact old-GPU descriptor/output preservation; failure retained, no threshold relaxation.

Memory: separate processes so other variant buffers never coexist. Record persistent GPU buffers, CUDA allocator peak allocated/reserved, process lifetime peak RSS (ru_maxrss), sampled system physical RAM/available and tegrastats. These overlap on unified-DRAM Jetson and must not be summed; system measurements include background activity. Keep power/clock policy unchanged and record load/frequency/thermal evidence.

Reject competing CUDA/training or CPU-heavy user tasks; do not terminate/change them. Keep all raw trials and failed runs. Full arrays on Orin; download lightweight CSV/JSON and reconstructed/comparison images. Original project and manuscript files are not edited; incremental handoff in independent result directory.
