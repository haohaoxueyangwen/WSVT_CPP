# Native four-edition reconstruction experiment

User authorized native C++ scheduling and cuFFT conversion, original preservation, server/Orin speed-memory-image comparison and GitHub update. CUDA subpixel/search/features sources remain byte-identical to frozen0.1.1. WSVT original operator order; no reordered experiment.

BB11 full2048x2048: WXST first float32 pair/template7x7; WSVT100 uint16 frames/no spatial template; half10/coarse5/fine4/L2/I1/cut1/noDF/GPU spline. Reference rebuilt each pair; warmup1/five formal. CUDA12.8/GCC12/SM89 server; CUDA11.4/GCC9.4/SM87 Orin; --fmad=false. No device power/clock changes, no competing user jobs.

Acceptance: displacement/DPC/transmission/negativeSSD and descriptors exact to preserved GPU implementation. Phase retains original sign, positive Nyquist, floor1e-12, inverse normalization and mean removal. Allow finite floating FFT/reduction differences only: max absolute phase <=2e-3 rad plus 1e-5*max(abs(original)); RMSE <=2e-4 rad plus1e-6*RMS(original). Declare and retain failures; never relax gates silently. Constant-field and odd/even dimension checks; caller-owned output retention after reuse/close.

Compare ready pageable host input on both original/native; pinned/mapped transport additionally evaluated with same host-array policy for the reference implementation. Report changed transport separately. Native CLI independent process demonstrates no Python/PyTorch runtime dependency. Full-input read/decode, constructor, reference/cache, sample compute, seven-field return, file save measured separately.

Separate processes for profiles. Memory: native explicit buffers/cuFFT workspace, process lifetime and sampled RSS, desktop NVML compute-process usage, Orin system RAM/tegrastats. Compare compatible memory metrics; Torch allocation peak vs native explicit bytes is not a total-physical-memory equivalence. Shared DRAM statistics overlap and must not be summed. Preserve all trials, sources/binary hashes, load and numerical/image comparisons. Original sources/manuscripts/outputs unchanged.
