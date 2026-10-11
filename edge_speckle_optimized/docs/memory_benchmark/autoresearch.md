# GPU memory optimization, 2026-10-11

Authorized: preserve baseline, no subpixel changes; optimize output copies and WSVT intermediate stacks, benchmark and show BB11 reconstructions.

Hypotheses: independent host tensor -> NumPy view removes a redundant host copy; tiled normalization/spatial-x and fine-DWT normalization loads remove full normalized stack; chunked spatial temporary reduces resident bytes.

Controls: identical normalization rounding and operator order, --fmad=false, SM89 CUDA12.8/GCC12, half10/coarse5/fine4, two levels, I1/cut1; WXST7x7, WSVT100 frames uint16. Reference rebuilt each pair, single GPU, host2. No subpixel/spline/SSD/phase edits. One warm-up, five formal measurements for baseline/copy-only/selected-streamed.

Acceptance: small real BB11 odd/even crops plus uint16/float32/constant engineering inputs; exact fine/coarse descriptors and seven outputs against preserved GPU baseline. Host output remains writable, independent after next solve and session close. Exact agreement is an engineering check, not physical truth. CPU baseline already measured; this experiment compares old/new GPU.

Bounded search: coarse temporary chunks16/32/100, three exploratory repeats. Retain all trials; choose tradeoff explicitly. Stop on unexpected numerical changes or competing compute load. Record source/binary/input hashes, timing scope, mean/sampleSD, resident buffers, diagnostics and negative results. No peak-memory claim from resident bytes alone.

Original edge_speckle and remote old benchmark remain immutable. Candidate edge_speckle_optimized0.1.1 lives in a separate branch and remote directory. Source release and lightweight receipts only are published.
