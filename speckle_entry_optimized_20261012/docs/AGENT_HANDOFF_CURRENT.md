# Entry optimization handoff, 2026-10-12

Completed CPU read-only borrowed array entry and native GPU SIMD finite checks;
40 formal matched-array trials plus smoke/invalid-input tests passed. See
BENCHMARK_20261012.md and ../evidence/audit.json. All prior originals remain.

Remote independent directory: bench_results/entry_optimized_20261012. Controller
completed and exited; no recurring automation was created or resumed. Source,
binaries and original load records retained. Full seven-field outputs remain
on server; lightweight metrics/crops/figures are available locally.

Keep cold array API, internal solver, hot sessions and file IO distinct. CPU
32/4 is fixed. The borrowed profile does not silently support dark field,
initial estimation, cropping or adaptive raw-input experiments. Input lifetime
and immutability are caller responsibilities for the synchronous API.

Orin validation is pending for these new entry optimizations. Do not claim
server SIMD timings as Jetson results. No subpixel/SSD/FFT changes were made.
