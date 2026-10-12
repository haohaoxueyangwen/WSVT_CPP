# Entry overhead optimization, 2026-10-12

Goal: remove CPU adapter raw-stack copies and speed native GPU finite validation
without changing matching, subpixel fitting, wavelet order, input precision,
thread settings or timing boundaries. Original release and libraries remain.

Protocol: BB11 canonical 2048-square inputs, WSVT 100 pairs, WXST first pair;
ready continuous pageable float32 host arrays; CPU main32/phase4; GPU0;
half10/coarse5/fine4, two spatial layers, cut1, one iteration, no dark field,
WXST7x7 and WSVT no spatial template. New session each call, rebuild reference,
new caller-owned seven host outputs. One warmup and five measured repeats.
Full array API includes init/validation/reference/output return; excludes
file IO/imports/destruction. CPU solver wall and GPU ref/sample/upload/download
remain separate columns, never substituted for complete API time.

Acceptance before formal runs: small-image seven-field bitwise comparison;
input SHA256 unchanged; finite validation must reject NaN/Inf even at tails and
after a previously validated input is modified. Full-size comparison must also
preserve all seven fields bitwise. No physical-truth claim follows this check.

Hypothesis H1: CPU normalized workspaces can be filled directly from read-only
borrowed input. Removing raw-vector copies and WXST duplicate ownership reduces
array API cost. The necessary private normalized WSVT stack stays allocated.
Hypothesis H2: a single-thread alias-safe integer exponent reduction can SIMD-
vectorize finite checks without temporary arrays, extra host threads or CUDA sync.

Other optimizations: eliminate zero initialization of the WSVT normalized buffer
which is fully overwritten. Keep FFT plans and output ownership unchanged until
the above measured changes pass. Do not bundle speculative algorithm changes.

Run scripts and artifacts are recorded in workflow/ and evidence/. Original
driver is retained as cpu_cpp/cpu_driver_original.cpp; original full release is
speckle_four_editions_20261011. Failure records are not deleted or relaxed.
