# WSVT / WXST: C++ CPU and Jetson GPU

C++20/OpenMP WSVT/WXST tracker and native author dark-field fit. Same paper
project as the root Python repo. Branch: `codex/cpp-wsvt`.

```bash
cmake -S . -B build-ddf-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-ddf-cpp --target wsvt_paper -j 4
```

- `wsvt_cli` — WSVT/WXST solver
- `wsvt_ddf_author` — shared library for fixed-u author DDF (`wsvt_ddf_author_fit`)

Do not merge SET4/raw-rerank into this line. Formal WSVT timing freeze is tag
`paper-wsvt-100f-20260829` (`5181ae4`).


## Source organization and first steps

See [source snapshot and teaching entry](docs/teaching_and_source_sync.md) for the 2026-10-11 source provenance, omitted reference fixtures and current verification scope. For a first runnable Python example, use the separate [teaching toolkit](https://github.com/haohaoxueyangwen/wavefront-starter).

## Jetson GPU engineering package

The independent [edge_speckle package](edge_speckle/README.md) contains the reviewed Orin WXST and WSVT implementations, native CUDA dependencies, separate method entry points, fixed configuration contracts, build receipts and reconstruction regression checks.

| Implementation | Purpose | Entry |
|---|---|---|
| FASTWSVT-cpp / FASTWXST-cpp | Existing C++20/OpenMP CPU source snapshot | Root CMake project |
| edge WXST | Sample feature/search fusion, GPU spline and phase integration | `python3 -m speckle_edge wxst` inside `edge_speckle` |
| edge WSVT | Original operation order, native uint16 input, GPU spline and phase integration | `python3 -m speckle_edge wsvt` inside `edge_speckle` |
| GPU v1/v2/v3 | Frozen WXST prototypes | `edge_speckle/baselines/legacy_wxst` |

The edge package implements search half-window 10, coarse radius 5, fine radius 4, two spatial levels, one iteration and wavelet cut 1. WXST uses a 7x7 spatial template; WSVT uses 100 scan frames without a spatial template. Geometry must be supplied explicitly. WSVT descriptor/pyramid reordering remains an opt-in experiment.

Build on Orin with `python3 tools/build.py` from `edge_speckle`. Native CUDA/CPU-spline rebuilds, four small-input packaging regressions and both CLI entries passed on the recorded Orin environment; [validation records](edge_speckle/docs/VALIDATION.md) retain the scope and known localized cross-backend differences. This publication does not rerun full-size benchmarks or native tests for the separate root C++ snapshot. Data and generated native binaries are not included.

See [version responsibilities](edge_speckle/docs/VERSIONS.md) and [GitHub publication provenance](docs/jetson_edge_publication.md). CPU benchmark source inside the edge package is an archival, separately configured comparison snapshot; it does not replace the root C++ source.
