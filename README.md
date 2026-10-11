# WSVT C++

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
