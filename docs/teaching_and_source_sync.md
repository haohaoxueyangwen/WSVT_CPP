# Source snapshot and teaching entry

The GitHub default `main` branch had an empty tree. This update restores the committed WSVT/WXST C++ source at `e3b470c17d9e69ec28b0fe3073940a7c6ff30552`, without modifying the workstation's existing development checkout or publishing its uncommitted scalar dark-field work.

The source tree comes from the canonical `codex/cpp-wsvt` line. `SOURCE_SNAPSHOT.json` records the exact source commit and byte-level hashes. The unrelated SET4/raw-rerank branch is not merged. Existing research benchmark numbers and results are not republished by this update.

Native building and C++ numerical acceptance were not run during this source organization task. Read [DEPENDENCIES.md](../DEPENDENCIES.md) and verify the required C++20/OpenMP/HDF5/FFTW/OpenCV toolchain before running. The existing server-oriented build recipes are not a Windows validation report.

Two historical NPZ reference fixtures are omitted from this source-only snapshot. Their manifest JSON files and generation/comparison scripts are retained, with the omitted paths recorded in `SOURCE_SNAPSHOT.json`. Regenerate or obtain those original fixtures before running Python stage comparisons; this update does not claim those comparisons passed.

The separate [wavefront-starter](https://github.com/haohaoxueyangwen/wavefront-starter) teaching repository provides small Python CPU examples for WXST, WSVT, grating, Hartmann centroid and Zernike postprocessing. Its numerical controls and input contract do not replace the native solver's research validation.
