#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p evidence
# Build with the recorded compiler/FFTW/CUDA presets before invoking this.
CUDA_VISIBLE_DEVICES=0 python workflow/finite_checks.py
python -u workflow/matched_entry.py controller
# The controller preserves failures and rejects reuse of incomplete runs.
