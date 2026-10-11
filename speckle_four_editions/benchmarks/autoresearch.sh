#!/usr/bin/env bash
# Remote result directory: preserve prior run folder, do not run with competing jobs.
set -euo pipefail
python3 validate_native.py > smoke.log 2>&1
python3 run_native_bench.py batch --mode wsvt
python3 run_native_bench.py batch --mode wxst
python3 analyze_outputs.py
