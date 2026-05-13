#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$HOME/vscodespace/anaconda3/envs/wsvt_cpp/lib:$HOME/vscodespace/anaconda3/lib:${LD_LIBRARY_PATH:-}"
conda run -n wsvt_cpp python launch.py --exe build-opencv/wsvt_cli --config config.json
