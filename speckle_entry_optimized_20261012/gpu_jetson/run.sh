#!/usr/bin/env bash
# Mapped input improves WSVT100; WXST showed no material mapping benefit.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY=copy
PREV=""
for ARG in "$@"; do
  if [[ "$PREV" == "--mode" && "$ARG" == "wsvt" ]]; then POLICY=mapped; fi
  PREV="$ARG"
done
exec "${ROOT}/build_jetson/bin/speckle_gpu" --platform jetson --workspace speed --input-policy "$POLICY" "$@"
