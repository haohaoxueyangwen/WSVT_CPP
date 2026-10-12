#!/usr/bin/env bash
# Select measured deployment policy; all compute/FFT execute in the native binary.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT}/build_desktop/bin/speckle_gpu" --platform desktop --workspace speed --input-policy copy "$@"
