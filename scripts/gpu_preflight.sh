#!/usr/bin/env bash
# Maintainer GPU preflight: must pass before any CUDA workload starts.
#
# Runs the full ordered sequence (power-limit -> isolation guard ->
# tensor-parallel check) via anatomiae.provenance.gpu_power. Aborts (exit
# nonzero) on any violation - callers must not ignore a nonzero exit here.
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Running GPU preflight (physical index ${CUDA_VISIBLE_DEVICES})..." >&2
uv run python -m anatomiae.provenance.gpu_power
