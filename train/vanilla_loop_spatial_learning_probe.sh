#!/usr/bin/env bash
set -euo pipefail

# Safe default is CPU-only Hydra composition. GPU use remains explicit:
#   bash train/vanilla_loop_spatial_learning_probe.sh --run <GPU_ID>
export RA_LOOP_PROFILE=learning_probe
exec bash /home/imc/yzy/RA_LOOP/train/vanilla_loop_spatial_smoke.sh "$@"
