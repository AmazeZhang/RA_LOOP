#!/usr/bin/env bash
set -euo pipefail

# K=8 / scale=5 probe for obtaining mixed binary rewards and non-zero LOO signal.
# Safe default is CPU-only Hydra composition; GPU use requires --run <GPU_ID>.
export RA_LOOP_PROFILE=mixed_reward_probe
exec bash /home/imc/yzy/RA_LOOP/train/vanilla_loop_spatial_smoke.sh "$@"
