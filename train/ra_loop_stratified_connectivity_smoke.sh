#!/usr/bin/env bash
set -euo pipefail

# One-task, one-step K8/h220 smoke for the mode-stratified optimizer path.
# Safe default remains CPU-only; GPU use requires --run <GPU_ID>.
export RA_LOOP_TRAIN_PROFILE=stratified_smoke
exec bash /home/imc/yzy/RA_LOOP/train/ra_loop_spatial_learning_probe.sh "$@"
