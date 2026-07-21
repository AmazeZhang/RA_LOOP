#!/usr/bin/env bash
set -euo pipefail

# Bounded 21-step (~4 hour) checkpointed RA-LOOP pilot.
# Safe default remains CPU-only; GPU use still requires --run <GPU_ID>.
export RA_LOOP_TRAIN_PROFILE=overnight_pilot
exec bash /home/imc/yzy/RA_LOOP/train/ra_loop_spatial_learning_probe.sh "$@"
