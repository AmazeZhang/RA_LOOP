#!/usr/bin/env bash
set -euo pipefail

# Bounded 35-step (~6-8 hour) four-task RA-LOOP run, warm-started from the
# independently evaluated step-5 pilot checkpoint. Safe default is CPU-only.
export RA_LOOP_TRAIN_PROFILE=afternoon_multitask
exec bash /home/imc/yzy/RA_LOOP/train/ra_loop_spatial_learning_probe.sh "$@"
