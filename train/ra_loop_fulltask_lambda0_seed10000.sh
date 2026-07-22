#!/usr/bin/env bash
set -euo pipefail

export RA_LOOP_TRAIN_PROFILE=fulltask_lambda0_seed10000
exec bash "$(dirname "$0")/ra_loop_spatial_learning_probe.sh" "$@"
