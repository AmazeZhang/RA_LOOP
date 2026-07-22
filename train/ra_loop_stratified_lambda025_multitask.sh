#!/usr/bin/env bash
set -euo pipefail

export RA_LOOP_TRAIN_PROFILE=stratified_lambda025_multitask
exec bash "$(dirname "$0")/ra_loop_spatial_learning_probe.sh" "$@"
