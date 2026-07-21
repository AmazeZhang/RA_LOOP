#!/usr/bin/env bash
set -euo pipefail

# Controlled 35-step (~6 hour) repeat of the four-task run with only the RLOO
# objective changed to mode-stratified. GPU use requires --run <GPU_ID>.
export RA_LOOP_TRAIN_PROFILE=stratified_multitask
exec bash /home/imc/yzy/RA_LOOP/train/ra_loop_spatial_learning_probe.sh "$@"
