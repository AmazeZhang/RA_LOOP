#!/usr/bin/env bash
set -euo pipefail

export RA_LOOP_TRAIN_PROFILE=counterfactual_soft_gate_seed10000
exec bash /home/imc/yzy/RA_LOOP/train/ra_loop_spatial_learning_probe.sh "$@"
