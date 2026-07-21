#!/usr/bin/env bash
set -euo pipefail

HF_BIN=/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/hf
SCALE_ROOT=/home/imc/models/ra-loop/ript-vla
DATA_ROOT=/home/imc/data/ra-loop/libero-datasets

echo "[$(date '+%F %T %Z')] Downloading RIPT Spatial scale header"
"${HF_BIN}" download tanshh97/RIPT_VLA \
  openvla_oft/scale_header/LIBERO_SPATIAL_scale_header.pth \
  --local-dir "${SCALE_ROOT}" \
  --max-workers 1

echo "[$(date '+%F %T %Z')] Downloading official LIBERO Spatial HDF5 files"
"${HF_BIN}" download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --include 'libero_spatial/*' \
  --local-dir "${DATA_ROOT}" \
  --max-workers 4

echo "[$(date '+%F %T %Z')] Spatial training assets downloaded"
