#!/usr/bin/env bash
set -euo pipefail

# CPU-only Hydra/factory preflight for the first bounded recovery-only pilot.
# This script intentionally has no GPU execution mode.

if [[ $# -ne 0 ]]; then
  echo "This preflight accepts no arguments and never runs GPU training." >&2
  exit 2
fi

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
RIPT_ROOT=/home/imc/code/ript-vla
PYTHON_BIN=/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python
OFFICIAL_LIBERO_ROOT=/home/imc/code/LIBERO-official
MODEL_PATH=/home/imc/yzy/RA_LOOP/runtime/openvla-oft-spatial-smoke
HEADER_PATH=/home/imc/models/ra-loop/ript-vla/openvla_oft/scale_header/LIBERO_SPATIAL_scale_header.pth
DATA_PATH=/home/imc/data/ra-loop/libero-datasets
TASK_NAME=pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate

export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/.libero_official"
export PYTHONPATH="${PROJECT_ROOT}:${OFFICIAL_LIBERO_ROOT}:${RIPT_ROOT}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR=/tmp/ra_loop_numba_cache
export MPLCONFIGDIR=/tmp/ra_loop_mpl_cache
export CUDA_VISIBLE_DEVICES=''

OVERRIDES=(
  exp_name=RA-LOOP_spatial_robot_init_preflight
  variant_name=one_task_one_step_k4_h10_robot_init_recovery
  make_unique_experiment_dir=true
  paths.data_prefix="${DATA_PATH}"
  paths.output_prefix="${PROJECT_ROOT}/outputs/ra_loop_spatial_preflight"
  task.suite_name=libero_spatial
  task.dataset.suite_name=.
  task.task_names_to_use="[${TASK_NAME}]"
  task.demos_per_env=1
  task.rollouts_per_env=1
  train_dataloader.batch_size=1
  train_dataloader.shuffle=false
  train_dataloader.num_workers=0
  train_dataloader.persistent_workers=false
  train_dataloader.pin_memory=false
  train_dataloader.multiprocessing_context=null
  training.gradient_accumulation_steps=1
  training.n_steps=1
  training.rollout_steps=1
  training.save_interval=9999
  training.log_interval=1
  training.use_tqdm=true
  rollout.enabled=false
  algo.env_runner._target_=ra_loop.ript_compat.InProcessOpenVLAOFTLiberoRunner
  algo.rollout_generator_factory._target_=ra_loop.ript_recovery.RobotInitRecoveryRolloutGenerator
  +algo.rollout_generator_factory.robot_init_strength=0.001
  +algo.rollout_generator_factory.perturb_seed=20260720
  algo.rl_optimizer_factory._target_=ra_loop.ript_recovery.RobotInitRecoveryOptimizer
  reward_function._target_=ra_loop.ript_recovery.RobotInitRecoveryReward
  +reward_function.lambda_recovery=0.5
  algo.rloo_batch_size=4
  algo.rollouts_per_env=4
  algo.num_parallel_envs=1
  algo.max_episode_length=10
  algo.enable_dynamic_sampling=false
  algo.use_val_init=false
  algo.mix_val_init_in_rloo=false
  algo.gradient_accumulation_steps=1
  algo.num_ppo_epochs=1
  algo.max_step_batch_size=1
  algo.scale_factor=2.0
  algo.fix_scale_head=true
  algo.checkpoint_path="${MODEL_PATH}"
  algo.header_checkpoint="${HEADER_PATH}"
  algo.lora_adaptor_ckpt=null
  logging.mode=disabled
  logging.resume=false
  logging.save_code=false
)

echo "CPU-only resolved Hydra configuration; no dataset/env/model/CUDA creation"
cd "${RIPT_ROOT}"
"${PYTHON_BIN}" train_ript_openvla_oft.py \
  --config-name=train_rl_openvla_oft_all_task_spatial.yaml \
  --cfg job --resolve "${OVERRIDES[@]}"

echo "CPU-only factory instantiation with create_env=false"
cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" scripts/preflight_ra_hydra_factories.py "${OVERRIDES[@]}"
