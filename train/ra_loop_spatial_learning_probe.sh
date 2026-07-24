#!/usr/bin/env bash
set -euo pipefail

# One-task, one-step RA-LOOP learning-signal probe, aligned to the lightest
# LIBERO-Plus Robot-init generation radius. Safe default is CPU-only preflight;
# GPU execution requires the explicit form: --run <GPU_ID>.

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
RIPT_ROOT=/home/imc/code/ript-vla
OFFICIAL_LIBERO_ROOT=/home/imc/code/LIBERO-official
PYTHON_BIN=/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python
TORCHRUN_BIN=/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/torchrun
MODEL_PATH=/home/imc/yzy/RA_LOOP/runtime/openvla-oft-spatial-smoke
HEADER_PATH=/home/imc/models/ra-loop/ript-vla/openvla_oft/scale_header/LIBERO_SPATIAL_scale_header.pth
DATA_PATH=/home/imc/data/ra-loop/libero-datasets
TASK_NAME=pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
PILOT_STEP5_CKPT="${PROJECT_ROOT}/outputs/ra_loop_spatial_overnight_pilot/libero_spatial/LIBERO_SPATIAL/openvla/RA-LOOP_spatial_robot_init_overnight_pilot/one_task_21step_k8_h220_fixed_l2_0p1_recovery_lr1e5/run_000/openvla_lora_step_000005"
RUN_PROFILE="${RA_LOOP_TRAIN_PROFILE:-learning_probe}"
TASK_NAMES_OVERRIDE="[${TASK_NAME}]"
LORA_ADAPTOR_CKPT=null
TRAIN_SHUFFLE=false
RECOVERY_LAMBDA=0.5
ADVANTAGE_MODE=mode_stratified
NOMINAL_ALLOWED_DROP=0.02
NOMINAL_EMA_DECAY=0.9
NOMINAL_DUAL_LR=0.1
NOMINAL_INITIAL_MULTIPLIER=1.0
NOMINAL_MAX_MULTIPLIER=10.0
NOMINAL_CALIBRATION_BATCHES=3
DEMOS_PER_ENV=4
TASK_ROLLOUTS_PER_ENV=4
TRAIN_SEED=10000
PERTURB_SEED=20260720

case "${RUN_PROFILE}" in
  learning_probe)
    EXP_NAME=RA-LOOP_spatial_robot_init_learning_probe
    VARIANT_NAME=one_task_one_step_k8_h220_fixed_l2_0p1_recovery
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/ra_loop_spatial_learning_probe"
    N_STEPS=1
    SAVE_INTERVAL=9999
    MODEL_LR=5e-5
    HEADER_LR=5e-5
    REQUIRE_FRESH_OUTPUT=false
    ;;
  overnight_pilot)
    EXP_NAME=RA-LOOP_spatial_robot_init_overnight_pilot
    VARIANT_NAME=one_task_21step_k8_h220_fixed_l2_0p1_recovery_lr1e5
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/ra_loop_spatial_overnight_pilot"
    N_STEPS=21
    SAVE_INTERVAL=5
    MODEL_LR=1e-5
    HEADER_LR=1e-5
    REQUIRE_FRESH_OUTPUT=true
    ;;
  stratified_smoke)
    EXP_NAME=RA-LOOP_spatial_stratified_connectivity_smoke
    VARIANT_NAME=one_task_one_step_k8_h220_fixed_l2_0p1_stratified_warmstart
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/ra_loop_stratified_connectivity_smoke"
    N_STEPS=1
    SAVE_INTERVAL=9999
    MODEL_LR=1e-5
    HEADER_LR=1e-5
    REQUIRE_FRESH_OUTPUT=true
    LORA_ADAPTOR_CKPT="${PILOT_STEP5_CKPT}"
    ;;
  stratified_multitask)
    EXP_NAME=RA-LOOP_spatial_stratified_multitask
    VARIANT_NAME=four_task_35step_k8_h220_fixed_l2_0p1_stratified_lr1e5_step5_warmstart
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/ra_loop_spatial_stratified_multitask"
    N_STEPS=35
    SAVE_INTERVAL=5
    MODEL_LR=1e-5
    HEADER_LR=1e-5
    REQUIRE_FRESH_OUTPUT=true
    TRAIN_SHUFFLE=true
    LORA_ADAPTOR_CKPT="${PILOT_STEP5_CKPT}"
    TASK_NAMES_OVERRIDE='[pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate,pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate]'
    ;;
  stratified_lambda0_multitask)
    EXP_NAME=RA-LOOP_spatial_stratified_lambda0_multitask
    VARIANT_NAME=four_task_35step_k8_h220_fixed_l2_0p1_stratified_lambda0_lr1e5_step5_warmstart
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/ra_loop_spatial_stratified_lambda0_multitask"
    N_STEPS=35
    SAVE_INTERVAL=5
    MODEL_LR=1e-5
    HEADER_LR=1e-5
    REQUIRE_FRESH_OUTPUT=true
    TRAIN_SHUFFLE=true
    RECOVERY_LAMBDA=0.0
    LORA_ADAPTOR_CKPT="${PILOT_STEP5_CKPT}"
    TASK_NAMES_OVERRIDE='[pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate,pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate]'
    ;;
  stratified_lambda025_multitask)
    EXP_NAME=RA-LOOP_spatial_stratified_lambda025_multitask
    VARIANT_NAME=four_task_35step_k8_h220_fixed_l2_0p1_stratified_lambda0p25_lr1e5_step5_warmstart
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/ra_loop_spatial_stratified_lambda025_multitask"
    N_STEPS=35
    SAVE_INTERVAL=5
    MODEL_LR=1e-5
    HEADER_LR=1e-5
    REQUIRE_FRESH_OUTPUT=true
    TRAIN_SHUFFLE=true
    RECOVERY_LAMBDA=0.25
    LORA_ADAPTOR_CKPT="${PILOT_STEP5_CKPT}"
    TASK_NAMES_OVERRIDE='[pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate,pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate]'
    ;;
  counterfactual_smoke)
    EXP_NAME=RA-LOOP_spatial_counterfactual_smoke
    VARIANT_NAME=one_task_2step_k8_h220_fixed_l2_0p1_cra_npc_cal1_step5_warmstart
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/ra_loop_spatial_counterfactual_smoke"
    N_STEPS=2
    SAVE_INTERVAL=9999
    MODEL_LR=1e-5
    HEADER_LR=1e-5
    REQUIRE_FRESH_OUTPUT=true
    RECOVERY_LAMBDA=0.0
    ADVANTAGE_MODE=counterfactual_constrained
    NOMINAL_CALIBRATION_BATCHES=1
    LORA_ADAPTOR_CKPT="${PILOT_STEP5_CKPT}"
    TASK_NAMES_OVERRIDE='[pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate]'
    ;;
  counterfactual_gate)
    EXP_NAME=RA-LOOP_spatial_counterfactual_gate
    VARIANT_NAME=four_task_50step_k8_h220_fixed_l2_0p1_cra_npc_cal3_step5_warmstart
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/ra_loop_spatial_counterfactual_gate"
    N_STEPS=50
    SAVE_INTERVAL=10
    MODEL_LR=1e-5
    HEADER_LR=1e-5
    REQUIRE_FRESH_OUTPUT=true
    TRAIN_SHUFFLE=true
    RECOVERY_LAMBDA=0.0
    ADVANTAGE_MODE=counterfactual_constrained
    NOMINAL_CALIBRATION_BATCHES=3
    LORA_ADAPTOR_CKPT="${PILOT_STEP5_CKPT}"
    TASK_NAMES_OVERRIDE='[pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate,pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate]'
    ;;
  fulltask_seed10000)
    EXP_NAME=RA-LOOP_spatial_fulltask_seed10000
    VARIANT_NAME=ten_task_100step_k8_h220_fixed_l2_0p1_stratified_lambda0p5_lr1e5_step5_warmstart
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/ra_loop_spatial_fulltask_seed10000"
    N_STEPS=100
    SAVE_INTERVAL=10
    MODEL_LR=1e-5
    HEADER_LR=1e-5
    REQUIRE_FRESH_OUTPUT=true
    TRAIN_SHUFFLE=true
    RECOVERY_LAMBDA=0.5
    DEMOS_PER_ENV=50
    TASK_ROLLOUTS_PER_ENV=50
    TRAIN_SEED=10000
    PERTURB_SEED=20260720
    LORA_ADAPTOR_CKPT="${PILOT_STEP5_CKPT}"
    TASK_NAMES_OVERRIDE='[pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate,pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate,pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate,pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate,pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate]'
    ;;
  fulltask_seed20000)
    EXP_NAME=RA-LOOP_spatial_fulltask_seed20000
    VARIANT_NAME=ten_task_100step_k8_h220_fixed_l2_0p1_stratified_lambda0p5_lr1e5_step5_warmstart
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/ra_loop_spatial_fulltask_seed20000"
    N_STEPS=100
    SAVE_INTERVAL=10
    MODEL_LR=1e-5
    HEADER_LR=1e-5
    REQUIRE_FRESH_OUTPUT=true
    TRAIN_SHUFFLE=true
    RECOVERY_LAMBDA=0.5
    DEMOS_PER_ENV=50
    TASK_ROLLOUTS_PER_ENV=50
    TRAIN_SEED=20000
    PERTURB_SEED=20270720
    LORA_ADAPTOR_CKPT="${PILOT_STEP5_CKPT}"
    TASK_NAMES_OVERRIDE='[pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate,pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate,pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate,pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate,pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate]'
    ;;
  fulltask_lambda0_seed10000)
    EXP_NAME=RA-LOOP_spatial_fulltask_lambda0_seed10000
    VARIANT_NAME=ten_task_100step_k8_h220_fixed_l2_0p1_stratified_lambda0_lr1e5_step5_warmstart
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/ra_loop_spatial_fulltask_lambda0_seed10000"
    N_STEPS=100
    SAVE_INTERVAL=10
    MODEL_LR=1e-5
    HEADER_LR=1e-5
    REQUIRE_FRESH_OUTPUT=true
    TRAIN_SHUFFLE=true
    RECOVERY_LAMBDA=0.0
    DEMOS_PER_ENV=50
    TASK_ROLLOUTS_PER_ENV=50
    TRAIN_SEED=10000
    PERTURB_SEED=20260720
    LORA_ADAPTOR_CKPT="${PILOT_STEP5_CKPT}"
    TASK_NAMES_OVERRIDE='[pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate,pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate,pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate,pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate,pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate]'
    ;;
  fulltask_lambda0_seed20000)
    EXP_NAME=RA-LOOP_spatial_fulltask_lambda0_seed20000
    VARIANT_NAME=ten_task_100step_k8_h220_fixed_l2_0p1_stratified_lambda0_lr1e5_step5_warmstart
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/ra_loop_spatial_fulltask_lambda0_seed20000"
    N_STEPS=100
    SAVE_INTERVAL=10
    MODEL_LR=1e-5
    HEADER_LR=1e-5
    REQUIRE_FRESH_OUTPUT=true
    TRAIN_SHUFFLE=true
    RECOVERY_LAMBDA=0.0
    DEMOS_PER_ENV=50
    TASK_ROLLOUTS_PER_ENV=50
    TRAIN_SEED=20000
    PERTURB_SEED=20270720
    LORA_ADAPTOR_CKPT="${PILOT_STEP5_CKPT}"
    TASK_NAMES_OVERRIDE='[pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate,pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate,pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate,pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate,pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate]'
    ;;
  afternoon_multitask)
    EXP_NAME=RA-LOOP_spatial_robot_init_afternoon_multitask
    VARIANT_NAME=four_task_35step_k8_h220_fixed_l2_0p1_recovery_lr1e5_step5_warmstart
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/ra_loop_spatial_afternoon_multitask"
    N_STEPS=35
    SAVE_INTERVAL=5
    MODEL_LR=1e-5
    HEADER_LR=1e-5
    REQUIRE_FRESH_OUTPUT=true
    TRAIN_SHUFFLE=true
    LORA_ADAPTOR_CKPT="${PILOT_STEP5_CKPT}"
    TASK_NAMES_OVERRIDE='[pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate,pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate,pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate]'
    ;;
  *)
    echo "Unknown RA_LOOP_TRAIN_PROFILE: ${RUN_PROFILE}" >&2
    exit 2
    ;;
esac

OVERRIDES=(
  exp_name="${EXP_NAME}"
  variant_name="${VARIANT_NAME}"
  seed="${TRAIN_SEED}"
  make_unique_experiment_dir=true
  paths.data_prefix="${DATA_PATH}"
  paths.output_prefix="${OUTPUT_PATH}"
  task.suite_name=libero_spatial
  task.dataset.suite_name=.
  task.task_names_to_use="${TASK_NAMES_OVERRIDE}"
  task.demos_per_env="${DEMOS_PER_ENV}"
  task.rollouts_per_env="${TASK_ROLLOUTS_PER_ENV}"
  train_dataloader.batch_size=1
  train_dataloader.shuffle="${TRAIN_SHUFFLE}"
  train_dataloader.num_workers=0
  train_dataloader.persistent_workers=false
  train_dataloader.pin_memory=false
  train_dataloader.multiprocessing_context=null
  training.gradient_accumulation_steps=1
  training.n_steps="${N_STEPS}"
  training.rollout_steps=1
  training.save_interval="${SAVE_INTERVAL}"
  training.log_interval=1
  training.use_tqdm=true
  rollout.enabled=false
  algo.env_runner._target_=ra_loop.ript_compat.InProcessOpenVLAOFTLiberoRunner
  algo.rollout_generator_factory._target_=ra_loop.ript_recovery.RobotInitRecoveryRolloutGenerator
  +algo.rollout_generator_factory.robot_init_strength=0.1
  +algo.rollout_generator_factory.robot_init_sampling_mode=fixed_l2
  +algo.rollout_generator_factory.perturb_seed="${PERTURB_SEED}"
  algo.rl_optimizer_factory._target_=ra_loop.ript_recovery.RobotInitRecoveryOptimizer
  +algo.rl_optimizer_factory.advantage_mode="${ADVANTAGE_MODE}"
  +algo.rl_optimizer_factory.nominal_allowed_drop="${NOMINAL_ALLOWED_DROP}"
  +algo.rl_optimizer_factory.nominal_ema_decay="${NOMINAL_EMA_DECAY}"
  +algo.rl_optimizer_factory.nominal_dual_learning_rate="${NOMINAL_DUAL_LR}"
  +algo.rl_optimizer_factory.nominal_initial_multiplier="${NOMINAL_INITIAL_MULTIPLIER}"
  +algo.rl_optimizer_factory.nominal_max_multiplier="${NOMINAL_MAX_MULTIPLIER}"
  +algo.rl_optimizer_factory.nominal_calibration_batches="${NOMINAL_CALIBRATION_BATCHES}"
  reward_function._target_=ra_loop.ript_recovery.RobotInitRecoveryReward
  +reward_function.lambda_recovery="${RECOVERY_LAMBDA}"
  algo.rloo_batch_size=8
  algo.rollouts_per_env=8
  algo.num_parallel_envs=1
  algo.max_episode_length=220
  algo.enable_dynamic_sampling=false
  algo.use_val_init=false
  algo.mix_val_init_in_rloo=false
  algo.gradient_accumulation_steps=1
  algo.num_ppo_epochs=1
  algo.max_step_batch_size=1
  algo.scale_factor=5.0
  algo.lr="${MODEL_LR}"
  algo.header_lr="${HEADER_LR}"
  algo.fix_scale_head=true
  algo.checkpoint_path="${MODEL_PATH}"
  algo.header_checkpoint="${HEADER_PATH}"
  algo.lora_adaptor_ckpt="${LORA_ADAPTOR_CKPT}"
  logging.mode=disabled
  logging.resume=false
  logging.save_code=false
)

COMMAND=(
  "${TORCHRUN_BIN}"
  --standalone
  --nproc_per_node=1
  train_ript_openvla_oft.py
  --config-name=train_rl_openvla_oft_all_task_spatial.yaml
  "${OVERRIDES[@]}"
)

export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/.libero_official"
export PYTHONPATH="${PROJECT_ROOT}:${OFFICIAL_LIBERO_ROOT}:${RIPT_ROOT}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR=/tmp/ra_loop_numba_cache
export MPLCONFIGDIR=/tmp/ra_loop_mpl_cache

for required in \
  "${PYTHON_BIN}" \
  "${TORCHRUN_BIN}" \
  "${RIPT_ROOT}/train_ript_openvla_oft.py" \
  "${MODEL_PATH}/config.json" \
  "${HEADER_PATH}" \
  "${DATA_PATH}/libero_spatial" \
  "${PROJECT_ROOT}/ra_loop/ript_recovery.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path is missing: ${required}" >&2
    exit 3
  fi
done

if [[ "${LORA_ADAPTOR_CKPT}" != null ]]; then
  for required in \
    "${LORA_ADAPTOR_CKPT}/adapter_config.json" \
    "${LORA_ADAPTOR_CKPT}/adapter_model.safetensors" \
    "${LORA_ADAPTOR_CKPT}/openvla_headers.pt"; do
    if [[ ! -f "${required}" ]]; then
      echo "Required warm-start file is missing: ${required}" >&2
      exit 3
    fi
  done
fi

if [[ $# -eq 0 ]]; then
  export CUDA_VISIBLE_DEVICES=''
  echo "CPU-only RA learning-probe preflight; no dataset/env/model/CUDA creation"
  cd "${RIPT_ROOT}"
  "${PYTHON_BIN}" train_ript_openvla_oft.py \
    --config-name=train_rl_openvla_oft_all_task_spatial.yaml \
    --cfg job --resolve "${OVERRIDES[@]}"
  cd "${PROJECT_ROOT}"
  exec "${PYTHON_BIN}" scripts/preflight_ra_hydra_factories.py "${OVERRIDES[@]}"
fi

if [[ $# -eq 1 && "$1" == "--print-command" ]]; then
  echo "Dry command only; no CUDA initialization"
  printf 'CUDA_VISIBLE_DEVICES=<GPU_ID> '
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

if [[ $# -ne 2 || "$1" != "--run" || ! "$2" =~ ^[0-7]$ ]]; then
  echo "Usage: bash $0 [--print-command | --run <GPU_ID>]" >&2
  exit 2
fi
GPU_ID=$2

if [[ "${REQUIRE_FRESH_OUTPUT}" == true && -e "${OUTPUT_PATH}" ]]; then
  echo "Refusing to reuse training output: ${OUTPUT_PATH}" >&2
  exit 5
fi

AVAILABLE_BYTES=$(df --output=avail -B1 "${PROJECT_ROOT}" | tail -n 1)
AVAILABLE_BYTES=${AVAILABLE_BYTES//[[:space:]]/}
if [[ ! "${AVAILABLE_BYTES}" =~ ^[0-9]+$ ]] || (( AVAILABLE_BYTES < 20000000000 )); then
  echo "Refusing training: less than 20 GB available at ${PROJECT_ROOT}" >&2
  exit 6
fi

GPU_STATUS=$(nvidia-smi --id="${GPU_ID}" \
  --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits)
IFS=',' read -r GPU_USED GPU_TOTAL GPU_UTIL GPU_TEMP <<< "${GPU_STATUS}"
GPU_USED=${GPU_USED//[[:space:]]/}
GPU_TOTAL=${GPU_TOTAL//[[:space:]]/}
GPU_UTIL=${GPU_UTIL//[[:space:]]/}
GPU_TEMP=${GPU_TEMP//[[:space:]]/}
if (( GPU_USED > 1024 || GPU_UTIL > 10 || GPU_TEMP > 75 )); then
  echo "Refusing GPU ${GPU_ID}: used=${GPU_USED}/${GPU_TOTAL} MiB util=${GPU_UTIL}% temp=${GPU_TEMP}C" >&2
  exit 4
fi

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export NCCL_TIMEOUT=108000

echo "Starting bounded RA-LOOP ${RUN_PROFILE} on physical GPU ${GPU_ID}"
echo "GPU before start: used=${GPU_USED}/${GPU_TOTAL} MiB util=${GPU_UTIL}% temp=${GPU_TEMP}C"
echo "tasks=${TASK_NAMES_OVERRIDE} demos_per_env=${DEMOS_PER_ENV} task_rollouts_per_env=${TASK_ROLLOUTS_PER_ENV}"
echo "steps=${N_STEPS} K=8 pairs=4 horizon=220 fixed_l2=0.1rad lambda_r=${RECOVERY_LAMBDA} advantage_mode=${ADVANTAGE_MODE} scale=5 train_seed=${TRAIN_SEED} perturb_seed=${PERTURB_SEED}"
echo "nominal_allowed_drop=${NOMINAL_ALLOWED_DROP} ema_decay=${NOMINAL_EMA_DECAY} dual_lr=${NOMINAL_DUAL_LR} initial_mu=${NOMINAL_INITIAL_MULTIPLIER} max_mu=${NOMINAL_MAX_MULTIPLIER} calibration_batches_per_task=${NOMINAL_CALIBRATION_BATCHES}"
echo "lora_adaptor_ckpt=${LORA_ADAPTOR_CKPT}"
echo "lr=${MODEL_LR} header_lr=${HEADER_LR} save_interval=${SAVE_INTERVAL} available_bytes=${AVAILABLE_BYTES}"
echo "W&B=disabled periodic_eval=disabled"

cd "${RIPT_ROOT}"
exec "${COMMAND[@]}"
