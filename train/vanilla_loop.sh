#!/bin/bash
# =====================================================================
# vanilla_loop.sh
# ---------------------------------------------------------------------
# 目的：跑通 RIPT-VLA + OpenVLA-OFT 的官方 LOOP 训练, 作为 RA-LOOP 的
#      baseline。等价于官方 scripts/openvla_oft/stage_3_ript/libero_long.sh,
#      仅把三个 checkpoint 路径改成 setup/step04_data.sh 下载好的绝对路径,
#      并把 nproc 通过参数暴露出来。
#
# 用法：
#   bash train/vanilla_loop.sh <N_GPUS>
# 例：
#   bash train/vanilla_loop.sh 7            # 用 7 张 4090
#
# 前置条件：
#   1) 已激活 conda env ript_vla_openvla_oft
#   2) 已跑完 setup/step01~step05 且 step05_verify.py 全绿
#   3) config/paths.yaml 已生成
#
# 期望成本：
#   7×4090, bz=24, K=8, n_steps=12  ≈ 8-12 小时（含 rollout, 主要瓶颈）
#
# 期望结果（Milestone Gate B）：
#   * 训练 loss 单调（第 5 步后 ratio 稳定在 0.9-1.1）
#   * mean_rlhf_reward 从 0.30-0.45 上升至 0.60+
#   * eval on LIBERO-Long: 90% -> 92-94% (RIPT-VLA 论文报的增幅)
#   * eval on LIBERO-Plus: 68% -> 68-70% (基本不变, 这就是本项目的 opportunity)
# =====================================================================
set -euo pipefail

# ----- 位置参数 -----
if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 <N_GPUS>"
    exit 1
fi
NPROC=$1

# ----- 项目根 & repo 根 -----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${RIPT_REPO:-${CODE_DIR:-$HOME/code}/ript-vla}"
CFG_ROOT="${PROJECT_ROOT}/config"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$PROJECT_ROOT/.libero}"

# ----- 读取 paths.yaml 里的 checkpoint 路径 -----
# 用 python 解析避免 grep 出错; 若没装 yaml, 就 fallback 到硬编码约定
CHECKPOINT_PATH=$(python -c "
import yaml, pathlib
p = pathlib.Path('${CFG_ROOT}/paths.yaml')
d = yaml.safe_load(p.read_text())
print(d['openvla_oft']['long'])
")
HEADER_CHECKPOINT=$(python -c "
import yaml, pathlib
p = pathlib.Path('${CFG_ROOT}/paths.yaml')
d = yaml.safe_load(p.read_text())
print(d['ript_vla']['scale_headers']['long'])
")
LORA_ADAPTOR=$(python -c "
import yaml, pathlib
p = pathlib.Path('${CFG_ROOT}/paths.yaml')
d = yaml.safe_load(p.read_text())
print(d['ript_vla']['lora_adaptors']['long'])
")

echo "==================== vanilla_loop.sh ===================="
echo "N_GPUS         : ${NPROC}"
echo "REPO_ROOT      : ${REPO_ROOT}"
echo "CHECKPOINT     : ${CHECKPOINT_PATH}"
echo "HEADER_CKPT    : ${HEADER_CHECKPOINT}"
echo "LORA_ADAPTOR   : ${LORA_ADAPTOR}"
echo "=========================================================="

# 找一个空闲端口, 避免多任务同机时冲突
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# NCCL 超时拉长, LOOP rollout 阶段容易触发默认 30min 超时
export NCCL_TIMEOUT=108000
export HYDRA_FULL_ERROR=1

# 结果目录 → 项目内 outputs/vanilla_loop_YYYYMMDD_HHMMSS
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${PROJECT_ROOT}/outputs/vanilla_loop_${TS}"
mkdir -p "${OUT_DIR}"

# 拷贝配置快照, 方便复盘
cp "${CFG_ROOT}/paths.yaml" "${OUT_DIR}/paths.snapshot.yaml"

cd "${REPO_ROOT}"

# --------- 正式启动 ---------
# 官方 libero_long.sh 用的是 RIPT-VLA 自带的 config, 我们只 override 三个 ckpt
torchrun --nproc_per_node="${NPROC}" --master_port "${MASTER_PORT}" \
    train_ript_openvla_oft.py \
    --config-name=train_rl_openvla_oft_all_task_long.yaml \
    exp_name=OpenVLA-OFT_libero_long_train \
    variant_name=bz24_scale5.0_vanilla \
    algo.model_seed=1 \
    train_dataloader.batch_size=24 \
    training.n_steps=12 \
    training.save_interval=1 \
    algo.env_runner.num_parallel_envs=2 \
    algo.model_seed=0 \
    algo.scale_factor=5.0 \
    algo.rollout_training_task_names=[LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket,LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate,LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate,LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket] \
    algo.checkpoint_path="${CHECKPOINT_PATH}" \
    algo.header_checkpoint="${HEADER_CHECKPOINT}" \
    algo.lora_adaptor_ckpt="${LORA_ADAPTOR}" \
    2>&1 | tee "${OUT_DIR}/train.log"

echo "==================== vanilla_loop done ===================="
echo "输出: ${OUT_DIR}"
echo "接着跑 eval:"
echo "  bash eval/run_libero_orig.sh  ${OUT_DIR}/checkpoints"
echo "  bash eval/run_libero_plus.sh  ${OUT_DIR}/checkpoints"
