#!/bin/bash
# =====================================================================
# ra_loop.sh
# ---------------------------------------------------------------------
# 启动 RA-LOOP 训练。相比 vanilla_loop.sh 的差异:
#   1) 用 --config-name=train_ra_loop.yaml 走 RA-LOOP 的 Hydra 配置
#   2) 通过 PYTHONPATH 把 RA-LOOP/code/ 挂进去, 让
#      train_ript_openvla_oft.py 能找到 code.ra_optimizer.RAOptimizer
#      (需要在 RIPT-VLA 官方 train script 里加一行 factory 调用, 见下)
#   3) 训练完成后自动跑 eval/run_libero_orig.sh + eval/run_libero_plus.sh,
#      并把结果打包到 outputs/ra_loop_<TS>/
#
# 用法:
#   bash train/ra_loop.sh <N_GPUS>
# 例:
#   bash train/ra_loop.sh 7
#
# 需要在 repos/ript-vla/train_ript_openvla_oft.py 里改一处（一次性）:
#     # 原:
#     rl_optimizer = RLOptimizerOpenVLAOFT(...)
#     # 改为:
#     from code.ra_optimizer import build_rl_optimizer_from_cfg
#     rl_optimizer = build_rl_optimizer_from_cfg(cfg, rollout_generator, reward_function)
#   同时把 RolloutGenerator 换成 RAPerturbedRolloutGenerator（同一处 factory 里
#   处理 perturbation 分支）。这份 patch 由 setup/step06_patch_ript.sh 施加,
#   若尚未存在请先创建（TODO 见 PLAN.md §6.3）。
#
# 期望结果（Milestone Gate C）:
#   * mean_R_consistency 从 -0.10 上升至 -0.03 附近（越接近 0 越一致）
#   * lambda_c_effective 曲线在前 200 步线性上升
#   * eval LIBERO-Plus 提升 5-10pt (baseline 68% → 目标 73-78%)
#   * eval LIBERO-Long 不显著下降（<1pt）
# =====================================================================
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 <N_GPUS>"
    exit 1
fi
NPROC=$1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${PROJECT_ROOT}/../repos/ript-vla"
CFG_ROOT="${PROJECT_ROOT}/config"

# 让 python 能 import code.ra_optimizer
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 检查一次 patch 是否已经施加, 否则拒绝启动 (fail fast)
if ! grep -q "build_rl_optimizer_from_cfg" "${REPO_ROOT}/train_ript_openvla_oft.py"; then
    echo "[FATAL] ${REPO_ROOT}/train_ript_openvla_oft.py 尚未 patch"
    echo "  请先运行: bash setup/step06_patch_ript.sh"
    echo "  该脚本会把 RIPT-VLA 官方 train script 里创建 RLOptimizer 的一行"
    echo "  替换为 code.ra_optimizer.build_rl_optimizer_from_cfg(...) 调用"
    exit 2
fi

# 读取 checkpoint 路径
CHECKPOINT_PATH=$(python -c "
import yaml, pathlib
d = yaml.safe_load(pathlib.Path('${CFG_ROOT}/paths.yaml').read_text())
print(d['models']['openvla_oft_libero_long_sft'])
")
HEADER_CHECKPOINT=$(python -c "
import yaml, pathlib
d = yaml.safe_load(pathlib.Path('${CFG_ROOT}/paths.yaml').read_text())
print(d['models']['ript_vla_scale_header_long'])
")
LORA_ADAPTOR=$(python -c "
import yaml, pathlib
d = yaml.safe_load(pathlib.Path('${CFG_ROOT}/paths.yaml').read_text())
print(d['models']['ript_vla_lora_long'])
")

echo "==================== ra_loop.sh ===================="
echo "N_GPUS        : ${NPROC}"
echo "REPO_ROOT     : ${REPO_ROOT}"
echo "CFG           : ${CFG_ROOT}/train_ra_loop.yaml"
echo "CKPT          : ${CHECKPOINT_PATH}"
echo "HEADER        : ${HEADER_CHECKPOINT}"
echo "LORA          : ${LORA_ADAPTOR}"
echo "===================================================="

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export NCCL_TIMEOUT=108000
export HYDRA_FULL_ERROR=1

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${PROJECT_ROOT}/outputs/ra_loop_${TS}"
mkdir -p "${OUT_DIR}"

# 让 hydra 输出目录落在项目内, 而不是 repo 内 (方便追踪版本)
export HYDRA_OUTPUT_DIR="${OUT_DIR}"

# 复制配置快照
cp "${CFG_ROOT}/train_ra_loop.yaml"                       "${OUT_DIR}/train_ra_loop.snapshot.yaml"
cp "${CFG_ROOT}/reward_function/robustness_aware.yaml"    "${OUT_DIR}/reward_snapshot.yaml"
cp "${CFG_ROOT}/paths.yaml"                               "${OUT_DIR}/paths.snapshot.yaml"

cd "${REPO_ROOT}"

# --------- 训练 ---------
torchrun --nproc_per_node="${NPROC}" --master_port "${MASTER_PORT}" \
    train_ript_openvla_oft.py \
    --config-path="${CFG_ROOT}" \
    --config-name=train_ra_loop \
    exp_name=RA-LOOP_libero_long_${TS} \
    algo.checkpoint_path="${CHECKPOINT_PATH}" \
    algo.header_checkpoint="${HEADER_CHECKPOINT}" \
    algo.lora_adaptor_ckpt="${LORA_ADAPTOR}" \
    hydra.run.dir="${OUT_DIR}/hydra" \
    2>&1 | tee "${OUT_DIR}/train.log"

echo "==================== RA-LOOP train done ===================="

# --------- 训练结束后自动评估 ---------
LATEST_CKPT=$(find "${OUT_DIR}" -type d -name "checkpoints" | head -1)
if [[ -z "${LATEST_CKPT}" ]]; then
    echo "[WARN] 找不到 checkpoint 目录, 跳过自动 eval"
    exit 0
fi

echo "==================== eval LIBERO-Long (orig) ===================="
bash "${PROJECT_ROOT}/eval/run_libero_orig.sh" "${LATEST_CKPT}" \
    | tee "${OUT_DIR}/eval_libero_orig.log"

echo "==================== eval LIBERO-Plus (7-dim) ===================="
bash "${PROJECT_ROOT}/eval/run_libero_plus.sh" "${LATEST_CKPT}" \
    | tee "${OUT_DIR}/eval_libero_plus.log"

echo "==================== ALL DONE ===================="
echo "结果目录: ${OUT_DIR}"
echo "关键文件:"
echo "  ${OUT_DIR}/train.log"
echo "  ${OUT_DIR}/eval_libero_orig.log"
echo "  ${OUT_DIR}/eval_libero_plus.log"
